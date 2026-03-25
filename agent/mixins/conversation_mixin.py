"""Conversation mixin for AIAgent class.

This mixin provides conversation management capabilities including:
- Conversation initialization and state setup
- Preflight context compression
- Main conversation loop with tool calling

Methods extracted from run_agent.py AIAgent class in Phase 5.
"""

import logging
import os
import random
import sys
import time
import uuid
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class _SafeWriter:
    """Transparent stdio wrapper that catches OSError/ValueError from broken pipes.

    When hermes-agent runs as a systemd service, Docker container, or headless
    daemon, the stdout/stderr pipe can become unavailable. Any print() call then raises
    ``OSError: [Errno 5] Input/output error``, which can crash agent setup or
    run_conversation().

    This wrapper delegates all writes to the underlying stream and silently
    catches both OSError and ValueError. It is transparent when the wrapped
    stream is healthy.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def write(self, data):
        try:
            return self._inner.write(data)
        except (OSError, ValueError):
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            self._inner.flush()
        except (OSError, ValueError):
            pass

    def fileno(self):
        return self._inner.fileno()

    def isatty(self):
        try:
            return self._inner.isatty()
        except (OSError, ValueError):
            return False

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _install_safe_stdio() -> None:
    """Wrap stdout/stderr so best-effort console output cannot crash the agent."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and not isinstance(stream, _SafeWriter):
            setattr(sys, stream_name, _SafeWriter(stream))


class ConversationMixin:
    """Mixin providing conversation management methods.

    This mixin is composed into AIAgent to provide
    conversation-related functionality extracted from the monolithic
    run_agent.py implementation.
    """

    def _initialize_conversation(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        task_id: Optional[str] = None,
        stream_callback: Optional[Callable] = None,
        persist_user_message: Optional[str] = None,
    ) -> tuple:
        """
        Initialize conversation state and prepare messages.

        Returns:
            Tuple of (messages, active_system_prompt, effective_task_id, current_turn_user_idx, _should_review_memory, original_user_message)
        """
        # Guard stdio against OSError from broken pipes (systemd/headless/daemon).
        # Installed once, transparent when streams are healthy, prevents crash on write.
        _install_safe_stdio()

        # Store stream callback for _interruptible_api_call to pick up
        self._stream_callback = stream_callback
        self._persist_user_message_idx = None
        self._persist_user_message_override = persist_user_message
        # Generate unique task_id if not provided to isolate VMs between concurrent tasks
        effective_task_id = task_id or str(uuid.uuid4())

        # Reset retry counters and iteration budget at the start of each turn
        # so subagent usage from a previous turn doesn't eat into the next one.
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self._empty_content_retries = 0
        self._incomplete_scratchpad_retries = 0
        self._codex_incomplete_retries = 0
        self._last_content_with_tools = None
        self._mute_post_response = False
        # NOTE: _turns_since_memory and _iters_since_skill are NOT reset here.
        # They are initialized in __init__ and must persist across run_conversation
        # calls so that nudge logic accumulates correctly in CLI mode.
        self.iteration_budget = IterationBudget(self.max_iterations)

        # Initialize conversation (copy to avoid mutating the caller's list)
        messages = list(conversation_history) if conversation_history else []

        # Hydrate todo store from conversation history (gateway creates a fresh
        # AIAgent per message, so the in-memory store is empty -- we need to
        # recover the todo state from the most recent todo tool response in history)
        if conversation_history and not self._todo_store.has_items():
            self._hydrate_todo_store(conversation_history)

        # Prefill messages (few-shot priming) are injected at API-call time only,
        # never stored in the messages list. This keeps them ephemeral: they won't
        # be saved to session DB, session logs, or batch trajectories, but they're
        # automatically re-applied on every API call (including session continuations).

        # Track user turns for memory flush and periodic nudge logic
        self._user_turn_count += 1

        # Preserve the original user message (no nudge injection).
        # Honcho should receive the actual user input, not system nudges.
        original_user_message = (
            persist_user_message if persist_user_message is not None else user_message
        )

        # Track memory nudge trigger (turn-based, checked here).
        # Skill trigger is checked AFTER the agent loop completes, based on
        # how many tool iterations THIS turn used.
        _should_review_memory = False
        if (
            self._memory_nudge_interval > 0
            and "memory" in self.valid_tool_names
            and self._memory_store
        ):
            self._turns_since_memory += 1
            if self._turns_since_memory >= self._memory_nudge_interval:
                _should_review_memory = True
                self._turns_since_memory = 0

        # Honcho prefetch consumption:
        # - First turn: bake into cached system prompt (stable for the session).
        # - Later turns: attach recall to the current-turn user message at
        #   API-call time only (never persisted to history / session DB).
        #
        # This keeps the system-prefix cache stable while still allowing turn N
        # to consume background prefetch results from turn N-1.
        self._honcho_context = ""
        self._honcho_turn_context = ""
        _recall_mode = (
            self._honcho_config.recall_mode if self._honcho_config else "hybrid"
        )
        if self._honcho and self._honcho_session_key and _recall_mode != "tools":
            try:
                prefetched_context = self._honcho_prefetch(original_user_message)
                if prefetched_context:
                    if not conversation_history:
                        self._honcho_context = prefetched_context
                    else:
                        self._honcho_turn_context = prefetched_context
            except Exception as e:
                logger.debug("Honcho prefetch failed (non-fatal): %s", e)

        # Add user message
        user_msg = {"role": "user", "content": user_message}
        messages.append(user_msg)
        current_turn_user_idx = len(messages) - 1
        self._persist_user_message_idx = current_turn_user_idx

        if not self.quiet_mode:
            self._safe_print(
                f"💬 Starting conversation: '{user_message[:60]}{'...' if len(user_message) > 60 else ''}'"
            )

        # ── System prompt (cached per session for prefix caching) ──
        # Built once on first call, reused for all subsequent calls.
        # Only rebuilt after context compression events (which invalidate
        # the cache and reload memory from disk).
        #
        # For continuing sessions (gateway creates a fresh AIAgent per
        # message), we load the stored system prompt from the session DB
        # instead of rebuilding.  Rebuilding would pick up memory changes
        # from disk that the model already knows about (it wrote them!),
        # producing a different system prompt and breaking the Anthropic
        # prefix cache.
        if self._cached_system_prompt is None:
            stored_prompt = None
            if conversation_history and self._session_db:
                try:
                    session_row = self._session_db.get_session(self.session_id)
                    if session_row:
                        stored_prompt = session_row.get("system_prompt") or None
                except Exception:
                    pass  # Fall through to build fresh

            if stored_prompt:
                # Continuing session — reuse the exact system prompt from
                # the previous turn so the Anthropic cache prefix matches.
                self._cached_system_prompt = stored_prompt
            else:
                # First turn of a new session — build from scratch.
                self._cached_system_prompt = self._build_system_prompt(system_message)
                # Bake Honcho context into the prompt so it's stable for
                # the entire session (not re-fetched per turn).
                if self._honcho_context:
                    self._cached_system_prompt = (
                        self._cached_system_prompt + "\n\n" + self._honcho_context
                    ).strip()
                # Store the system prompt snapshot in SQLite
                if self._session_db:
                    try:
                        self._session_db.update_system_prompt(
                            self.session_id, self._cached_system_prompt
                        )
                    except Exception as e:
                        logger.debug("Session DB update_system_prompt failed: %s", e)

        active_system_prompt = self._cached_system_prompt
        return (
            messages,
            active_system_prompt,
            effective_task_id,
            current_turn_user_idx,
            _should_review_memory,
            original_user_message,
        )

    def _perform_preflight_compression(
        self,
        messages: List[Dict[str, Any]],
        active_system_prompt: str,
        system_message: Optional[str],
        effective_task_id: str,
    ) -> tuple:
        """
        Perform preflight context compression if needed.

        Returns:
            Tuple of (messages, active_system_prompt)
        """
        # ── Preflight context compression ──
        # Before entering the main loop, check if the loaded conversation
        # history already exceeds the model's context threshold.  This handles
        # cases where a user switches to a model with a smaller context window
        # while having a large existing session — compress proactively rather
        # than waiting for an API error (which might be caught as a non-retryable
        # 4xx and abort the request entirely).
        if (
            self.compression_enabled
            and self.context_compressor
            and len(messages)
            > self.context_compressor.protect_first_n
            + self.context_compressor.protect_last_n
            + 1
        ):
            _sys_tok_est = estimate_tokens_rough(active_system_prompt or "")
            _msg_tok_est = estimate_messages_tokens_rough(messages)
            _preflight_tokens = _sys_tok_est + _msg_tok_est

            if _preflight_tokens >= self.context_compressor.threshold_tokens:
                logger.info(
                    "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                    f"{_preflight_tokens:,}",
                    f"{self.context_compressor.threshold_tokens:,}",
                    self.model,
                    f"{self.context_compressor.context_length:,}",
                )
                if not self.quiet_mode:
                    self._safe_print(
                        f"📦 Preflight compression: ~{_preflight_tokens:,} tokens "
                        f">= {self.context_compressor.threshold_tokens:,} threshold"
                    )
                # May need multiple passes for very large sessions with small
                # context windows (each pass summarises the middle N turns).
                for _pass in range(3):
                    _orig_len = len(messages)
                    messages, active_system_prompt = self._compress_context(
                        messages,
                        system_message,
                        approx_tokens=_preflight_tokens,
                        task_id=effective_task_id,
                    )
                    if len(messages) >= _orig_len:
                        break  # Cannot compress further
                    # Re-estimate after compression
                    _sys_tok_est = estimate_tokens_rough(active_system_prompt or "")
                    _msg_tok_est = estimate_messages_tokens_rough(messages)
                    _preflight_tokens = _sys_tok_est + _msg_tok_est
                    if _preflight_tokens < self.context_compressor.threshold_tokens:
                        break  # Under threshold
        return messages, active_system_prompt

    def _run_conversation_loop(
        self,
        messages: List[Dict[str, Any]],
        active_system_prompt: str,
        effective_task_id: str,
        current_turn_user_idx: int,
        _should_review_memory: bool,
        original_user_message: str,
        sync_honcho: bool,
    ) -> Dict[str, Any]:
        """
        Run the main conversation loop with tool calling.

        Returns:
            Dict: Complete conversation result
        """
        api_call_count = 0
        final_response = None
        interrupted = False
        codex_ack_continuations = 0
        length_continue_retries = 0
        truncated_response_prefix = ""
        compression_attempts = 0

        # Clear any stale interrupt state at start
        self.clear_interrupt()

        while (
            api_call_count < self.max_iterations and self.iteration_budget.remaining > 0
        ):
            # Reset per-turn checkpoint dedup so each iteration can take one snapshot
            self._checkpoint_mgr.new_turn()

            # Check for interrupt request (e.g., user sent new message)
            if self._interrupt_requested:
                interrupted = True
                if not self.quiet_mode:
                    self._safe_print(
                        f"\n⚡ Breaking out of tool loop due to interrupt..."
                    )
                break

            api_call_count += 1
            if not self.iteration_budget.consume():
                if not self.quiet_mode:
                    self._safe_print(
                        f"\n⚠️  Session iteration budget exhausted ({self.iteration_budget.max_total} total across agent + subagents)"
                    )
                break

            # Fire step_callback for gateway hooks (agent:step event)
            if self.step_callback is not None:
                try:
                    prev_tools = []
                    for _m in reversed(messages):
                        if _m.get("role") == "assistant" and _m.get("tool_calls"):
                            prev_tools = [
                                tc["function"]["name"]
                                for tc in _m["tool_calls"]
                                if isinstance(tc, dict)
                            ]
                            break
                    self.step_callback(api_call_count, prev_tools)
                except Exception as _step_err:
                    logger.debug(
                        "step_callback error (iteration %s): %s",
                        api_call_count,
                        _step_err,
                    )

            # Track tool-calling iterations for skill nudge.
            # Counter resets whenever skill_manage is actually used.
            if (
                self._skill_nudge_interval > 0
                and "skill_manage" in self.valid_tool_names
            ):
                self._iters_since_skill += 1

            # Prepare messages for API call
            # If we have an ephemeral system prompt, prepend it to the messages
            # Note: Reasoning is embedded in content via <think> tags for trajectory storage.
            # However, providers like Moonshot AI require a separate 'reasoning_content' field
            # on assistant messages with tool_calls. We handle both cases here.
            api_messages = []
            for idx, msg in enumerate(messages):
                api_msg = msg.copy()

                if (
                    idx == current_turn_user_idx
                    and msg.get("role") == "user"
                    and self._honcho_turn_context
                ):
                    api_msg["content"] = _inject_honcho_turn_context(
                        api_msg.get("content", ""), self._honcho_turn_context
                    )

                # For ALL assistant messages, pass reasoning back to the API
                # This ensures multi-turn reasoning context is preserved
                if msg.get("role") == "assistant":
                    reasoning_text = msg.get("reasoning")
                    if reasoning_text:
                        # Add reasoning_content for API compatibility (Moonshot AI, Novita, OpenRouter)
                        api_msg["reasoning_content"] = reasoning_text

                # Remove 'reasoning' field - it's for trajectory storage only
                # We've copied it to 'reasoning_content' for the API above
                if "reasoning" in api_msg:
                    api_msg.pop("reasoning")
                # Remove finish_reason - not accepted by strict APIs (e.g. Mistral)
                if "finish_reason" in api_msg:
                    api_msg.pop("finish_reason")
                # Strip Codex Responses API fields (call_id, response_item_id) for
                # strict providers like Mistral that reject unknown fields with 422.
                # Uses new dicts so the internal messages list retains the fields
                # for Codex Responses compatibility.
                if "api.mistral.ai" in self._base_url_lower:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                # Keep 'reasoning_details' - OpenRouter uses this for multi-turn reasoning context
                # The signature field helps maintain reasoning continuity
                api_messages.append(api_msg)

            # Build the final system message: cached prompt + ephemeral system prompt.
            # Ephemeral additions are API-call-time only (not persisted to session DB).
            # Honcho later-turn recall is intentionally kept OUT of the system prompt
            # so the stable cache prefix remains unchanged.
            effective_system = active_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (
                    effective_system + "\n\n" + self.ephemeral_system_prompt
                ).strip()
            if effective_system:
                api_messages = [
                    {"role": "system", "content": effective_system}
                ] + api_messages

            # Inject ephemeral prefill messages right after the system prompt
            # but before conversation history. Same API-call-time-only pattern.
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            # Apply Anthropic prompt caching for Claude models via OpenRouter.
            # Auto-detected: if model name contains "claude" and base_url is OpenRouter,
            # inject cache_control breakpoints (system + last 3 messages) to reduce
            # input token costs by ~75% on multi-turn conversations.
            if self._use_prompt_caching:
                api_messages = apply_anthropic_cache_control(
                    api_messages,
                    cache_ttl=self._cache_ttl,
                    native_anthropic=(self.api_mode == "anthropic_messages"),
                )

            # Safety net: strip orphaned tool results / add stubs for missing
            # results before sending to the API.  Runs unconditionally — not
            # gated on context_compressor — so orphans from session loading or
            # manual message manipulation are always caught.
            api_messages = self._sanitize_api_messages(api_messages)

            # Calculate approximate request size for logging
            total_chars = sum(len(str(msg)) for msg in api_messages)
            approx_tokens = total_chars // 4  # Rough estimate: 4 chars per token

            # Thinking spinner for quiet mode (animated during API call)
            thinking_spinner = None

            if not self.quiet_mode:
                self._vprint(
                    f"\n{self.log_prefix}🔄 Making API call #{api_call_count}/{self.max_iterations}..."
                )
                self._vprint(
                    f"{self.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)"
                )
                self._vprint(
                    f"{self.log_prefix}   🔧 Available tools: {len(self.tools) if self.tools else 0}"
                )
            else:
                # Animated thinking spinner in quiet mode
                face = random.choice(KawaiiSpinner.KAWAII_THINKING)
                verb = random.choice(KawaiiSpinner.THINKING_VERBS)
                if self.thinking_callback:
                    # CLI TUI mode: use prompt_toolkit widget instead of raw spinner
                    # (works in both streaming and non-streaming modes)
                    self.thinking_callback(f"{face} {verb}...")
                elif not self._has_stream_consumers():
                    # Raw KawaiiSpinner only when no streaming consumers
                    # (would conflict with streamed token output)
                    spinner_type = random.choice(
                        ["brain", "sparkle", "pulse", "moon", "star"]
                    )
                    thinking_spinner = KawaiiSpinner(
                        f"{face} {verb}...", spinner_type=spinner_type
                    )
                    thinking_spinner.start()

            # Log request details if verbose
            if self.verbose_logging:
                logging.debug(
                    f"API Request - Model: {self.model}, Messages: {len(messages)}, Tools: {len(self.tools) if self.tools else 0}"
                )
                logging.debug(
                    f"Last message role: {messages[-1]['role'] if messages else 'none'}"
                )
                logging.debug(f"Total message size: ~{approx_tokens:,} tokens")

            api_start_time = time.time()
            retry_count = 0
            max_retries = 3
            max_compression_attempts = 3
            codex_auth_retry_attempted = False
            anthropic_auth_retry_attempted = False
            nous_auth_retry_attempted = False
            restart_with_compressed_messages = False
            restart_with_length_continuation = False

            finish_reason = "stop"
            response = None  # Guard against UnboundLocalError if all retries fail

            while retry_count < max_retries:
                try:
                    api_kwargs = self._build_api_kwargs(api_messages)
                    if self.api_mode == "codex_responses":
                        api_kwargs = self._preflight_codex_api_kwargs(
                            api_kwargs, allow_stream=False
                        )

                    if os.getenv("HERMES_DUMP_REQUESTS", "").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }:
                        self._dump_api_request_debug(api_kwargs, reason="preflight")

                    # Check prompt cache for potential hits before making API call
                    cached_response = self._prompt_cache.check_and_apply(
                        messages, api_kwargs, self.model, self.provider
                    )
                    if cached_response is not None:
                        response = cached_response
                        cached_hit = True
                    else:
                        if self._has_stream_consumers():
                            # Streaming path: fire delta callbacks for real-time
                            # token delivery to CLI display, gateway, or TTS.
                            def _stop_spinner():
                                nonlocal thinking_spinner
                                if thinking_spinner:
                                    thinking_spinner.stop("")
                                    thinking_spinner = None
                                if self.thinking_callback:
                                    self.thinking_callback("")

                            response = self._interruptible_streaming_api_call(
                                api_kwargs, on_first_delta=_stop_spinner
                            )
                        else:
                            response = self._interruptible_api_call(api_kwargs)
                        cached_hit = False

                    api_duration = time.time() - api_start_time

                    # Record cache statistics if applicable
                    if cached_hit:
                        from agent.token_stats import record_cache_hit

                        estimated_tokens = (
                            self.context_compressor.last_prompt_tokens
                            if hasattr(self, "context_compressor")
                            else 0
                        )
                        record_cache_hit(self.provider, estimated_tokens, self.model)

                    # Stop thinking spinner silently -- the response box or tool
                    # execution messages that follow are more informative.
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")

                    if not self.quiet_mode:
                        self._vprint(
                            f"{self.log_prefix}⏱️  API call completed in {api_duration:.2f}s"
                        )

                    if self.verbose_logging:
                        # Log response with provider info if available
                        resp_model = (
                            getattr(response, "model", "N/A") if response else "N/A"
                        )
                        logging.debug(
                            f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}"
                        )

                    # Validate response shape before proceeding
                    response_invalid = False
                    error_details = []
                    if self.api_mode == "codex_responses":
                        output_items = (
                            getattr(response, "output", None)
                            if response is not None
                            else None
                        )
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        elif not isinstance(output_items, list):
                            response_invalid = True
                            error_details.append("response.output is not a list")
                        elif len(output_items) == 0:
                            response_invalid = True
                            error_details.append("response.output is empty")
                    elif self.api_mode == "anthropic_messages":
                        content_blocks = (
                            getattr(response, "content", None)
                            if response is not None
                            else None
                        )
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        elif not isinstance(content_blocks, list):
                            response_invalid = True
                            error_details.append("response.content is not a list")
                        elif len(content_blocks) == 0:
                            response_invalid = True
                            error_details.append("response.content is empty")
                    else:
                        if (
                            response is None
                            or not hasattr(response, "choices")
                            or response.choices is None
                            or len(response.choices) == 0
                        ):
                            response_invalid = True
                            if response is None:
                                error_details.append("response is None")
                            elif not hasattr(response, "choices"):
                                error_details.append(
                                    "response has no 'choices' attribute"
                                )
                            elif response.choices is None:
                                error_details.append("response.choices is None")
                            else:
                                error_details.append("response.choices is empty")

                    if response_invalid:
                        # Stop spinner before printing error messages
                        if thinking_spinner:
                            thinking_spinner.stop(f"(´;ω;`) oops, retrying...")
                            thinking_spinner = None
                        if self.thinking_callback:
                            self.thinking_callback("")

                        # This is often rate limiting or provider returning malformed response
                        retry_count += 1

                        # Eager fallback: empty/malformed responses are a common
                        # rate-limit symptom.  Switch to fallback immediately
                        # rather than retrying with extended backoff.
                        if (
                            not self._fallback_activated
                            and self._try_activate_fallback()
                        ):
                            retry_count = 0
                            continue

                        # Check for error field in response (some providers include this)
                        error_msg = "Unknown"
                        provider_name = "Unknown"
                        if response and hasattr(response, "error") and response.error:
                            error_msg = str(response.error)
                            # Try to extract provider from error metadata
                            if (
                                hasattr(response.error, "metadata")
                                and response.error.metadata
                            ):
                                provider_name = response.error.metadata.get(
                                    "provider_name", "Unknown"
                                )
                        elif (
                            response
                            and hasattr(response, "message")
                            and response.message
                        ):
                            error_msg = str(response.message)

                        # Try to get provider from model field (OpenRouter often returns actual model used)
                        if (
                            provider_name == "Unknown"
                            and response
                            and hasattr(response, "model")
                            and response.model
                        ):
                            provider_name = f"model={response.model}"

                        # Check for x-openrouter-provider or similar metadata
                        if provider_name == "Unknown" and response:
                            # Log all response attributes for debugging
                            resp_attrs = {
                                k: str(v)[:100]
                                for k, v in vars(response).items()
                                if not k.startswith("_")
                            }
                            if self.verbose_logging:
                                logging.debug(
                                    f"Response attributes for invalid response: {resp_attrs}"
                                )

                        self._vprint(
                            f"{self.log_prefix}⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}",
                            force=True,
                        )
                        self._vprint(
                            f"{self.log_prefix}   🏢 Provider: {provider_name}",
                            force=True,
                        )
                        self._vprint(
                            f"{self.log_prefix}   📝 Provider message: {error_msg[:200]}",
                            force=True,
                        )
                        self._vprint(
                            f"{self.log_prefix}   ⏱️  Response time: {api_duration:.2f}s (fast response often indicates rate limiting)",
                            force=True,
                        )

                        if retry_count >= max_retries:
                            # Try fallback before giving up
                            if self._try_activate_fallback():
                                retry_count = 0
                                continue
                            self._vprint(
                                f"{self.log_prefix}❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.",
                                force=True,
                            )
                            logging.error(
                                f"{self.log_prefix}Invalid API response after {max_retries} retries."
                            )
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": "Invalid API response shape. Likely rate limited or malformed provider response.",
                                "failed": True,  # Mark as failure for filtering
                            }

                        # Longer backoff for rate limiting (likely cause of None choices)
                        wait_time = min(
                            5 * (2 ** (retry_count - 1)), 120
                        )  # 5s, 10s, 20s, 40s, 80s, 120s
                        self._vprint(
                            f"{self.log_prefix}⏳ Retrying in {wait_time}s (extended backoff for possible rate limit)...",
                            force=True,
                        )
                        logging.warning(
                            f"Invalid API response (retry {retry_count}/{max_retries}): {', '.join(error_details)} | Provider: {provider_name}"
                        )

                        # Sleep in small increments to stay responsive to interrupts
                        sleep_end = time.time() + wait_time
                        while time.time() < sleep_end:
                            if self._interrupt_requested:
                                self._vprint(
                                    f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.",
                                    force=True,
                                )
                                self._persist_session(messages, conversation_history)
                                self.clear_interrupt()
                                return {
                                    "final_response": f"Operation interrupted: retrying API call after rate limit (retry {retry_count}/{max_retries}).",
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "interrupted": True,
                                }
                            time.sleep(0.2)
                        continue  # Retry the API call

                    # Check finish_reason before proceeding
                    if self.api_mode == "codex_responses":
                        status = getattr(response, "status", None)
                        incomplete_details = getattr(
                            response, "incomplete_details", None
                        )
                        incomplete_reason = None
                        if isinstance(incomplete_details, dict):
                            incomplete_reason = incomplete_details.get("reason")
                        else:
                            incomplete_reason = getattr(
                                incomplete_details, "reason", None
                            )
                        if status == "incomplete" and incomplete_reason in {
                            "max_output_tokens",
                            "length",
                        }:
                            finish_reason = "length"
                        else:
                            finish_reason = "stop"
                    elif self.api_mode == "anthropic_messages":
                        stop_reason_map = {
                            "end_turn": "stop",
                            "tool_use": "tool_calls",
                            "max_tokens": "length",
                            "stop_sequence": "stop",
                        }
                        finish_reason = stop_reason_map.get(
                            response.stop_reason, "stop"
                        )
                    else:
                        finish_reason = response.choices[0].finish_reason

                    if finish_reason == "length":
                        self._vprint(
                            f"{self.log_prefix}⚠️  Response truncated (finish_reason='length') - model hit max output tokens",
                            force=True,
                        )

                        if self.api_mode == "chat_completions":
                            assistant_message = response.choices[0].message
                            if not assistant_message.tool_calls:
                                length_continue_retries += 1
                                interim_msg = self._build_assistant_message(
                                    assistant_message, finish_reason
                                )
                                messages.append(interim_msg)
                                if assistant_message.content:
                                    truncated_response_prefix += (
                                        assistant_message.content
                                    )

                                if length_continue_retries < 3:
                                    self._vprint(
                                        f"{self.log_prefix}↻ Requesting continuation "
                                        f"({length_continue_retries}/3)..."
                                    )
                                    continue_msg = {
                                        "role": "user",
                                        "content": (
                                            "[System: Your previous response was truncated by the output "
                                            "length limit. Continue exactly where you left off. Do not "
                                            "restart or repeat prior text. Finish the answer directly.]"
                                        ),
                                    }
                                    messages.append(continue_msg)
                                    self._session_messages = messages
                                    self._save_session_log(messages)
                                    restart_with_length_continuation = True
                                    break

                                partial_response = self._strip_think_blocks(
                                    truncated_response_prefix
                                ).strip()
                                self._cleanup_task_resources(effective_task_id)
                                self._persist_session(messages, conversation_history)
                                return {
                                    "final_response": partial_response or None,
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "partial": True,
                                    "error": "Response remained truncated after 3 continuation attempts",
                                }

                        # If we have prior messages, roll back to last complete state
                        if len(messages) > 1:
                            self._vprint(
                                f"{self.log_prefix}   ⏪ Rolling back to last complete assistant turn"
                            )
                            rolled_back_messages = (
                                self._get_messages_up_to_last_assistant(messages)
                            )

                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)

                            return {
                                "final_response": None,
                                "messages": rolled_back_messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response truncated due to output length limit",
                            }
                        else:
                            # First message was truncated - mark as failed
                            self._vprint(
                                f"{self.log_prefix}❌ First response truncated - cannot recover",
                                force=True,
                            )
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "failed": True,
                                "error": "First response truncated due to output length limit",
                            }

                    # Track actual token usage from response for context management
                    if hasattr(response, "usage") and response.usage:
                        canonical_usage = normalize_usage(
                            response.usage,
                            provider=self.provider,
                            api_mode=self.api_mode,
                        )
                        prompt_tokens = canonical_usage.prompt_tokens
                        completion_tokens = canonical_usage.output_tokens
                        total_tokens = canonical_usage.total_tokens
                        usage_dict = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        }
                        self.context_compressor.update_from_response(usage_dict)

                        # Cache discovered context length after successful call
                        if self.context_compressor._context_probed:
                            ctx = self.context_compressor.context_length
                            save_context_length(self.model, self.base_url, ctx)
                            self._safe_print(
                                f"{self.log_prefix}💾 Cached context length: {ctx:,} tokens for {self.model}"
                            )
                            self.context_compressor._context_probed = False

                        self.session_prompt_tokens += prompt_tokens
                        self.session_completion_tokens += completion_tokens
                        self.session_total_tokens += total_tokens
                        self.session_api_calls += 1
                        self.session_input_tokens += canonical_usage.input_tokens
                        self.session_output_tokens += canonical_usage.output_tokens
                        self.session_cache_read_tokens += (
                            canonical_usage.cache_read_tokens
                        )
                        self.session_cache_write_tokens += (
                            canonical_usage.cache_write_tokens
                        )
                        self.session_reasoning_tokens += (
                            canonical_usage.reasoning_tokens
                        )

                        cost_result = estimate_usage_cost(
                            self.model,
                            canonical_usage,
                            provider=self.provider,
                            base_url=self.base_url,
                            api_key=getattr(self, "api_key", ""),
                        )
                        if cost_result.amount_usd is not None:
                            self.session_estimated_cost_usd += float(
                                cost_result.amount_usd
                            )
                        self.session_cost_status = cost_result.status
                        self.session_cost_source = cost_result.source

                        # Persist token counts to session DB for /insights.
                        # Gateway sessions persist via session_store.update_session()
                        # after run_conversation returns, so only persist here for
                        # CLI (and other non-gateway) platforms to avoid double-counting.
                        if (
                            self._session_db
                            and self.session_id
                            and getattr(self, "platform", None) == "cli"
                        ):
                            try:
                                self._session_db.update_token_counts(
                                    self.session_id,
                                    input_tokens=canonical_usage.input_tokens,
                                    output_tokens=canonical_usage.output_tokens,
                                    cache_read_tokens=canonical_usage.cache_read_tokens,
                                    cache_write_tokens=canonical_usage.cache_write_tokens,
                                    reasoning_tokens=canonical_usage.reasoning_tokens,
                                    estimated_cost_usd=(
                                        float(cost_result.amount_usd)
                                        if cost_result.amount_usd is not None
                                        else None
                                    ),
                                    cost_status=cost_result.status,
                                    cost_source=cost_result.source,
                                    billing_provider=self.provider,
                                    billing_base_url=self.base_url,
                                    billing_mode=(
                                        "subscription_included"
                                        if cost_result.status == "included"
                                        else None
                                    ),
                                    model=self.model,
                                )
                            except Exception:
                                pass  # never block the agent loop

                        if self.verbose_logging:
                            logging.debug(
                                f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}"
                            )

                        # Log cache hit stats when prompt caching is active
                        if self._use_prompt_caching:
                            if self.api_mode == "anthropic_messages":
                                # Anthropic uses cache_read_input_tokens / cache_creation_input_tokens
                                cached = (
                                    getattr(
                                        response.usage, "cache_read_input_tokens", 0
                                    )
                                    or 0
                                )
                                written = (
                                    getattr(
                                        response.usage, "cache_creation_input_tokens", 0
                                    )
                                    or 0
                                )
                            else:
                                # OpenRouter uses prompt_tokens_details.cached_tokens
                                details = getattr(
                                    response.usage, "prompt_tokens_details", None
                                )
                                cached = (
                                    getattr(details, "cached_tokens", 0) or 0
                                    if details
                                    else 0
                                )
                                written = (
                                    getattr(details, "cache_write_tokens", 0) or 0
                                    if details
                                    else 0
                                )
                            prompt = usage_dict["prompt_tokens"]
                            hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                            if not self.quiet_mode:
                                self._vprint(
                                    f"{self.log_prefix}   💾 Cache: {cached:,}/{prompt:,} tokens ({hit_pct:.0f}% hit, {written:,} written)"
                                )

                    break  # Success, exit retry loop

                except InterruptedError:
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")
                    api_elapsed = time.time() - api_start_time
                    self._vprint(
                        f"{self.log_prefix}⚡ Interrupted during API call.", force=True
                    )
                    self._persist_session(messages, conversation_history)
                    interrupted = True
                    final_response = f"Operation interrupted: waiting for model response ({api_elapsed:.1f}s elapsed)."
                    break

                except Exception as api_error:
                    # Stop spinner before printing error messages
                    if thinking_spinner:
                        thinking_spinner.stop(f"(╥_╥) error, retrying...")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")

                    status_code = getattr(api_error, "status_code", None)
                    if (
                        self.api_mode == "codex_responses"
                        and self.provider == "openai-codex"
                        and status_code == 401
                        and not codex_auth_retry_attempted
                    ):
                        codex_auth_retry_attempted = True
                        if self._try_refresh_codex_client_credentials(force=True):
                            self._vprint(
                                f"{self.log_prefix}🔐 Codex auth refreshed after 401. Retrying request..."
                            )
                            continue
                    if (
                        self.api_mode == "chat_completions"
                        and self.provider == "nous"
                        and status_code == 401
                        and not nous_auth_retry_attempted
                    ):
                        nous_auth_retry_attempted = True
                        if self._try_refresh_nous_client_credentials(force=True):
                            print(
                                f"{self.log_prefix}🔐 Nous agent key refreshed after 401. Retrying request..."
                            )
                            continue
                    if (
                        self.api_mode == "anthropic_messages"
                        and status_code == 401
                        and hasattr(self, "_anthropic_api_key")
                        and not anthropic_auth_retry_attempted
                    ):
                        anthropic_auth_retry_attempted = True
                        from agent.anthropic_adapter import _is_oauth_token

                        if self._try_refresh_anthropic_client_credentials():
                            print(
                                f"{self.log_prefix}🔐 Anthropic credentials refreshed after 401. Retrying request..."
                            )
                            continue
                        # Credential refresh didn't help — show diagnostic info
                        key = self._anthropic_api_key
                        auth_method = (
                            "Bearer (OAuth/setup-token)"
                            if _is_oauth_token(key)
                            else "x-api-key (API key)"
                        )
                        print(
                            f"{self.log_prefix}🔐 Anthropic 401 — authentication failed."
                        )
                        print(f"{self.log_prefix}   Auth method: {auth_method}")
                        print(
                            f"{self.log_prefix}   Token prefix: {key[:12]}..."
                            if key and len(key) > 12
                            else f"{self.log_prefix}   Token: (empty or short)"
                        )
                        print(f"{self.log_prefix}   Troubleshooting:")
                        print(
                            f"{self.log_prefix}     • Check ANTHROPIC_TOKEN in ~/.hermes/.env for Hermes-managed OAuth/setup tokens"
                        )
                        print(
                            f"{self.log_prefix}     • Check ANTHROPIC_API_KEY in ~/.hermes/.env for API keys or legacy token values"
                        )
                        print(
                            f"{self.log_prefix}     • For API keys: verify at https://console.anthropic.com/settings/keys"
                        )
                        print(
                            f"{self.log_prefix}     • For Claude Code: run 'claude /login' to refresh, then retry"
                        )
                        print(
                            f'{self.log_prefix}     • Clear stale keys: hermes config set ANTHROPIC_TOKEN ""'
                        )
                        print(
                            f'{self.log_prefix}     • Legacy cleanup: hermes config set ANTHROPIC_API_KEY ""'
                        )

                    retry_count += 1
                    elapsed_time = time.time() - api_start_time

                    # Enhanced error logging
                    error_type = type(api_error).__name__
                    error_msg = str(api_error).lower()
                    logger.warning(
                        "API call failed (attempt %s/%s) error_type=%s %s error=%s",
                        retry_count,
                        max_retries,
                        error_type,
                        self._client_log_context(),
                        api_error,
                    )

                    _provider = getattr(self, "provider", "unknown")
                    _base = getattr(self, "base_url", "unknown")
                    _model = getattr(self, "model", "unknown")
                    self._vprint(
                        f"{self.log_prefix}⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}",
                        force=True,
                    )
                    self._vprint(
                        f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}",
                        force=True,
                    )
                    self._vprint(
                        f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True
                    )
                    self._vprint(
                        f"{self.log_prefix}   📝 Error: {str(api_error)[:200]}",
                        force=True,
                    )
                    self._vprint(
                        f"{self.log_prefix}   ⏱️  Elapsed: {elapsed_time:.2f}s  Context: {len(api_messages)} msgs, ~{approx_tokens:,} tokens"
                    )

                    # Check for interrupt before deciding to retry
                    if self._interrupt_requested:
                        self._vprint(
                            f"{self.log_prefix}⚡ Interrupt detected during error handling, aborting retries.",
                            force=True,
                        )
                        self._persist_session(messages, conversation_history)
                        self.clear_interrupt()
                        return {
                            "final_response": f"Operation interrupted: handling API error ({error_type}: {str(api_error)[:80]}).",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        }

                    # Check for 413 payload-too-large BEFORE generic 4xx handler.
                    # A 413 is a payload-size error — the correct response is to
                    # compress history and retry, not abort immediately.
                    status_code = getattr(api_error, "status_code", None)

                    # Eager fallback for rate-limit errors (429 or quota exhaustion).
                    # When a fallback model is configured, switch immediately instead
                    # of burning through retries with exponential backoff -- the
                    # primary provider won't recover within the retry window.
                    is_rate_limited = (
                        status_code == 429
                        or "rate limit" in error_msg
                        or "too many requests" in error_msg
                        or "rate_limit" in error_msg
                        or "usage limit" in error_msg
                        or "quota" in error_msg
                    )
                    if is_rate_limited and not self._fallback_activated:
                        if self._try_activate_fallback():
                            retry_count = 0
                            continue

                    is_payload_too_large = (
                        status_code == 413
                        or "request entity too large" in error_msg
                        or "payload too large" in error_msg
                        or "error code: 413" in error_msg
                    )

                    if is_payload_too_large:
                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(
                                f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached for payload-too-large error.",
                                force=True,
                            )
                            logging.error(
                                f"{self.log_prefix}413 compression failed after {max_compression_attempts} attempts."
                            )
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Request payload too large: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True,
                            }
                        if not self.context_compressor:
                            logger.warning(
                                "Context compressor not available, cannot compress for 413 error"
                            )
                            break

                        self._vprint(
                            f"{self.log_prefix}⚠️  Request payload too large (413) — compression attempt {compression_attempts}/{max_compression_attempts}..."
                        )

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages,
                            system_message,
                            approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )

                        if len(messages) < original_len:
                            self._vprint(
                                f"{self.log_prefix}   🗜️  Compressed {original_len} → {len(messages)} messages, retrying..."
                            )
                            time.sleep(2)  # Brief pause between compression retries
                            restart_with_compressed_messages = True
                            break
                        else:
                            self._vprint(
                                f"{self.log_prefix}❌ Payload too large and cannot compress further.",
                                force=True,
                            )
                            logging.error(
                                f"{self.log_prefix}413 payload too large. Cannot compress further."
                            )
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": "Request payload too large (413). Cannot compress further.",
                                "partial": True,
                            }

                    # Check for context-length errors BEFORE generic 4xx handler.
                    # Local backends (LM Studio, Ollama, llama.cpp) often return
                    # HTTP 400 with messages like "Context size has been exceeded"
                    # which must trigger compression, not an immediate abort.
                    is_context_length_error = any(
                        phrase in error_msg
                        for phrase in [
                            "context length",
                            "context size",
                            "maximum context",
                            "token limit",
                            "too many tokens",
                            "reduce the length",
                            "exceeds the limit",
                            "context window",
                            "request entity too large",  # OpenRouter/Nous 413 safety net
                            "prompt is too long",  # Anthropic: "prompt is too long: N tokens > M maximum"
                        ]
                    )

                    # Fallback heuristic: Anthropic sometimes returns a generic
                    # 400 invalid_request_error with just "Error" as the message
                    # when the context is too large.  If the error message is very
                    # short/generic AND the session is large, treat it as a
                    # probable context-length error and attempt compression rather
                    # than aborting.  This prevents an infinite failure loop where
                    # each failed message gets persisted, making the session even
                    # larger. (#1630)
                    if not is_context_length_error and status_code == 400:
                        ctx_len = getattr(
                            getattr(self, "context_compressor", None),
                            "context_length",
                            200000,
                        )
                        is_large_session = (
                            approx_tokens > ctx_len * 0.4 or len(api_messages) > 80
                        )
                        is_generic_error = (
                            len(error_msg.strip()) < 30
                        )  # e.g. just "error"
                        if is_large_session and is_generic_error:
                            is_context_length_error = True
                            self._vprint(
                                f"{self.log_prefix}⚠️  Generic 400 with large session "
                                f"(~{approx_tokens:,} tokens, {len(api_messages)} msgs) — "
                                f"treating as probable context overflow.",
                                force=True,
                            )

                    if is_context_length_error:
                        compressor = self.context_compressor
                        old_ctx = compressor.context_length

                        # Try to parse the actual limit from the error message
                        parsed_limit = parse_context_limit_from_error(error_msg)
                        if parsed_limit and parsed_limit < old_ctx:
                            new_ctx = parsed_limit
                            self._vprint(
                                f"{self.log_prefix}⚠️  Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})",
                                force=True,
                            )
                        else:
                            # Step down to the next probe tier
                            new_ctx = get_next_probe_tier(old_ctx)

                        if new_ctx and new_ctx < old_ctx:
                            compressor.context_length = new_ctx
                            compressor.threshold_tokens = int(
                                new_ctx * compressor.threshold_percent
                            )
                            compressor._context_probed = True
                            self._vprint(
                                f"{self.log_prefix}⚠️  Context length exceeded — stepping down: {old_ctx:,} → {new_ctx:,} tokens",
                                force=True,
                            )
                        else:
                            self._vprint(
                                f"{self.log_prefix}⚠️  Context length exceeded at minimum tier — attempting compression...",
                                force=True,
                            )

                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(
                                f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.",
                                force=True,
                            )
                            logging.error(
                                f"{self.log_prefix}Context compression failed after {max_compression_attempts} attempts."
                            )
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True,
                            }
                        if not self.context_compressor:
                            logger.warning(
                                "Context compressor not available, cannot compress for context length error"
                            )
                            break

                        self._vprint(
                            f"{self.log_prefix}   🗜️  Context compression attempt {compression_attempts}/{max_compression_attempts}..."
                        )

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages,
                            system_message,
                            approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )

                        if (
                            len(messages) < original_len
                            or new_ctx
                            and new_ctx < old_ctx
                        ):
                            if len(messages) < original_len:
                                self._vprint(
                                    f"{self.log_prefix}   🗜️  Compressed {original_len} → {len(messages)} messages, retrying..."
                                )
                            time.sleep(2)  # Brief pause between compression retries
                            restart_with_compressed_messages = True
                            break
                        else:
                            # Can't compress further and already at minimum tier
                            self._vprint(
                                f"{self.log_prefix}❌ Context length exceeded and cannot compress further.",
                                force=True,
                            )
                            self._vprint(
                                f"{self.log_prefix}   💡 The conversation has accumulated too much content.",
                                force=True,
                            )
                            logging.error(
                                f"{self.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot compress further."
                            )
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded ({approx_tokens:,} tokens). Cannot compress further.",
                                "partial": True,
                            }

                    # Check for non-retryable client errors (4xx HTTP status codes).
                    # These indicate a problem with the request itself (bad model ID,
                    # invalid API key, forbidden, etc.) and will never succeed on retry.
                    # Note: 413 and context-length errors are excluded — handled above.
                    # 429 (rate limit) is transient and MUST be retried with backoff.
                    # 529 (Anthropic overloaded) is also transient.
                    # Also catch local validation errors (ValueError, TypeError) — these
                    # are programming bugs, not transient failures.
                    _RETRYABLE_STATUS_CODES = {413, 429, 529}
                    is_local_validation_error = isinstance(
                        api_error, (ValueError, TypeError)
                    )
                    # Detect generic 400s from Anthropic OAuth (transient server-side failures).
                    # Real invalid_request_error responses include a descriptive message;
                    # transient ones contain only "Error" or are empty. (ref: issue #1608)
                    _err_body = getattr(api_error, "body", None) or {}
                    _err_message = (
                        _err_body.get("error", {}).get("message", "")
                        if isinstance(_err_body, dict)
                        else ""
                    )
                    _is_generic_400 = (
                        status_code == 400
                        and _err_message.strip().lower() in ("error", "")
                    )
                    is_client_status_error = (
                        isinstance(status_code, int)
                        and 400 <= status_code < 500
                        and status_code not in _RETRYABLE_STATUS_CODES
                        and not _is_generic_400
                    )
                    is_client_error = (
                        is_local_validation_error
                        or is_client_status_error
                        or any(
                            phrase in error_msg
                            for phrase in [
                                "error code: 401",
                                "error code: 403",
                                "error code: 404",
                                "error code: 422",
                                "is not a valid model",
                                "invalid model",
                                "model not found",
                                "invalid api key",
                                "invalid_api_key",
                                "authentication",
                                "unauthorized",
                                "forbidden",
                                "not found",
                            ]
                        )
                    ) and not is_context_length_error

                    if is_client_error:
                        # Try fallback before aborting — a different provider
                        # may not have the same issue (rate limit, auth, etc.)
                        if self._try_activate_fallback():
                            retry_count = 0
                            continue
                        self._dump_api_request_debug(
                            api_kwargs,
                            reason="non_retryable_client_error",
                            error=api_error,
                        )
                        self._vprint(
                            f"{self.log_prefix}❌ Non-retryable client error (HTTP {status_code}). Aborting.",
                            force=True,
                        )
                        self._vprint(
                            f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}",
                            force=True,
                        )
                        self._vprint(
                            f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True
                        )
                        # Actionable guidance for common auth errors
                        if (
                            status_code in (401, 403)
                            or "unauthorized" in error_msg
                            or "forbidden" in error_msg
                            or "permission" in error_msg
                        ):
                            self._vprint(
                                f"{self.log_prefix}   💡 Your API key was rejected by the provider. Check:",
                                force=True,
                            )
                            self._vprint(
                                f"{self.log_prefix}      • Is the key valid? Run: hermes setup",
                                force=True,
                            )
                            self._vprint(
                                f"{self.log_prefix}      • Does your account have access to {_model}?",
                                force=True,
                            )
                            if "openrouter" in str(_base).lower():
                                self._vprint(
                                    f"{self.log_prefix}      • Check credits: https://openrouter.ai/settings/credits",
                                    force=True,
                                )
                        else:
                            self._vprint(
                                f"{self.log_prefix}   💡 This type of error won't be fixed by retrying.",
                                force=True,
                            )
                        logging.error(
                            f"{self.log_prefix}Non-retryable client error: {api_error}"
                        )
                        # Skip session persistence when the error is likely
                        # context-overflow related (status 400 + large session).
                        # Persisting the failed user message would make the
                        # session even larger, causing the same failure on the
                        # next attempt. (#1630)
                        if status_code == 400 and (
                            approx_tokens > 50000 or len(api_messages) > 80
                        ):
                            self._vprint(
                                f"{self.log_prefix}⚠️  Skipping session persistence "
                                f"for large failed session to prevent growth loop.",
                                force=True,
                            )
                        else:
                            self._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": str(api_error),
                        }

                    if retry_count >= max_retries:
                        # Try fallback before giving up entirely
                        if self._try_activate_fallback():
                            retry_count = 0
                            continue
                        self._vprint(
                            f"{self.log_prefix}❌ Max retries ({max_retries}) exceeded. Giving up.",
                            force=True,
                        )
                        logging.error(
                            f"{self.log_prefix}API call failed after {max_retries} retries. Last error: {api_error}"
                        )
                        logging.error(
                            f"{self.log_prefix}Request details - Messages: {len(api_messages)}, Approx tokens: {approx_tokens:,}"
                        )
                        raise api_error

                    wait_time = min(
                        2**retry_count, 60
                    )  # Exponential backoff: 2s, 4s, 8s, 16s, 32s, 60s, 60s
                    logger.warning(
                        "Retrying API call in %ss (attempt %s/%s) %s error=%s",
                        wait_time,
                        retry_count,
                        max_retries,
                        self._client_log_context(),
                        api_error,
                    )
                    # Sleep in small increments so we can respond to interrupts quickly
                    # instead of blocking the entire wait_time in one sleep() call
                    sleep_end = time.time() + wait_time
                    while time.time() < sleep_end:
                        if self._interrupt_requested:
                            self._vprint(
                                f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.",
                                force=True,
                            )
                            self._persist_session(messages, conversation_history)
                            self.clear_interrupt()
                            return {
                                "final_response": f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries}).",
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "interrupted": True,
                            }
                        time.sleep(0.2)  # Check interrupt every 200ms

            # If the API call was interrupted, skip response processing
            if interrupted:
                break

            if restart_with_compressed_messages:
                api_call_count -= 1
                self.iteration_budget.refund()
                continue

            if restart_with_length_continuation:
                continue

            # Guard: if all retries exhausted without a successful response
            # (e.g. repeated context-length errors that exhausted retry_count),
            # the `response` variable is still None. Break out cleanly.
            if response is None:
                print(
                    f"{self.log_prefix}❌ All API retries exhausted with no successful response."
                )
                self._persist_session(messages, conversation_history)
                break

            try:
                if self.api_mode == "codex_responses":
                    assistant_message, finish_reason = self._normalize_codex_response(
                        response
                    )
                elif self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import normalize_anthropic_response

                    assistant_message, finish_reason = normalize_anthropic_response(
                        response,
                        strip_tool_prefix=getattr(self, "_is_anthropic_oauth", False),
                    )
                else:
                    assistant_message = response.choices[0].message

                # Normalize content to string — some OpenAI-compatible servers
                # (llama-server, etc.) return content as a dict or list instead
                # of a plain string, which crashes downstream .strip() calls.
                if assistant_message.content is not None and not isinstance(
                    assistant_message.content, str
                ):
                    raw = assistant_message.content
                    if isinstance(raw, dict):
                        assistant_message.content = (
                            raw.get("text", "")
                            or raw.get("content", "")
                            or json.dumps(raw)
                        )
                    elif isinstance(raw, list):
                        # Multimodal content list — extract text parts
                        parts = []
                        for part in raw:
                            if isinstance(part, str):
                                parts.append(part)
                            elif isinstance(part, dict) and part.get("type") == "text":
                                parts.append(part.get("text", ""))
                            elif isinstance(part, dict) and "text" in part:
                                parts.append(str(part["text"]))
                        assistant_message.content = "\n".join(parts)
                    else:
                        assistant_message.content = str(raw)

                # Handle assistant response
                if assistant_message.content and not self.quiet_mode:
                    if self.verbose_logging:
                        self._vprint(
                            f"{self.log_prefix}🤖 Assistant: {assistant_message.content}"
                        )
                    else:
                        self._vprint(
                            f"{self.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}"
                        )

                # Notify progress callback of model's thinking (used by subagent
                # delegation to relay the child's reasoning to the parent display).
                # Guard: only fire for subagents (_delegate_depth >= 1) to avoid
                # spamming gateway platforms with the main agent's every thought.
                if (
                    assistant_message.content
                    and self.tool_progress_callback
                    and getattr(self, "_delegate_depth", 0) > 0
                ):
                    _think_text = assistant_message.content.strip()
                    # Strip reasoning XML tags that shouldn't leak to parent display
                    _think_text = re.sub(
                        r"</?(?:REASONING_SCRATCHPAD|think|reasoning)>", "", _think_text
                    ).strip()
                    first_line = _think_text.split("\n")[0][:80] if _think_text else ""
                    if first_line:
                        try:
                            self.tool_progress_callback("_thinking", first_line)
                        except Exception:
                            pass

                # Check for incomplete <REASONING_SCRATCHPAD> (opened but never closed)
                # This means the model ran out of output tokens mid-reasoning — retry up to 2 times
                if has_incomplete_scratchpad(assistant_message.content or ""):
                    if not hasattr(self, "_incomplete_scratchpad_retries"):
                        self._incomplete_scratchpad_retries = 0
                    self._incomplete_scratchpad_retries += 1

                    self._vprint(
                        f"{self.log_prefix}⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)"
                    )

                    if self._incomplete_scratchpad_retries <= 2:
                        self._vprint(
                            f"{self.log_prefix}🔄 Retrying API call ({self._incomplete_scratchpad_retries}/2)..."
                        )
                        # Don't add the broken message, just retry
                        continue
                    else:
                        # Max retries - discard this turn and save as partial
                        self._vprint(
                            f"{self.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.",
                            force=True,
                        )
                        self._incomplete_scratchpad_retries = 0

                        rolled_back_messages = self._get_messages_up_to_last_assistant(
                            messages
                        )
                        self._cleanup_task_resources(effective_task_id)
                        self._persist_session(messages, conversation_history)

                        return {
                            "final_response": None,
                            "messages": rolled_back_messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Incomplete REASONING_SCRATCHPAD after 2 retries",
                        }

                # Reset incomplete scratchpad counter on clean response
                if hasattr(self, "_incomplete_scratchpad_retries"):
                    self._incomplete_scratchpad_retries = 0

                if self.api_mode == "codex_responses" and finish_reason == "incomplete":
                    if not hasattr(self, "_codex_incomplete_retries"):
                        self._codex_incomplete_retries = 0
                    self._codex_incomplete_retries += 1

                    interim_msg = self._build_assistant_message(
                        assistant_message, finish_reason
                    )
                    interim_has_content = bool(
                        (interim_msg.get("content") or "").strip()
                    )
                    interim_has_reasoning = (
                        bool(interim_msg.get("reasoning", "").strip())
                        if isinstance(interim_msg.get("reasoning"), str)
                        else False
                    )
                    interim_has_codex_reasoning = bool(
                        interim_msg.get("codex_reasoning_items")
                    )

                    if (
                        interim_has_content
                        or interim_has_reasoning
                        or interim_has_codex_reasoning
                    ):
                        last_msg = messages[-1] if messages else None
                        # Duplicate detection: two consecutive incomplete assistant
                        # messages with identical content AND reasoning are collapsed.
                        # For reasoning-only messages (codex_reasoning_items differ but
                        # visible content/reasoning are both empty), we also compare
                        # the encrypted items to avoid silently dropping new state.
                        last_codex_items = (
                            last_msg.get("codex_reasoning_items")
                            if isinstance(last_msg, dict)
                            else None
                        )
                        interim_codex_items = interim_msg.get("codex_reasoning_items")
                        duplicate_interim = (
                            isinstance(last_msg, dict)
                            and last_msg.get("role") == "assistant"
                            and last_msg.get("finish_reason") == "incomplete"
                            and (last_msg.get("content") or "")
                            == (interim_msg.get("content") or "")
                            and (last_msg.get("reasoning") or "")
                            == (interim_msg.get("reasoning") or "")
                            and last_codex_items == interim_codex_items
                        )
                        if not duplicate_interim:
                            messages.append(interim_msg)

                    if self._codex_incomplete_retries < 3:
                        if not self.quiet_mode:
                            self._vprint(
                                f"{self.log_prefix}↻ Codex response incomplete; continuing turn ({self._codex_incomplete_retries}/3)"
                            )
                        self._session_messages = messages
                        self._save_session_log(messages)
                        continue

                    self._codex_incomplete_retries = 0
                    self._persist_session(messages, conversation_history)
                    return {
                        "final_response": None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Codex response remained incomplete after 3 continuation attempts",
                    }
                elif hasattr(self, "_codex_incomplete_retries"):
                    self._codex_incomplete_retries = 0

                # Check for tool calls
                if assistant_message.tool_calls:
                    if not self.quiet_mode:
                        self._vprint(
                            f"{self.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)..."
                        )

                    if self.verbose_logging:
                        for tc in assistant_message.tool_calls:
                            logging.debug(
                                f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}..."
                            )

                    # Validate tool call names - detect model hallucinations
                    # Repair mismatched tool names before validating
                    for tc in assistant_message.tool_calls:
                        if tc.function.name not in self.valid_tool_names:
                            repaired = self._repair_tool_call(tc.function.name)
                            if repaired:
                                print(
                                    f"{self.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'"
                                )
                                tc.function.name = repaired
                    invalid_tool_calls = [
                        tc.function.name
                        for tc in assistant_message.tool_calls
                        if tc.function.name not in self.valid_tool_names
                    ]
                    if invalid_tool_calls:
                        # Track retries for invalid tool calls
                        if not hasattr(self, "_invalid_tool_retries"):
                            self._invalid_tool_retries = 0
                        self._invalid_tool_retries += 1

                        # Return helpful error to model — model can self-correct next turn
                        available = ", ".join(sorted(self.valid_tool_names))
                        invalid_name = invalid_tool_calls[0]
                        invalid_preview = (
                            invalid_name[:80] + "..."
                            if len(invalid_name) > 80
                            else invalid_name
                        )
                        self._vprint(
                            f"{self.log_prefix}⚠️  Unknown tool '{invalid_preview}' — sending error to model for self-correction ({self._invalid_tool_retries}/3)"
                        )

                        if self._invalid_tool_retries >= 3:
                            self._vprint(
                                f"{self.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.",
                                force=True,
                            )
                            self._invalid_tool_retries = 0
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": f"Model generated invalid tool call: {invalid_preview}",
                            }

                        assistant_msg = self._build_assistant_message(
                            assistant_message, finish_reason
                        )
                        messages.append(assistant_msg)
                        for tc in assistant_message.tool_calls:
                            if tc.function.name not in self.valid_tool_names:
                                content = f"Tool '{tc.function.name}' does not exist. Available tools: {available}"
                            else:
                                content = f"Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": content,
                                }
                            )
                        continue
                    # Reset retry counter on successful tool call validation
                    if hasattr(self, "_invalid_tool_retries"):
                        self._invalid_tool_retries = 0

                    # Validate tool call arguments are valid JSON
                    # Handle empty strings as empty objects (common model quirk)
                    invalid_json_args = []
                    for tc in assistant_message.tool_calls:
                        args = tc.function.arguments
                        if isinstance(args, (dict, list)):
                            tc.function.arguments = json.dumps(args)
                            continue
                        if args is not None and not isinstance(args, str):
                            tc.function.arguments = str(args)
                            args = tc.function.arguments
                        # Treat empty/whitespace strings as empty object
                        if not args or not args.strip():
                            tc.function.arguments = "{}"
                            continue
                        try:
                            json.loads(args)
                        except json.JSONDecodeError as e:
                            invalid_json_args.append((tc.function.name, str(e)))

                    if invalid_json_args:
                        # Track retries for invalid JSON arguments
                        self._invalid_json_retries += 1

                        tool_name, error_msg = invalid_json_args[0]
                        self._vprint(
                            f"{self.log_prefix}⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}"
                        )

                        if self._invalid_json_retries < 3:
                            self._vprint(
                                f"{self.log_prefix}🔄 Retrying API call ({self._invalid_json_retries}/3)..."
                            )
                            # Don't add anything to messages, just retry the API call
                            continue
                        else:
                            # Instead of returning partial, inject tool error results so the model can recover.
                            # Using tool results (not user messages) preserves role alternation.
                            self._vprint(
                                f"{self.log_prefix}⚠️  Injecting recovery tool results for invalid JSON..."
                            )
                            self._invalid_json_retries = 0  # Reset for next attempt

                            # Append the assistant message with its (broken) tool_calls
                            recovery_assistant = self._build_assistant_message(
                                assistant_message, finish_reason
                            )
                            messages.append(recovery_assistant)

                            # Respond with tool error results for each tool call
                            invalid_names = {name for name, _ in invalid_json_args}
                            for tc in assistant_message.tool_calls:
                                if tc.function.name in invalid_names:
                                    err = next(
                                        e
                                        for n, e in invalid_json_args
                                        if n == tc.function.name
                                    )
                                    tool_result = (
                                        f"Error: Invalid JSON arguments. {err}. "
                                        f"For tools with no required parameters, use an empty object: {{}}. "
                                        f"Please retry with valid JSON."
                                    )
                                else:
                                    tool_result = "Skipped: other tool call in this response had invalid JSON."
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tc.id,
                                        "content": tool_result,
                                    }
                                )
                            continue

                    # Reset retry counter on successful JSON validation
                    self._invalid_json_retries = 0

                    # ── Post-call guardrails ──────────────────────────
                    assistant_message.tool_calls = self._cap_delegate_task_calls(
                        assistant_message.tool_calls
                    )
                    assistant_message.tool_calls = self._deduplicate_tool_calls(
                        assistant_message.tool_calls
                    )

                    assistant_msg = self._build_assistant_message(
                        assistant_message, finish_reason
                    )

                    # If this turn has both content AND tool_calls, capture the content
                    # as a fallback final response. Common pattern: model delivers its
                    # answer and calls memory/skill tools as a side-effect in the same
                    # turn. If the follow-up turn after tools is empty, we use this.
                    turn_content = assistant_message.content or ""
                    if turn_content and self._has_content_after_think_block(
                        turn_content
                    ):
                        self._last_content_with_tools = turn_content
                        # The response was already streamed to the user in the
                        # response box.  The remaining tool calls (memory, skill,
                        # todo, etc.) are post-response housekeeping — mute all
                        # subsequent CLI output so they run invisibly.
                        if self._has_stream_consumers():
                            self._mute_post_response = True
                        elif self.quiet_mode:
                            clean = self._strip_think_blocks(turn_content).strip()
                            if clean:
                                self._vprint(f"  ┊ 💬 {clean}")

                    messages.append(assistant_msg)

                    # Close any open streaming display (response box, reasoning
                    # box) before tool execution begins.  Intermediate turns may
                    # have streamed early content that opened the response box;
                    # flushing here prevents it from wrapping tool feed lines.
                    # Only signal the display callback — TTS (_stream_callback)
                    # should NOT receive None (it uses None as end-of-stream).
                    if self.stream_delta_callback:
                        try:
                            self.stream_delta_callback(None)
                        except Exception:
                            pass

                    _msg_count_before_tools = len(messages)
                    self._execute_tool_calls(
                        assistant_message, messages, effective_task_id, api_call_count
                    )

                    # Signal that a paragraph break is needed before the next
                    # streamed text.  We don't emit it immediately because
                    # multiple consecutive tool iterations would stack up
                    # redundant blank lines.  Instead, _fire_stream_delta()
                    # will prepend a single "\n\n" the next time real text
                    # arrives.
                    self._stream_needs_break = True

                    # Refund the iteration if the ONLY tool(s) called were
                    # execute_code (programmatic tool calling).  These are
                    # cheap RPC-style calls that shouldn't eat the budget.
                    _tc_names = {
                        tc.function.name for tc in assistant_message.tool_calls
                    }
                    if _tc_names == {"execute_code"}:
                        self.iteration_budget.refund()

                    # Estimate next prompt size using real token counts from the
                    # last API response + rough estimate of newly appended tool
                    # results.  This catches cases where tool results push the
                    # context past the limit that last_prompt_tokens alone misses
                    # (e.g. large file reads, web extractions).
                    _compressor = self.context_compressor
                    _new_tool_msgs = messages[_msg_count_before_tools:]
                    _new_chars = sum(
                        len(str(m.get("content", "") or "")) for m in _new_tool_msgs
                    )
                    _estimated_next_prompt = (
                        _compressor.last_prompt_tokens
                        + _compressor.last_completion_tokens
                        + _new_chars
                        // 3  # conservative: JSON-heavy tool results ≈ 3 chars/token
                    )

                    # ── Context pressure warnings (user-facing only) ──────────
                    # Notify the user (NOT the LLM) as context approaches the
                    # compaction threshold.  Thresholds are relative to where
                    # compaction fires, not the raw context window.
                    # Does not inject into messages — just prints to CLI output
                    # and fires status_callback for gateway platforms.
                    if _compressor.threshold_tokens > 0:
                        _compaction_progress = (
                            _estimated_next_prompt / _compressor.threshold_tokens
                        )
                        if _compaction_progress >= 0.85 and not self._context_70_warned:
                            self._context_70_warned = True
                            self._context_50_warned = (
                                True  # skip first tier if we jumped past it
                            )
                            self._emit_context_pressure(
                                _compaction_progress, _compressor
                            )
                        elif (
                            _compaction_progress >= 0.60 and not self._context_50_warned
                        ):
                            self._context_50_warned = True
                            self._emit_context_pressure(
                                _compaction_progress, _compressor
                            )

                    if self.compression_enabled and _compressor.should_compress(
                        _estimated_next_prompt
                    ):
                        # Record context compaction for statistics
                        tokens_before = self.context_compressor.last_prompt_tokens
                        messages, active_system_prompt = self._compress_context(
                            messages,
                            system_message,
                            approx_tokens=tokens_before,
                            task_id=effective_task_id,
                        )
                        tokens_after = self.context_compressor.last_prompt_tokens
                        record_context_compaction(
                            tokens_before, tokens_after, self.provider
                        )

                    # Save session log incrementally (so progress is visible even if interrupted)
                    self._session_messages = messages
                    self._save_session_log(messages)

                    # Continue loop for next response
                    continue

                else:
                    # No tool calls - this is the final response
                    final_response = assistant_message.content or ""

                    # Check if response only has think block with no actual content after it
                    if not self._has_content_after_think_block(final_response):
                        # If the previous turn already delivered real content alongside
                        # tool calls (e.g. "You're welcome!" + memory save), the model
                        # has nothing more to say. Use the earlier content immediately
                        # instead of wasting API calls on retries that won't help.
                        fallback = getattr(self, "_last_content_with_tools", None)
                        if fallback:
                            logger.debug(
                                "Empty follow-up after tool calls — using prior turn content as final response"
                            )
                            self._last_content_with_tools = None
                            self._empty_content_retries = 0
                            for i in range(len(messages) - 1, -1, -1):
                                msg = messages[i]
                                if msg.get("role") == "assistant" and msg.get(
                                    "tool_calls"
                                ):
                                    tool_names = []
                                    for tc in msg["tool_calls"]:
                                        if not tc or not isinstance(tc, dict):
                                            continue
                                        fn = tc.get("function", {})
                                        tool_names.append(fn.get("name", "unknown"))
                                    msg["content"] = (
                                        f"Calling the {', '.join(tool_names)} tool{'s' if len(tool_names) > 1 else ''}..."
                                    )
                                    break
                            final_response = self._strip_think_blocks(fallback).strip()
                            self._response_was_previewed = True
                            break

                        # No fallback available — this is a genuine empty response.
                        # Retry in case the model just had a bad generation.
                        if not hasattr(self, "_empty_content_retries"):
                            self._empty_content_retries = 0
                        self._empty_content_retries += 1

                        reasoning_text = self._extract_reasoning(assistant_message)
                        self._vprint(
                            f"{self.log_prefix}⚠️  Response only contains think block with no content after it"
                        )
                        if reasoning_text:
                            reasoning_preview = (
                                reasoning_text[:500] + "..."
                                if len(reasoning_text) > 500
                                else reasoning_text
                            )
                            self._vprint(
                                f"{self.log_prefix}   Reasoning: {reasoning_preview}"
                            )
                        else:
                            content_preview = (
                                final_response[:80] + "..."
                                if len(final_response) > 80
                                else final_response
                            )
                            self._vprint(
                                f"{self.log_prefix}   Content: '{content_preview}'"
                            )

                        if self._empty_content_retries < 3:
                            self._vprint(
                                f"{self.log_prefix}🔄 Retrying API call ({self._empty_content_retries}/3)..."
                            )
                            continue
                        else:
                            self._vprint(
                                f"{self.log_prefix}❌ Max retries (3) for empty content exceeded.",
                                force=True,
                            )
                            self._empty_content_retries = 0

                            # If a prior tool_calls turn had real content, salvage it:
                            # rewrite that turn's content to a brief tool description,
                            # and use the original content as the final response here.
                            fallback = getattr(self, "_last_content_with_tools", None)
                            if fallback:
                                self._last_content_with_tools = None
                                # Find the last assistant message with tool_calls and rewrite it
                                for i in range(len(messages) - 1, -1, -1):
                                    msg = messages[i]
                                    if msg.get("role") == "assistant" and msg.get(
                                        "tool_calls"
                                    ):
                                        tool_names = []
                                        for tc in msg["tool_calls"]:
                                            if not tc or not isinstance(tc, dict):
                                                continue
                                            fn = tc.get("function", {})
                                            tool_names.append(fn.get("name", "unknown"))
                                        msg["content"] = (
                                            f"Calling the {', '.join(tool_names)} tool{'s' if len(tool_names) > 1 else ''}..."
                                        )
                                        break
                                # Strip <think> blocks from fallback content for user display
                                final_response = self._strip_think_blocks(
                                    fallback
                                ).strip()
                                self._response_was_previewed = True
                                break

                            # No fallback -- if reasoning_text exists, the model put its
                            # entire response inside <think> tags; use that as the content.
                            if reasoning_text:
                                self._vprint(
                                    f"{self.log_prefix}Using reasoning as response content (model wrapped entire response in think tags).",
                                    force=True,
                                )
                                final_response = reasoning_text
                                empty_msg = {
                                    "role": "assistant",
                                    "content": final_response,
                                    "reasoning": reasoning_text,
                                    "finish_reason": finish_reason,
                                }
                                messages.append(empty_msg)
                                break

                            # Truly empty -- no reasoning and no content
                            empty_msg = {
                                "role": "assistant",
                                "content": final_response,
                                "reasoning": reasoning_text,
                                "finish_reason": finish_reason,
                            }
                            messages.append(empty_msg)

                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)

                            return {
                                "final_response": final_response or None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Model generated only think blocks with no actual response after 3 retries",
                            }

                    # Reset retry counter on successful content
                    if hasattr(self, "_empty_content_retries"):
                        self._empty_content_retries = 0

                    if (
                        self.api_mode == "codex_responses"
                        and self.valid_tool_names
                        and codex_ack_continuations < 2
                        and self._looks_like_codex_intermediate_ack(
                            user_message=user_message,
                            assistant_content=final_response,
                            messages=messages,
                        )
                    ):
                        codex_ack_continuations += 1
                        interim_msg = self._build_assistant_message(
                            assistant_message, "incomplete"
                        )
                        messages.append(interim_msg)

                        continue_msg = {
                            "role": "user",
                            "content": (
                                "[System: Continue now. Execute the required tool calls and only "
                                "send your final answer after completing the task.]"
                            ),
                        }
                        messages.append(continue_msg)
                        self._session_messages = messages
                        self._save_session_log(messages)
                        continue

                    codex_ack_continuations = 0

                    if truncated_response_prefix:
                        final_response = truncated_response_prefix + final_response
                        truncated_response_prefix = ""
                        length_continue_retries = 0

                    # Strip <think> blocks from user-facing response (keep raw in messages for trajectory)
                    final_response = self._strip_think_blocks(final_response).strip()

                    final_msg = self._build_assistant_message(
                        assistant_message, finish_reason
                    )

                    messages.append(final_msg)

                    if not self.quiet_mode:
                        self._safe_print(
                            f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)"
                        )
                    break

            except Exception as e:
                error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
                try:
                    print(f"❌ {error_msg}")
                except OSError:
                    logger.error(error_msg)

                if self.verbose_logging:
                    logging.exception("Detailed error information:")

                # If an assistant message with tool_calls was already appended,
                # the API expects a role="tool" result for every tool_call_id.
                # Fill in error results for any that weren't answered yet.
                pending_handled = False
                for idx in range(len(messages) - 1, -1, -1):
                    msg = messages[idx]
                    if not isinstance(msg, dict):
                        break
                    if msg.get("role") == "tool":
                        continue
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        answered_ids = {
                            m["tool_call_id"]
                            for m in messages[idx + 1 :]
                            if isinstance(m, dict) and m.get("role") == "tool"
                        }
                        for tc in msg["tool_calls"]:
                            if not tc or not isinstance(tc, dict):
                                continue
                            if tc["id"] not in answered_ids:
                                err_msg = {
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": f"Error executing tool: {error_msg}",
                                }
                                messages.append(err_msg)
                        pending_handled = True
                    break

                # Non-tool errors don't need a synthetic message injected.
                # The error is already printed to the user (line above), and
                # the retry loop continues.  Injecting a fake user/assistant
                # message pollutes history, burns tokens, and risks violating
                # role-alternation invariants.

                # If we're near the limit, break to avoid infinite loops
                if api_call_count >= self.max_iterations - 1:
                    final_response = (
                        f"I apologize, but I encountered repeated errors: {error_msg}"
                    )
                    # Append as assistant so the history stays valid for
                    # session resume (avoids consecutive user messages).
                    messages.append({"role": "assistant", "content": final_response})
                    break

        if final_response is None and (
            api_call_count >= self.max_iterations
            or self.iteration_budget.remaining <= 0
        ):
            if self.iteration_budget.remaining <= 0 and not self.quiet_mode:
                print(
                    f"\n⚠️  Session iteration budget exhausted ({self.iteration_budget.used}/{self.iteration_budget.max_total} used, including subagents)"
                )
            final_response = self._handle_max_iterations(messages, api_call_count)

        # Determine if conversation completed successfully
        completed = final_response is not None and api_call_count < self.max_iterations

        # Save trajectory if enabled
        self._save_trajectory(messages, user_message, completed)

        # Clean up VM and browser for this task after conversation completes
        self._cleanup_task_resources(effective_task_id)

        # Persist session to both JSON log and SQLite
        self._persist_session(messages, conversation_history)

        # Sync conversation to Honcho for user modeling
        if final_response and not interrupted and sync_honcho:
            self._honcho_sync(original_user_message, final_response)
            self._queue_honcho_prefetch(original_user_message)

        # Extract reasoning from the last assistant message (if any)
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break

        # Build result with interrupt info if applicable
        result = {
            "final_response": final_response,
            "last_reasoning": last_reasoning,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": completed,
            "partial": False,  # True only when stopped due to invalid tool calls
            "interrupted": interrupted,
            "response_previewed": getattr(self, "_response_was_previewed", False),
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "input_tokens": self.session_input_tokens,
            "output_tokens": self.session_output_tokens,
            "cache_read_tokens": self.session_cache_read_tokens,
            "cache_write_tokens": self.session_cache_write_tokens,
            "reasoning_tokens": self.session_reasoning_tokens,
            "prompt_tokens": self.session_prompt_tokens,
            "completion_tokens": self.session_completion_tokens,
            "total_tokens": self.session_total_tokens,
            "last_prompt_tokens": getattr(
                self.context_compressor, "last_prompt_tokens", 0
            )
            or 0,
            "estimated_cost_usd": self.session_estimated_cost_usd,
            "cost_status": self.session_cost_status,
            "cost_source": self.session_cost_source,
        }
        self._response_was_previewed = False

        # Include interrupt message if one triggered the interrupt
        if interrupted and self._interrupt_message:
            result["interrupt_message"] = self._interrupt_message

        # Clear interrupt state after handling
        self.clear_interrupt()

        # Clear stream callback so it doesn't leak into future calls
        self._stream_callback = None

        # Check skill trigger NOW — based on how many tool iterations THIS turn used.
        _should_review_skills = False
        if (
            self._skill_nudge_interval > 0
            and self._iters_since_skill >= self._skill_nudge_interval
            and "skill_manage" in self.valid_tool_names
        ):
            _should_review_skills = True
            self._iters_since_skill = 0

        # Background memory/skill review — runs AFTER the response is delivered
        # so it never competes with the user's task for model attention.
        if (
            final_response
            and not interrupted
            and (_should_review_memory or _should_review_skills)
        ):
            try:
                self._spawn_background_review(
                    messages_snapshot=list(messages),
                    review_memory=_should_review_memory,
                    review_skills=_should_review_skills,
                )
            except Exception:
                pass  # Background review is best-effort

        return result
