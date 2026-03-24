"""Streaming mixin for AIAgent class.

This mixin provides streaming response handling capabilities including:
- Stream delta callback firing
- Reasoning delta callback firing  
- Stream consumer detection
- Interruptible streaming API calls

Methods extracted from run_agent.py AIAgent class in Phase 3.
"""

import logging
import threading
import uuid
from types import SimpleNamespace
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)


class StreamingMixin:
    """Mixin providing streaming handler methods.
    
    This mixin is composed into AIAgent to provide
    streaming-related functionality extracted from the monolithic
    run_agent.py implementation.
    """
    
    # ── Unified streaming API call ─────────────────────────────────────────

    def _fire_stream_delta(self, text: str) -> None:
        """Fire all registered stream delta callbacks (display + TTS)."""
        # If a tool iteration set the break flag, prepend a single paragraph
        # break before the first real text delta. This prevents the original
        # problem (text concatenation across tool boundaries) without stacking
        # blank lines when multiple tool iterations run back-to-back.
        if getattr(self, "_stream_needs_break", False) and text and text.strip():
            self._stream_needs_break = False
            text = "\n\n" + text
        for cb in (self.stream_delta_callback, self._stream_callback):
            if cb is not None:
                try:
                    cb(text)
                except Exception:
                    pass

    def _fire_reasoning_delta(self, text: str) -> None:
        """Fire reasoning callback if registered."""
        cb = self.reasoning_callback
        if cb is not None:
            try:
                cb(text)
            except Exception:
                pass

    def _has_stream_consumers(self) -> bool:
        """Return True if any streaming consumer is registered."""
        return (
            self.stream_delta_callback is not None
            or getattr(self, "_stream_callback", None) is not None
        )

    def _interruptible_streaming_api_call(
        self, api_kwargs: dict, *, on_first_delta: Optional[Callable] = None
    ):
        """Streaming variant of _interruptible_api_call for real-time token delivery.

        Handles all three api_modes:
        - chat_completions: stream=True on OpenAI-compatible endpoints
        - anthropic_messages: client.messages.stream() via Anthropic SDK
        - codex_responses: delegates to _run_codex_stream (already streaming)

        Fires stream_delta_callback and _stream_callback for each text token.
        Tool-call turns suppress the callback — only text-only final responses
        stream to the consumer. Returns a SimpleNamespace that mimics the
        non-streaming response shape so the rest of the agent loop is unchanged.

        Falls back to _interruptible_api_call on provider errors indicating
        streaming is not supported.
        """
        if self.api_mode == "codex_responses":
            # Codex streams internally via _run_codex_stream. The main dispatch
            # in _interruptible_api_call already calls it; we just need to
            # ensure on_first_delta reaches it. Store it on the instance
            # temporarily so _run_codex_stream can pick it up.
            self._codex_on_first_delta = on_first_delta
            try:
                return self._interruptible_api_call(api_kwargs)
            finally:
                self._codex_on_first_delta = None

        result: Dict[str, Any] = {"response": None, "error": None}
        request_client_holder = {"client": None}
        first_delta_fired = {"done": False}
        deltas_were_sent = {"yes": False}  # Track if any deltas were fired (for fallback)

        def _fire_first_delta():
            if not first_delta_fired["done"] and on_first_delta:
                first_delta_fired["done"] = True
                try:
                    on_first_delta()
                except Exception:
                    pass

        def _call_chat_completions():
            """Stream a chat completions response."""
            stream_kwargs = {**api_kwargs, "stream": True, "stream_options": {"include_usage": True}}
            request_client_holder["client"] = self._create_request_openai_client(
                reason="chat_completion_stream_request"
            )
            stream = request_client_holder["client"].chat.completions.create(**stream_kwargs)

            content_parts: list = []
            tool_calls_acc: dict = {}
            finish_reason = None
            model_name = None
            role = "assistant"
            reasoning_parts: list = []
            usage_obj = None

            for chunk in stream:
                if self._interrupt_requested:
                    break

                if not chunk.choices:
                    if hasattr(chunk, "model") and chunk.model:
                        model_name = chunk.model
                    # Usage comes in the final chunk with empty choices
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_obj = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if hasattr(chunk, "model") and chunk.model:
                    model_name = chunk.model

                # Accumulate reasoning content
                reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                    self._fire_reasoning_delta(reasoning_text)

                # Accumulate text content — fire callback only when no tool calls
                if delta and delta.content:
                    content_parts.append(delta.content)
                    if not tool_calls_acc:
                        _fire_first_delta()
                        self._fire_stream_delta(delta.content)
                        deltas_were_sent["yes"] = True

                # Accumulate tool call deltas (silently, no callback)
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index if tc_delta.index is not None else 0
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        entry = tool_calls_acc[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["function"]["arguments"] += tc_delta.function.arguments

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # Usage in the final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage

            # Build mock response matching non-streaming shape
            full_content = "".join(content_parts) or None
            mock_tool_calls = None
            if tool_calls_acc:
                mock_tool_calls = []
                for idx in sorted(tool_calls_acc):
                    tc = tool_calls_acc[idx]
                    mock_tool_calls.append(SimpleNamespace(
                        id=tc["id"],
                        type=tc["type"],
                        function=SimpleNamespace(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    ))

            full_reasoning = "".join(reasoning_parts) or None
            mock_message = SimpleNamespace(
                role=role,
                content=full_content,
                tool_calls=mock_tool_calls,
                reasoning_content=full_reasoning,
            )
            mock_choice = SimpleNamespace(
                index=0,
                message=mock_message,
                finish_reason=finish_reason or "stop",
            )
            return SimpleNamespace(
                id="stream-" + str(uuid.uuid4()),
                model=model_name,
                choices=[mock_choice],
                usage=usage_obj,
            )

        def _call_anthropic():
            """Stream an Anthropic Messages API response.

            Fires delta callbacks for real-time token delivery, but returns
            the native Anthropic Message object from get_final_message() so
            the rest of the agent loop (validation, tool extraction, etc.)
            works unchanged.
            """
            has_tool_use = False

            if self._anthropic_client is None:
                raise RuntimeError("Anthropic client is not available")

            # Use the Anthropic SDK's streaming context manager
            with self._anthropic_client.messages.stream(**api_kwargs) as stream:
                for event in stream:
                    if self._interrupt_requested:
                        break

                    event_type = getattr(event, "type", None)

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", None) == "tool_use":
                            has_tool_use = True

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            delta_type = getattr(delta, "type", None)
                            if delta_type == "text_delta":
                                text = getattr(delta, "text", "")
                                if text and not has_tool_use:
                                    _fire_first_delta()
                                    self._fire_stream_delta(text)
                            elif delta_type == "thinking_delta":
                                thinking_text = getattr(delta, "thinking", "")
                                if thinking_text:
                                    self._fire_reasoning_delta(thinking_text)

                # Return the native Anthropic Message for downstream processing
                return stream.get_final_message()

        def _call():
            try:
                if self.api_mode == "anthropic_messages":
                    self._try_refresh_anthropic_client_credentials()
                    result["response"] = _call_anthropic()
                else:
                    result["response"] = _call_chat_completions()
            except Exception as e:
                if deltas_were_sent["yes"]:
                    # Streaming failed AFTER some tokens were already delivered
                    # to consumers. Don't fall back — that would cause
                    # double-delivery (partial streamed + full non-streamed).
                    # Let the error propagate; the partial content already
                    # reached the user via the stream.
                    logger.warning("Streaming failed after partial delivery, not falling back: %s", e)
                    result["error"] = e
                else:
                    # Streaming failed before any tokens reached consumers.
                    # Safe to fall back to the standard non-streaming path.
                    logger.info("Streaming failed before delivery, falling back to non-streaming: %s", e)
                    try:
                        result["response"] = self._interruptible_api_call(api_kwargs)
                    except Exception as fallback_err:
                        result["error"] = fallback_err
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="stream_request_complete")

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.3)
            if self._interrupt_requested:
                try:
                    if self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import build_anthropic_client

                        if self._anthropic_client is not None:
                            if self._anthropic_client is not None:
                                self._anthropic_client.close()
                                self._anthropic_client = build_anthropic_client(
                                    self._anthropic_api_key,
                                    getattr(self, "_anthropic_base_url", None),
                                )
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="stream_interrupt_abort")
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during streaming API call")
        if result["error"] is not None:
            raise result["error"]
        return result["response"]
