"""Mixin providing context and API-related methods for AIAgent.

This mixin handles:
- System prompt building (_build_system_prompt)
- API kwargs construction (_build_api_kwargs, _max_tokens_param, etc.)
- Context compression (_compress_context)
- Provider fallback (_try_activate_fallback)
- Client management (OpenAI, Anthropic, Codex)
- Reasoning extraction (_extract_reasoning)
- Content cleaning (_strip_think_blocks, etc.)
"""

import copy
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Callable, Tuple

logger = logging.getLogger(__name__)


class ContextMixin:
    """Mixin providing context and API-related methods for AIAgent."""

    def _is_direct_openai_url(self, base_url: Optional[str] = None) -> bool:
        """Return True when a base URL targets OpenAI's native API."""
        url = (base_url or getattr(self, '_base_url_lower', '')).lower()
        return "api.openai.com" in url and "openrouter" not in url

    def _max_tokens_param(self, value: int) -> dict:
        """Return the correct max tokens kwarg for the current provider.

        OpenAI's newer models (gpt-4o, o-series, gpt-5+) require
        'max_completion_tokens'. OpenRouter, local models, and older
        OpenAI models use 'max_tokens'.
        """
        if self._is_direct_openai_url():
            return {"max_completion_tokens": value}
        return {"max_tokens": value}

    def _strip_think_blocks(self, content: str) -> str:
        """Remove reasoning/thinking blocks from content, returning only visible text."""
        if not content:
            return ""
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
        content = re.sub(r'<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>', '', content, flags=re.DOTALL)
        return content

    def _has_content_after_think_block(self, content: str) -> bool:
        """Check if content has actual text after any reasoning/thinking blocks."""
        if not content:
            return False
        cleaned = self._strip_think_blocks(content)
        return bool(cleaned.strip())

    def _extract_reasoning(self, assistant_message) -> Optional[str]:
        """Extract reasoning/thinking content from an assistant message."""
        reasoning_parts = []

        if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
            reasoning_parts.append(assistant_message.reasoning)

        if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
            if assistant_message.reasoning_content not in reasoning_parts:
                reasoning_parts.append(assistant_message.reasoning_content)

        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            for detail in assistant_message.reasoning_details:
                if isinstance(detail, dict):
                    summary = detail.get('summary') or detail.get('content') or detail.get('text')
                    if summary and summary not in reasoning_parts:
                        reasoning_parts.append(summary)

        if reasoning_parts:
            return "\n\n".join(reasoning_parts)
        return None

    def _build_system_prompt(self, system_message: Optional[str] = None) -> str:
        """Assemble the full system prompt from all layers."""
        from agent.prompt_builder import (
            DEFAULT_AGENT_IDENTITY, PLATFORM_HINTS,
            MEMORY_GUIDANCE, SESSION_SEARCH_GUIDANCE, SKILLS_GUIDANCE,
            build_skills_system_prompt, build_context_files_prompt, load_soul_md,
        )

        _soul_loaded = False
        if not getattr(self, 'skip_context_files', False):
            _soul_content = load_soul_md()
            if _soul_content:
                prompt_parts = [_soul_content]
                _soul_loaded = True

        if not _soul_loaded:
            _ai_peer_name = (
                getattr(self, '_honcho_config', None)
                and getattr(getattr(self, '_honcho_config', None), 'ai_peer', None)
            )
            if _ai_peer_name and _ai_peer_name != "hermes":
                _identity = DEFAULT_AGENT_IDENTITY.replace(
                    "You are Hermes Agent", f"You are {_ai_peer_name}", 1
                )
            else:
                _identity = DEFAULT_AGENT_IDENTITY
            prompt_parts = [_identity]

        tool_guidance = []
        valid_tools = getattr(self, 'valid_tool_names', set())
        if "memory" in valid_tools:
            tool_guidance.append(MEMORY_GUIDANCE)
        if "session_search" in valid_tools:
            tool_guidance.append(SESSION_SEARCH_GUIDANCE)
        if "skill_manage" in valid_tools:
            tool_guidance.append(SKILLS_GUIDANCE)
        if tool_guidance:
            prompt_parts.append(" ".join(tool_guidance))

        if getattr(self, '_honcho', None) and getattr(self, '_honcho_session_key', None):
            hcfg = getattr(self, '_honcho_config', None)
            mode = getattr(hcfg, 'memory_mode', 'hybrid') if hcfg else "hybrid"
            freq = getattr(hcfg, 'write_frequency', 'async') if hcfg else "async"
            recall_mode = getattr(hcfg, 'recall_mode', 'hybrid') if hcfg else "hybrid"
            honcho_block = (
                f"# Honcho memory integration\n"
                f"Active. Session: {self._honcho_session_key}. "
                f"Mode: {mode}. Write frequency: {freq}. Recall: {recall_mode}.\n"
            )
            prompt_parts.append(honcho_block)

        if system_message is not None:
            prompt_parts.append(system_message)

        memory_store = getattr(self, '_memory_store', None)
        if memory_store:
            if getattr(self, '_memory_enabled', False):
                mem_block = memory_store.format_for_system_prompt("memory")
                if mem_block:
                    prompt_parts.append(mem_block)
            if getattr(self, '_user_profile_enabled', False):
                user_block = memory_store.format_for_system_prompt("user")
                if user_block:
                    prompt_parts.append(user_block)

        has_skills_tools = any(name in valid_tools for name in ['skills_list', 'skill_view', 'skill_manage'])
        if has_skills_tools:
            from model_tools import check_toolset_requirements
            avail_toolsets = {ts for ts, avail in check_toolset_requirements().items() if avail}
            skills_prompt = build_skills_system_prompt(
                available_tools=valid_tools,
                available_toolsets=avail_toolsets,
            )
            if skills_prompt:
                prompt_parts.append(skills_prompt)

        if not getattr(self, 'skip_context_files', False):
            context_files_prompt = build_context_files_prompt(skip_soul=_soul_loaded)
            if context_files_prompt:
                prompt_parts.append(context_files_prompt)

        from hermes_time import now as _hermes_now
        now = _hermes_now()
        timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
        if getattr(self, 'pass_session_id', False) and getattr(self, 'session_id', None):
            timestamp_line += f"\nSession ID: {self.session_id}"
        if getattr(self, 'model', None):
            timestamp_line += f"\nModel: {self.model}"
        if getattr(self, 'provider', None):
            timestamp_line += f"\nProvider: {self.provider}"
        prompt_parts.append(timestamp_line)

        if getattr(self, 'provider', '') == "alibaba":
            _model_short = self.model.split("/")[-1] if "/" in self.model else self.model
            prompt_parts.append(
                f"You are powered by the model named {_model_short}. "
                f"The exact model ID is {self.model}. "
                f"When asked what model you are, always answer based on this information."
            )

        platform_key = (getattr(self, 'platform', '') or "").lower().strip()
        if platform_key in PLATFORM_HINTS:
            prompt_parts.append(PLATFORM_HINTS[platform_key])

        return "\n\n".join(prompt_parts)

    def _invalidate_system_prompt(self):
        """Invalidate the cached system prompt, forcing a rebuild on the next turn."""
        self._cached_system_prompt = None
        if getattr(self, '_memory_store', None):
            self._memory_store.load_from_disk()

    def _compress_context(
        self,
        messages: List[Dict[str, Any]],
        system_message: str,
        *,
        approx_tokens: Optional[int] = None,
        task_id: str = "default"
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Compress conversation context and split the session in SQLite."""
        compressor = getattr(self, 'context_compressor', None)
        if not compressor:
            logger.warning("Context compressor not available, skipping compression")
            self._invalidate_system_prompt()
            new_system_prompt = self._build_system_prompt(system_message)
            self._cached_system_prompt = new_system_prompt
            return messages, new_system_prompt

        from agent.model_metadata import estimate_tokens_rough, estimate_messages_tokens_rough
        from agent.token_stats import record_context_compaction
        from agent.usage_pricing import normalize_usage, estimate_usage_cost

        self.flush_memories(messages, min_turns=0)

        compressed = compressor.compress(messages, current_tokens=approx_tokens)

        todo_snapshot = getattr(self, '_todo_store', None)
        if todo_snapshot:
            fmt = todo_snapshot.format_for_injection()
            if fmt:
                compressed.append({"role": "user", "content": fmt})

        self._invalidate_system_prompt()
        new_system_prompt = self._build_system_prompt(system_message)
        self._cached_system_prompt = new_system_prompt

        session_db = getattr(self, '_session_db', None)
        if session_db:
            try:
                old_title = session_db.get_session_title(getattr(self, 'session_id', None))
                session_db.end_session(getattr(self, 'session_id', None), "compression")
                old_session_id = self.session_id
                from datetime import datetime
                self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                session_db.create_session(
                    session_id=self.session_id,
                    source=getattr(self, 'platform', 'cli'),
                    model=getattr(self, 'model', None),
                    parent_session_id=old_session_id,
                )
                if old_title:
                    try:
                        new_title = session_db.get_next_title_in_lineage(old_title)
                        session_db.set_session_title(self.session_id, new_title)
                    except (ValueError, Exception) as e:
                        logger.debug("Could not propagate title on compression: %s", e)
                session_db.update_system_prompt(self.session_id, new_system_prompt)
                self._last_flushed_db_idx = 0
            except Exception as e:
                logger.debug("Session DB compression split failed: %s", e)

        self._context_50_warned = False
        self._context_70_warned = False

        return compressed, new_system_prompt

    def _build_api_kwargs(self, api_messages: list) -> dict:
        """Build the keyword arguments dict for the active API mode."""
        if getattr(self, 'api_mode', None) == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_kwargs
            anthropic_messages = self._prepare_anthropic_messages_for_api(api_messages)
            return build_anthropic_kwargs(
                model=getattr(self, 'model', None),
                messages=anthropic_messages,
                tools=getattr(self, 'tools', None),
                max_tokens=getattr(self, 'max_tokens', None),
                reasoning_config=getattr(self, 'reasoning_config', None),
                is_oauth=getattr(self, '_is_anthropic_oauth', False),
                preserve_dots=self._anthropic_preserve_dots(),
            )

        if getattr(self, 'api_mode', None) == "codex_responses":
            instructions = ""
            payload_messages = api_messages
            if api_messages and api_messages[0].get("role") == "system":
                instructions = str(api_messages[0].get("content") or "").strip()
                payload_messages = api_messages[1:]
            if not instructions:
                from agent.prompt_builder import DEFAULT_AGENT_IDENTITY
                instructions = DEFAULT_AGENT_IDENTITY

            is_github_responses = (
                "models.github.ai" in getattr(self, 'base_url', '').lower()
                or "api.githubcopilot.com" in getattr(self, 'base_url', '').lower()
            )

            reasoning_effort = "medium"
            reasoning_enabled = True
            rc = getattr(self, 'reasoning_config', None)
            if rc and isinstance(rc, dict):
                if rc.get("enabled") is False:
                    reasoning_enabled = False
                elif rc.get("effort"):
                    reasoning_effort = rc["effort"]

            kwargs = {
                "model": getattr(self, 'model', None),
                "instructions": instructions,
                "input": self._chat_messages_to_responses_input(payload_messages),
                "tools": self._responses_tools(),
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "store": False,
            }

            if not is_github_responses:
                kwargs["prompt_cache_key"] = getattr(self, 'session_id', None)

            if reasoning_enabled:
                if is_github_responses:
                    github_reasoning = self._github_models_reasoning_extra_body()
                    if github_reasoning is not None:
                        kwargs["reasoning"] = github_reasoning
                else:
                    kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
                    kwargs["include"] = ["reasoning.encrypted_content"]
            elif not is_github_responses:
                kwargs["include"] = []

            if getattr(self, 'max_tokens', None) is not None:
                kwargs["max_output_tokens"] = self.max_tokens

            return kwargs

        sanitized_messages = api_messages
        needs_sanitization = False
        for msg in api_messages:
            if not isinstance(msg, dict):
                continue
            if "codex_reasoning_items" in msg:
                needs_sanitization = True
                break
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    if "call_id" in tool_call or "response_item_id" in tool_call:
                        needs_sanitization = True
                        break
                if needs_sanitization:
                    break

        if needs_sanitization:
            sanitized_messages = copy.deepcopy(api_messages)
            for msg in sanitized_messages:
                if not isinstance(msg, dict):
                    continue
                msg.pop("codex_reasoning_items", None)
                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_call.pop("call_id", None)
                            tool_call.pop("response_item_id", None)

        provider_preferences = {}
        if getattr(self, 'providers_allowed', None):
            provider_preferences["only"] = self.providers_allowed
        if getattr(self, 'providers_ignored', None):
            provider_preferences["ignore"] = self.providers_ignored
        if getattr(self, 'providers_order', None):
            provider_preferences["order"] = self.providers_order
        if getattr(self, 'provider_sort', None):
            provider_preferences["sort"] = self.provider_sort
        if getattr(self, 'provider_require_parameters', None):
            provider_preferences["require_parameters"] = True
        if getattr(self, 'provider_data_collection', None):
            provider_preferences["data_collection"] = self.provider_data_collection

        api_kwargs = {
            "model": getattr(self, 'model', None),
            "messages": sanitized_messages,
            "tools": getattr(self, 'tools', None) if getattr(self, 'tools', None) else None,
            "timeout": float(os.getenv("HERMES_API_TIMEOUT", 900.0)),
        }

        if getattr(self, 'max_tokens', None) is not None:
            api_kwargs.update(self._max_tokens_param(self.max_tokens))

        extra_body = {}
        _is_openrouter = "openrouter" in getattr(self, '_base_url_lower', '')
        _is_github_models = (
            "models.github.ai" in getattr(self, '_base_url_lower', '')
            or "api.githubcopilot.com" in getattr(self, '_base_url_lower', '')
        )

        if provider_preferences and _is_openrouter:
            extra_body["provider"] = provider_preferences

        if self._supports_reasoning_extra_body():
            if _is_github_models:
                github_reasoning = self._github_models_reasoning_extra_body()
                if github_reasoning is not None:
                    extra_body["reasoning"] = github_reasoning
            else:
                if getattr(self, 'reasoning_config', None) is not None:
                    rc = dict(self.reasoning_config)
                    _is_nous = "nousresearch" in getattr(self, '_base_url_lower', '')
                    if _is_nous and rc.get("enabled") is False:
                        pass
                    else:
                        extra_body["reasoning"] = rc
                else:
                    extra_body["reasoning"] = {"enabled": True, "effort": "medium"}

        _is_nous = "nousresearch" in getattr(self, '_base_url_lower', '')
        if _is_nous:
            extra_body["tags"] = ["product=hermes-agent"]

        if extra_body:
            api_kwargs["extra_body"] = extra_body

        return api_kwargs

    def _supports_reasoning_extra_body(self) -> bool:
        """Return True when reasoning extra_body is safe to send."""
        if "nousresearch" in getattr(self, '_base_url_lower', ''):
            return True
        if "ai-gateway.vercel.sh" in getattr(self, '_base_url_lower', ''):
            return True
        if "models.github.ai" in getattr(self, '_base_url_lower', '') or "api.githubcopilot.com" in getattr(self, '_base_url_lower', ''):
            try:
                from hermes_cli.models import github_model_reasoning_efforts
                return bool(github_model_reasoning_efforts(getattr(self, 'model', None)))
            except Exception:
                return False
        if "openrouter" not in getattr(self, '_base_url_lower', ''):
            return False
        if "api.mistral.ai" in getattr(self, '_base_url_lower', ''):
            return False

        model = (getattr(self, 'model', '') or "").lower()
        reasoning_model_prefixes = (
            "deepseek/", "anthropic/", "openai/", "x-ai/",
            "google/gemini-2", "qwen/qwen3",
        )
        return any(model.startswith(prefix) for prefix in reasoning_model_prefixes)

    def _github_models_reasoning_extra_body(self) -> Optional[dict]:
        """Format reasoning payload for GitHub Models/OpenAI-compatible routes."""
        try:
            from hermes_cli.models import github_model_reasoning_efforts
        except Exception:
            return None

        supported_efforts = github_model_reasoning_efforts(getattr(self, 'model', None))
        if not supported_efforts:
            return None

        rc = getattr(self, 'reasoning_config', None)
        if rc and isinstance(rc, dict):
            if rc.get("enabled") is False:
                return None
            requested_effort = str(rc.get("effort", "medium")).strip().lower()
        else:
            requested_effort = "medium"

        if requested_effort == "xhigh" and "high" in supported_efforts:
            requested_effort = "high"
        elif requested_effort not in supported_efforts:
            if requested_effort == "minimal" and "low" in supported_efforts:
                requested_effort = "low"
            elif "medium" in supported_efforts:
                requested_effort = "medium"
            else:
                requested_effort = supported_efforts[0]

        return {"effort": requested_effort}

    def _responses_tools(self, tools: Optional[List[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
        """Convert chat-completions tool schemas to Responses function-tool schemas."""
        source_tools = tools if tools is not None else getattr(self, 'tools', None)
        if not source_tools:
            return None

        converted: List[Dict[str, Any]] = []
        for item in source_tools:
            fn = item.get("function", {}) if isinstance(item, dict) else {}
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            converted.append({
                "type": "function",
                "name": name,
                "description": fn.get("description", ""),
                "strict": False,
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return converted or None

    def _chat_messages_to_responses_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert internal chat-style messages to Responses input items."""
        from agent.prompt_builder import DEFAULT_AGENT_IDENTITY
        items: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content_text = msg.get("content", "")

            if role == "system":
                continue

            if role == "assistant":
                if content_text:
                    items.append({"role": "assistant", "content": content_text})

                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if not tc or not isinstance(tc, dict):
                            continue
                        fn = tc.get("function", {})
                        fn_name = fn.get("name") if isinstance(fn, dict) else None
                        if not isinstance(fn_name, str) or not fn_name.strip():
                            continue

                        embedded_call_id, embedded_response_item_id = self._split_responses_tool_id(tc.get("id"))
                        call_id = tc.get("call_id")
                        if not isinstance(call_id, str) or not call_id.strip():
                            call_id = embedded_call_id
                        if not isinstance(call_id, str) or not call_id.strip():
                            if (
                                embedded_response_item_id
                                and embedded_response_item_id.startswith("fc_")
                                and len(embedded_response_item_id) > len("fc_")
                            ):
                                call_id = f"call_{embedded_response_item_id[len('fc_'):]} "
                            else:
                                call_id = f"call_{uuid.uuid4().hex[:12]}"
                        call_id = call_id.strip()

                        response_item_id = tc.get("response_item_id")
                        if not isinstance(response_item_id, str) or not response_item_id.strip():
                            response_item_id = self._derive_responses_function_call_id(call_id, embedded_response_item_id)

                        arguments = fn.get("arguments", "{}")
                        if isinstance(arguments, dict):
                            arguments = json.dumps(arguments, ensure_ascii=False)
                        elif not isinstance(arguments, str):
                            arguments = str(arguments)
                        arguments = arguments.strip() or "{}"

                        items.append({
                            "type": "function_call",
                            "call_id": call_id,
                            "name": fn_name,
                            "arguments": arguments,
                        })
                continue

            if role == "tool":
                raw_tool_call_id = msg.get("tool_call_id", "")
                call_id = ""
                if isinstance(raw_tool_call_id, str):
                    call_id = raw_tool_call_id.strip()
                if not isinstance(call_id, str) or not call_id.strip():
                    continue
                items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(msg.get("content", "") or ""),
                })
                continue

            items.append({"role": role, "content": content_text})

        return items

    def _split_responses_tool_id(self, raw_id: Any) -> Tuple[Optional[str], Optional[str]]:
        """Split a stored tool id into (call_id, response_item_id)."""
        if not isinstance(raw_id, str):
            return None, None
        parts = raw_id.split("__", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return raw_id, None

    def _derive_responses_function_call_id(self, call_id: str, response_item_id: Optional[str]) -> str:
        """Derive a response_item_id for a function call."""
        if isinstance(response_item_id, str) and response_item_id.strip():
            return response_item_id.strip()
        seed = f"{call_id}__fc"
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]
        return f"fc_{digest}"

    def _preflight_codex_input_items(self, raw_items: Any) -> List[Dict[str, Any]]:
        """Validate and normalize Codex Responses input items."""
        if not isinstance(raw_items, list):
            raise ValueError("Codex Responses input must be a list of input items.")

        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise ValueError(f"Codex Responses input[{idx}] must be an object.")

            item_type = item.get("type")
            if item_type == "function_call":
                call_id = item.get("call_id")
                name = item.get("name")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call is missing call_id.")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call is missing name.")

                arguments = item.get("arguments", "{}")
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                elif not isinstance(arguments, str):
                    arguments = str(arguments)
                arguments = arguments.strip() or "{}"

                normalized.append({
                    "type": "function_call",
                    "call_id": call_id.strip(),
                    "name": name.strip(),
                    "arguments": arguments,
                })
                continue

            if item_type == "function_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call_output is missing call_id.")
                output = item.get("output", "")
                if output is None:
                    output = ""
                if not isinstance(output, str):
                    output = str(output)

                normalized.append({
                    "type": "function_call_output",
                    "call_id": call_id.strip(),
                    "output": output,
                })
                continue

            if item_type == "reasoning":
                encrypted = item.get("encrypted_content")
                if isinstance(encrypted, str) and encrypted:
                    reasoning_item: Dict[str, Any] = {"type": "reasoning", "encrypted_content": encrypted}
                    item_id = item.get("id")
                    if isinstance(item_id, str) and item_id:
                        reasoning_item["id"] = item_id
                    summary = item.get("summary")
                    if isinstance(summary, list):
                        reasoning_item["summary"] = summary
                    else:
                        reasoning_item["summary"] = []
                    normalized.append(reasoning_item)
                continue

            role = item.get("role")
            if role in {"user", "assistant"}:
                content = item.get("content", "")
                if content is None:
                    content = ""
                if not isinstance(content, str):
                    content = str(content)

                normalized.append({"role": role, "content": content})
                continue

            raise ValueError(
                f"Codex Responses input[{idx}] has unsupported item shape (type={item_type!r}, role={role!r})."
            )

        return normalized

    def _extract_responses_message_text(self, item: Any) -> str:
        """Extract assistant text from a Responses message output item."""
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            return ""

        chunks: List[str] = []
        for part in content:
            ptype = getattr(part, "type", None)
            if ptype not in {"output_text", "text"}:
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
        return "".join(chunks).strip()

    def _extract_responses_reasoning_text(self, item: Any) -> str:
        """Extract a compact reasoning text from a Responses reasoning item."""
        summary = getattr(item, "summary", None)
        if isinstance(summary, list):
            chunks: List[str] = []
            for part in summary:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    chunks.append(text)
            if chunks:
                return "\n".join(chunks).strip()
        text = getattr(item, "text", None)
        if isinstance(text, str) and text:
            return text.strip()
        return ""

    def _normalize_codex_response(self, response: Any) -> Tuple[Any, str]:
        """Normalize a Responses API object to an assistant_message-like object."""
        output = getattr(response, "output", None)
        if not isinstance(output, list) or not output:
            raise RuntimeError("Responses API returned no output items")

        response_status = getattr(response, "status", None)
        if isinstance(response_status, str):
            response_status = response_status.strip().lower()
        else:
            response_status = None

        if response_status in {"failed", "cancelled"}:
            error_obj = getattr(response, "error", None)
            if isinstance(error_obj, dict):
                error_msg = error_obj.get("message") or str(error_obj)
            else:
                error_msg = str(error_obj) if error_obj else f"Responses API returned status '{response_status}'"
            raise RuntimeError(error_msg)

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        reasoning_items_raw: List[Dict[str, Any]] = []
        tool_calls: List[Any] = []
        has_incomplete_items = response_status in {"queued", "in_progress", "incomplete"}
        saw_commentary_phase = False
        saw_final_answer_phase = False

        for item in output:
            item_type = getattr(item, "type", None)
            item_status = getattr(item, "status", None)
            if isinstance(item_status, str):
                item_status = item_status.strip().lower()
            else:
                item_status = None

            if item_status in {"queued", "in_progress", "incomplete"}:
                has_incomplete_items = True

            if item_type == "message":
                item_phase = getattr(item, "phase", None)
                if isinstance(item_phase, str):
                    normalized_phase = item_phase.strip().lower()
                    if normalized_phase in {"commentary", "analysis"}:
                        saw_commentary_phase = True
                    elif normalized_phase in {"final_answer", "final"}:
                        saw_final_answer_phase = True
                message_text = self._extract_responses_message_text(item)
                if message_text:
                    content_parts.append(message_text)
            elif item_type == "reasoning":
                reasoning_text = self._extract_responses_reasoning_text(item)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                encrypted = getattr(item, "encrypted_content", None)
                if isinstance(encrypted, str) and encrypted:
                    raw_item: Dict[str, Any] = {"type": "reasoning", "encrypted_content": encrypted}
                    item_id = getattr(item, "id", None)
                    if isinstance(item_id, str) and item_id:
                        raw_item["id"] = item_id
                    summary = getattr(item, "summary", None)
                    if isinstance(summary, list):
                        raw_summary = []
                        for part in summary:
                            text = getattr(part, "text", None)
                            if isinstance(text, str):
                                raw_summary.append({"type": "summary_text", "text": text})
                        raw_item["summary"] = raw_summary
                    reasoning_items_raw.append(raw_item)
            elif item_type == "function_call":
                if item_status in {"queued", "in_progress", "incomplete"}:
                    continue
                fn_name = getattr(item, "name", "") or ""
                arguments = getattr(item, "arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                raw_call_id = getattr(item, "call_id", None)
                raw_item_id = getattr(item, "id", None)
                embedded_call_id, _ = self._split_responses_tool_id(raw_item_id)
                call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = f"call_{uuid.uuid4().hex[:12]}"
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(call_id, response_item_id)
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    call_id=call_id,
                    response_item_id=response_item_id,
                    type="function",
                    function=SimpleNamespace(name=fn_name, arguments=arguments),
                ))
            elif item_type == "custom_tool_call":
                fn_name = getattr(item, "name", "") or ""
                arguments = getattr(item, "input", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                raw_call_id = getattr(item, "call_id", None)
                raw_item_id = getattr(item, "id", None)
                embedded_call_id, _ = self._split_responses_tool_id(raw_item_id)
                call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = f"call_{uuid.uuid4().hex[:12]}"
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(call_id, response_item_id)
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    call_id=call_id,
                    response_item_id=response_item_id,
                    type="function",
                    function=SimpleNamespace(name=fn_name, arguments=arguments),
                ))

        final_text = "\n".join([p for p in content_parts if p]).strip()
        if not final_text and hasattr(response, "output_text"):
            out_text = getattr(response, "output_text", "")
            if isinstance(out_text, str):
                final_text = out_text.strip()

        assistant_message = SimpleNamespace(
            content=final_text,
            tool_calls=tool_calls,
            reasoning="\n\n".join(reasoning_parts).strip() if reasoning_parts else None,
            reasoning_content=None,
            reasoning_details=None,
            codex_reasoning_items=reasoning_items_raw or None,
        )

        if tool_calls:
            finish_reason = "tool_calls"
        elif has_incomplete_items or (saw_commentary_phase and not saw_final_answer_phase):
            finish_reason = "incomplete"
        elif reasoning_items_raw and not final_text:
            finish_reason = "incomplete"
        else:
            finish_reason = "stop"
        return assistant_message, finish_reason
