#!/usr/bin/env python3
"""
AI Agent Runner with Tool Calling

This module provides a clean, standalone agent that can execute AI models
with tool calling capabilities. It handles the conversation loop, tool execution,
and response management.

Features:
- Automatic tool calling loop until completion
- Configurable model parameters
- Error handling and recovery
- Message history management
- Support for multiple model providers

Usage:
    from run_agent import AIAgent

    agent = AIAgent(base_url="http://localhost:30000/v1", model="claude-opus-4-20250514")
    response = agent.run_conversation("Tell me about the latest Python updates")
"""

import atexit
import asyncio
import base64
import concurrent.futures
import copy
import hashlib
import json
import logging

logger = logging.getLogger(__name__)
import os
import random
import re
import sys
import tempfile
import time
import threading
import weakref
from types import SimpleNamespace
import uuid
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime
from pathlib import Path

# Load .env from ~/.hermes/.env first, then project root as dev fallback.
# User-managed env files should override stale shell exports on restart.
from hermes_cli.env_loader import load_hermes_dotenv

_hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
_project_env = Path(__file__).parent / ".env"
_loaded_env_paths = load_hermes_dotenv(
    hermes_home=_hermes_home, project_env=_project_env
)
if _loaded_env_paths:
    for _env_path in _loaded_env_paths:
        logger.info("Loaded environment variables from %s", _env_path)
else:
    logger.info("No .env file found. Using system environment variables.")

# Point mini-swe-agent at ~/.hermes/ so it shares our config
os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", str(_hermes_home))
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

# Conditional imports for tool system and dependencies
try:
    # Import our tool system
    from model_tools import (
        get_tool_definitions,
        handle_function_call,
        check_toolset_requirements,
    )
    from tools.terminal_tool import cleanup_vm
    from tools.interrupt import set_interrupt as _set_interrupt
    from tools.browser_tool import cleanup_browser

    tools_available = True
except ImportError as e:
    logger.warning("Tool system not available: %s", e)

    # Define stubs for when tools aren't available
    def get_tool_definitions(*args, **kwargs):
        return []

    def handle_function_call(*args, **kwargs):
        return json.dumps({"error": "Tool system not available"})

    def check_toolset_requirements(*args, **kwargs):
        return {}

    cleanup_vm = lambda task_id: None
    _set_interrupt = lambda x: None
    cleanup_browser = lambda task_id: None
    tools_available = False

try:
    import requests

    requests_available = True
except ImportError:
    logger.warning("requests not available")
    requests = None
    requests_available = False

try:
    from hermes_constants import OPENROUTER_BASE_URL, OPENROUTER_MODELS_URL

    constants_available = True
except ImportError:
    logger.warning("hermes_constants not available")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
    constants_available = False

# Agent internals extracted to agent/ package for modularity
try:
    from agent.prompt_builder import (
        DEFAULT_AGENT_IDENTITY,
        PLATFORM_HINTS,
        MEMORY_GUIDANCE,
        SESSION_SEARCH_GUIDANCE,
        SKILLS_GUIDANCE,
    )
    from agent.model_metadata import (
        fetch_model_metadata,
        get_model_context_length,
        estimate_tokens_rough,
        estimate_messages_tokens_rough,
        get_next_probe_tier,
        parse_context_limit_from_error,
        save_context_length,
    )
    from agent.context_compressor import ContextCompressor
    from agent.prompt_caching import apply_anthropic_cache_control
    from agent.prompt_builder import (
        build_skills_system_prompt,
        build_context_files_prompt,
        load_soul_md,
    )
    from agent.usage_pricing import estimate_usage_cost, normalize_usage
    from agent.display import (
        KawaiiSpinner,
        build_tool_preview as _build_tool_preview,
        get_cute_tool_message as _get_cute_tool_message_impl,
        _detect_tool_failure,
        get_tool_emoji as _get_tool_emoji,
    )
    from agent.trajectory import (
        convert_scratchpad_to_think,
        has_incomplete_scratchpad,
        save_trajectory as _save_trajectory_to_file,
    )
    from utils import atomic_json_write

    agent_internals_available = True
except ImportError as e:
    logger.warning("Agent internals not available: %s", e)
    # Define stubs for agent internals
    DEFAULT_AGENT_IDENTITY = ""
    PLATFORM_HINTS = {}
    MEMORY_GUIDANCE = ""
    SESSION_SEARCH_GUIDANCE = ""
    SKILLS_GUIDANCE = ""
    fetch_model_metadata = lambda: {}
    get_model_context_length = lambda x: 4096
    estimate_tokens_rough = lambda x: len(x) // 4
    estimate_messages_tokens_rough = lambda x: sum(
        estimate_tokens_rough(str(msg.get("content", ""))) for msg in x
    )
    get_next_probe_tier = lambda x: 4096
    parse_context_limit_from_error = lambda x: None
    save_context_length = lambda x, y: None
    ContextCompressor = lambda **kwargs: None
    apply_anthropic_cache_control = lambda x, y: x
    build_skills_system_prompt = lambda **kwargs: ""
    build_context_files_prompt = lambda **kwargs: ""
    load_soul_md = lambda: ""
    estimate_usage_cost = lambda **kwargs: 0.0
    normalize_usage = lambda x: x
    KawaiiSpinner = lambda *args, **kwargs: None
    _build_tool_preview = lambda *args, **kwargs: ""
    _get_cute_tool_message_impl = lambda *args, **kwargs: ""
    _detect_tool_failure = lambda x, y: (False, None)
    _get_tool_emoji = lambda x: "🔧"
    convert_scratchpad_to_think = lambda x: x
    has_incomplete_scratchpad = lambda x: False
    _save_trajectory_to_file = lambda trajectory, model, completed: None
    atomic_json_write = lambda *args, **kwargs: None
    agent_internals_available = False

# Token efficiency imports
try:
    from agent.prompt_cache import PromptCache
    from agent.rate_limiter import CoordinatedRateLimiter
    from agent.context_compaction import ContextCompactionManager
    from agent.token_stats import record_context_compaction
    from agent.token_stats import (
        get_global_stats,
        record_cache_hit,
        record_cache_miss,
        record_rate_limit_wait,
        record_context_compaction,
    )

    token_efficiency_available = True
except ImportError as e:
    logger.warning("Token efficiency modules not available: %s", e)
    # Define stubs for token efficiency
    PromptCache = lambda **kwargs: None
    CoordinatedRateLimiter = lambda **kwargs: None
    ContextCompactionManager = lambda **kwargs: None
    record_context_compaction = lambda **kwargs: None
    get_global_stats = lambda: {}
    record_cache_hit = lambda **kwargs: None
    record_cache_miss = lambda **kwargs: None
    record_rate_limit_wait = lambda *args, **kwargs: None
    record_context_compaction = lambda **kwargs: None
    token_efficiency_available = False

# Execution mixin for tool call handling
from agent.mixins import (
    ExecutionMixin,
    StreamingMixin,
    SessionMixin,
    ConversationMixin,
    ContextMixin,
)

HONCHO_TOOL_NAMES = {
    "honcho_context",
    "honcho_profile",
    "honcho_search",
    "honcho_conclude",
}


class _SafeWriter:
    """Transparent stdio wrapper that catches OSError/ValueError from broken pipes.

    When hermes-agent runs as a systemd service, Docker container, or headless
    daemon, the stdout/stderr pipe can become unavailable (idle timeout, buffer
    exhaustion, socket reset). Any print() call then raises
    ``OSError: [Errno 5] Input/output error``, which can crash agent setup or
    run_conversation() — especially via double-fault when an except handler
    also tries to print.

    Additionally, when subagents run in ThreadPoolExecutor threads, the shared
    stdout handle can close between thread teardown and cleanup, raising
    ``ValueError: I/O operation on closed file`` instead of OSError.

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


class IterationBudget:
    """Thread-safe shared iteration counter for parent and child agents.

    Tracks total LLM-call iterations consumed across a parent agent and all
    its subagents.  A single ``IterationBudget`` is created by the parent
    and passed to every child so they share the same cap.

    ``execute_code`` (programmatic tool calling) iterations are refunded via
    :meth:`refund` so they don't eat into the budget.
    """

    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Try to consume one iteration.  Returns True if allowed."""
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """Give back one iteration (e.g. for execute_code turns)."""
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)


# Tools that must never run concurrently (interactive / user-facing).
# When any of these appear in a batch, we fall back to sequential execution.
_NEVER_PARALLEL_TOOLS = frozenset({"clarify"})

# Read-only tools with no shared mutable session state.
_PARALLEL_SAFE_TOOLS = frozenset(
    {
        "ha_get_state",
        "ha_list_entities",
        "ha_list_services",
        "honcho_context",
        "honcho_profile",
        "honcho_search",
        "read_file",
        "search_files",
        "session_search",
        "skill_view",
        "skills_list",
        "vision_analyze",
        "web_extract",
        "web_search",
    }
)

# File tools can run concurrently when they target independent paths.
_PATH_SCOPED_TOOLS = frozenset({"read_file", "write_file", "patch"})

# Maximum number of concurrent worker threads for parallel tool execution.
_MAX_TOOL_WORKERS = 8

# Patterns that indicate a terminal command may modify/delete files.
_DESTRUCTIVE_PATTERNS = re.compile(
    r"""(?:^|\s|&&|\|\||;|`)(?:
        rm\s|rmdir\s|
        mv\s|
        sed\s+-i|
        truncate\s|
        dd\s|
        shred\s|
        git\s+(?:reset|clean|checkout)\s
    )""",
    re.VERBOSE,
)
# Output redirects that overwrite files (> but not >>)
_REDIRECT_OVERWRITE = re.compile(r"[^>]>[^>]|^>[^>]")


def _is_destructive_command(cmd: str) -> bool:
    """Heuristic: does this terminal command look like it modifies/deletes files?"""
    if not cmd:
        return False
    if _DESTRUCTIVE_PATTERNS.search(cmd):
        return True
    if _REDIRECT_OVERWRITE.search(cmd):
        return True
    return False


def _should_parallelize_tool_batch(tool_calls) -> bool:
    """Return True when a tool-call batch is safe to run concurrently."""
    if len(tool_calls) <= 1:
        return False

    tool_names = [tc.function.name for tc in tool_calls]
    if any(name in _NEVER_PARALLEL_TOOLS for name in tool_names):
        return False

    reserved_paths: list[Path] = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        try:
            function_args = json.loads(tool_call.function.arguments)
        except Exception:
            logging.debug(
                "Could not parse args for %s — defaulting to sequential; raw=%s",
                tool_name,
                tool_call.function.arguments[:200],
            )
            return False
        if not isinstance(function_args, dict):
            logging.debug(
                "Non-dict args for %s (%s) — defaulting to sequential",
                tool_name,
                type(function_args).__name__,
            )
            return False

        if tool_name in _PATH_SCOPED_TOOLS:
            scoped_path = _extract_parallel_scope_path(tool_name, function_args)
            if scoped_path is None:
                return False
            if any(
                _paths_overlap(scoped_path, existing) for existing in reserved_paths
            ):
                return False
            reserved_paths.append(scoped_path)
            continue

        if tool_name not in _PARALLEL_SAFE_TOOLS:
            return False

    return True


def _extract_parallel_scope_path(tool_name: str, function_args: dict) -> Path | None:
    """Return the normalized file target for path-scoped tools."""
    if tool_name not in _PATH_SCOPED_TOOLS:
        return None

    raw_path = function_args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    # Avoid resolve(); the file may not exist yet.
    return Path(raw_path).expanduser()


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return True when two paths may refer to the same subtree."""
    left_parts = left.parts
    right_parts = right.parts
    if not left_parts or not right_parts:
        # Empty paths shouldn't reach here (guarded upstream), but be safe.
        return bool(left_parts) == bool(right_parts) and bool(left_parts)
    common_len = min(len(left_parts), len(right_parts))
    return left_parts[:common_len] == right_parts[:common_len]


def _inject_honcho_turn_context(content, turn_context: str):
    """Append Honcho recall to the current-turn user message without mutating history.

    The returned content is sent to the API for this turn only. Keeping Honcho
    recall out of the system prompt preserves the stable cache prefix while
    still giving the model continuity context.
    """
    if not turn_context:
        return content

    note = (
        "[System note: The following Honcho memory was retrieved from prior "
        "sessions. It is continuity context for this turn only, not new user "
        "input.]\n\n"
        f"{turn_context}"
    )

    if isinstance(content, list):
        return list(content) + [{"type": "text", "text": note}]

    text = "" if content is None else str(content)
    if not text.strip():
        return note
    return f"{text}\n\n{note}"


class AIAgent(
    ExecutionMixin,
    StreamingMixin,
    SessionMixin,
    ConversationMixin,
    ContextMixin,
):
    """
    AI Agent with tool calling capabilities.

    This class manages the conversation flow, tool execution, and response handling
    for AI models that support function calling.
    """

    @property
    def base_url(self) -> str:
        return self._base_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        self._base_url = value
        self._base_url_lower = value.lower() if value else ""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str | None = None,
        api_mode: str | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        model: str = "anthropic/claude-opus-4.6",  # OpenRouter format
        max_iterations: int = 90,  # Default tool-calling iterations (shared with subagents)
        tool_delay: float = 1.0,
        enabled_toolsets: List[str] | None = None,
        disabled_toolsets: List[str] | None = None,
        save_trajectories: bool = False,
        verbose_logging: bool = False,
        quiet_mode: bool = False,
        ephemeral_system_prompt: str | None = None,
        log_prefix_chars: int = 100,
        log_prefix: str = "",
        providers_allowed: List[str] | None = None,
        providers_ignored: List[str] | None = None,
        providers_order: List[str] | None = None,
        provider_sort: str | None = None,
        provider_require_parameters: bool = False,
        provider_data_collection: str | None = None,
        session_id: str | None = None,
        tool_progress_callback: Callable | None = None,
        thinking_callback: Callable | None = None,
        reasoning_callback: Callable | None = None,
        clarify_callback: Callable | None = None,
        step_callback: Callable | None = None,
        stream_delta_callback: Callable | None = None,
        status_callback: Callable | None = None,
        max_tokens: int | None = None,
        reasoning_config: Dict[str, Any] | None = None,
        prefill_messages: List[Dict[str, Any]] | None = None,
        platform: str | None = None,
        skip_context_files: bool = False,
        skip_memory: bool = False,
        session_db=None,
        honcho_session_key: str | None = None,
        honcho_manager=None,
        honcho_config=None,
        iteration_budget: Optional[IterationBudget] = None,
        fallback_model: Optional[Dict[str, Any]] = None,
        checkpoints_enabled: bool = False,
        checkpoint_max_snapshots: int = 50,
        pass_session_id: bool = False,
    ):
        """
        Initialize the AI Agent.

        Args:
            base_url (str): Base URL for the model API (optional)
            api_key (str): API key for authentication (optional, uses env var if not provided)
            provider (str): Provider identifier (optional; used for telemetry/routing hints)
            api_mode (str): API mode override: "chat_completions" or "codex_responses"
            model (str): Model name to use (default: "anthropic/claude-opus-4.6")
            max_iterations (int): Maximum number of tool calling iterations (default: 90)
            tool_delay (float): Delay between tool calls in seconds (default: 1.0)
            enabled_toolsets (List[str]): Only enable tools from these toolsets (optional)
            disabled_toolsets (List[str]): Disable tools from these toolsets (optional)
            save_trajectories (bool): Whether to save conversation trajectories to JSONL files (default: False)
            verbose_logging (bool): Enable verbose logging for debugging (default: False)
            quiet_mode (bool): Suppress progress output for clean CLI experience (default: False)
            ephemeral_system_prompt (str): System prompt used during agent execution but NOT saved to trajectories (optional)
            log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses (default: 100)
            log_prefix (str): Prefix to add to all log messages for identification in parallel processing (default: "")
            providers_allowed (List[str]): OpenRouter providers to allow (optional)
            providers_ignored (List[str]): OpenRouter providers to ignore (optional)
            providers_order (List[str]): OpenRouter providers to try in order (optional)
            provider_sort (str): Sort providers by price/throughput/latency (optional)
            session_id (str): Pre-generated session ID for logging (optional, auto-generated if not provided)
            tool_progress_callback (callable): Callback function(tool_name, args_preview) for progress notifications
            clarify_callback (callable): Callback function(question, choices) -> str for interactive user questions.
                Provided by the platform layer (CLI or gateway). If None, the clarify tool returns an error.
            max_tokens (int): Maximum tokens for model responses (optional, uses model default if not set)
            reasoning_config (Dict): OpenRouter reasoning configuration override (e.g. {"effort": "none"} to disable thinking).
                If None, defaults to {"enabled": True, "effort": "medium"} for OpenRouter. Set to disable/customize reasoning.
            prefill_messages (List[Dict]): Messages to prepend to conversation history as prefilled context.
                Useful for injecting a few-shot example or priming the model's response style.
                Example: [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello!"}]
            platform (str): The interface platform the user is on (e.g. "cli", "telegram", "discord", "whatsapp").
                Used to inject platform-specific formatting hints into the system prompt.
            skip_context_files (bool): If True, skip auto-injection of SOUL.md, AGENTS.md, and .cursorrules
                into the system prompt. Use this for batch processing and data generation to avoid
                polluting trajectories with user-specific persona or project instructions.
            honcho_session_key (str): Session key for Honcho integration (e.g., "telegram:123456" or CLI session_id).
                When provided and Honcho is enabled in config, enables persistent cross-session user modeling.
            honcho_manager: Optional shared HonchoSessionManager owned by the caller.
            honcho_config: Optional HonchoClientConfig corresponding to honcho_manager.
        """
        _install_safe_stdio()

        self.model = model
        self.max_iterations = max_iterations
        # Shared iteration budget — parent creates, children inherit.
        # Consumed by every LLM turn across parent + all subagents.
        self.iteration_budget = iteration_budget or IterationBudget(max_iterations)
        self.tool_delay = tool_delay
        self.save_trajectories = save_trajectories
        self.verbose_logging = verbose_logging
        self.quiet_mode = quiet_mode
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.platform = platform  # "cli", "telegram", "discord", "whatsapp", etc.
        # Pluggable print function — CLI replaces this with _cprint so that
        # raw ANSI status lines are routed through prompt_toolkit's renderer
        # instead of going directly to stdout where patch_stdout's StdoutProxy
        # would mangle the escape sequences.  None = use builtins.print.
        self._print_fn = None
        self.skip_context_files = skip_context_files
        self.pass_session_id = pass_session_id
        self.log_prefix_chars = log_prefix_chars
        self.log_prefix = f"{log_prefix} " if log_prefix else ""
        # Store effective base URL for feature detection (prompt caching, reasoning, etc.)
        # When no base_url is provided, the client defaults to OpenRouter, so reflect that here.
        self.base_url = base_url or OPENROUTER_BASE_URL
        provider_name = (
            provider.strip().lower()
            if isinstance(provider, str) and provider.strip()
            else None
        )
        self.provider = provider_name or "openrouter"
        self.acp_command = acp_command or command
        self.acp_args = list(acp_args or args or [])
        if api_mode in {"chat_completions", "codex_responses", "anthropic_messages"}:
            self.api_mode = api_mode
        elif self.provider == "openai-codex":
            self.api_mode = "codex_responses"
        elif (
            provider_name is None
        ) and "chatgpt.com/backend-api/codex" in self._base_url_lower:
            self.api_mode = "codex_responses"
            self.provider = "openai-codex"
        elif self.provider == "anthropic" or (
            provider_name is None and "api.anthropic.com" in self._base_url_lower
        ):
            self.api_mode = "anthropic_messages"
            self.provider = "anthropic"
        elif self._base_url_lower.rstrip("/").endswith("/anthropic"):
            # Third-party Anthropic-compatible endpoints (e.g. MiniMax, DashScope)
            # use a URL convention ending in /anthropic. Auto-detect these so the
            # Anthropic Messages API adapter is used instead of chat completions.
            self.api_mode = "anthropic_messages"
        else:
            self.api_mode = "chat_completions"

        # Direct OpenAI sessions use the Responses API path.  GPT-5.x tool
        # calls with reasoning are rejected on /v1/chat/completions, and
        # Hermes is a tool-using client by default.
        if self.api_mode == "chat_completions" and self._is_direct_openai_url():
            self.api_mode = "codex_responses"

        # Pre-warm OpenRouter model metadata cache in a background thread.
        # fetch_model_metadata() is cached for 1 hour; this avoids a blocking
        # HTTP request on the first API response when pricing is estimated.
        if self.provider == "openrouter" or "openrouter" in self._base_url_lower:
            threading.Thread(
                target=lambda: fetch_model_metadata(),
                daemon=True,
            ).start()

        self.tool_progress_callback = tool_progress_callback
        self.thinking_callback = thinking_callback
        self.reasoning_callback = reasoning_callback
        self.clarify_callback = clarify_callback
        self.step_callback = step_callback
        self.stream_delta_callback = stream_delta_callback
        self.status_callback = status_callback
        self._last_reported_tool = None  # Track for "new tool" mode

        # Tool execution state — allows _vprint during tool execution
        # even when stream consumers are registered (no tokens streaming then)
        self._executing_tools = False

        # Interrupt mechanism for breaking out of tool loops
        self._interrupt_requested = False
        self._interrupt_message = None  # Optional message that triggered interrupt
        self._client_lock = threading.RLock()

        # Subagent delegation state
        self._delegate_depth = 0  # 0 = top-level agent, incremented for children
        self._active_children = []  # Running child AIAgents (for interrupt propagation)
        self._active_children_lock = threading.Lock()

        # Store OpenRouter provider preferences
        self.providers_allowed = providers_allowed
        self.providers_ignored = providers_ignored
        self.providers_order = providers_order
        self.provider_sort = provider_sort
        self.provider_require_parameters = provider_require_parameters
        self.provider_data_collection = provider_data_collection

        # Store toolset filtering options
        self.enabled_toolsets = enabled_toolsets
        self.disabled_toolsets = disabled_toolsets

        # Model response configuration
        self.max_tokens = max_tokens  # None = use model default
        self.reasoning_config = (
            reasoning_config  # None = use default (medium for OpenRouter)
        )
        self.prefill_messages = prefill_messages or []  # Prefilled conversation turns

        # Anthropic prompt caching: auto-enabled for Claude models via OpenRouter.
        # Reduces input costs by ~75% on multi-turn conversations by caching the
        # conversation prefix. Uses system_and_3 strategy (4 breakpoints).
        is_openrouter = "openrouter" in self._base_url_lower
        is_claude = "claude" in self.model.lower()
        is_native_anthropic = self.api_mode == "anthropic_messages"
        self._use_prompt_caching = (is_openrouter and is_claude) or is_native_anthropic
        self._cache_ttl = "5m"  # Default 5-minute TTL (1.25x write cost)

        # Iteration budget pressure: warn the LLM as it approaches max_iterations.
        # Warnings are injected into the last tool result JSON (not as separate
        # messages) so they don't break message structure or invalidate caching.
        self._budget_caution_threshold = 0.7  # 70% — nudge to start wrapping up
        self._budget_warning_threshold = 0.9  # 90% — urgent, respond now
        self._budget_pressure_enabled = True

        # Context pressure warnings: notify the USER (not the LLM) as context
        # fills up.  Purely informational — displayed in CLI output and sent via
        # status_callback for gateway platforms.  Does NOT inject into messages.
        self._context_50_warned = False
        self._context_70_warned = False

        # Persistent error log -- always writes WARNING+ to ~/.hermes/logs/errors.log
        # so tool failures, API errors, etc. are inspectable after the fact.
        # In gateway mode, each incoming message creates a new AIAgent instance,
        # while the root logger is process-global. Re-adding the same errors.log
        # handler would cause each warning/error line to be written multiple times.
        from logging.handlers import RotatingFileHandler

        root_logger = logging.getLogger()
        error_log_dir = _hermes_home / "logs"
        error_log_path = error_log_dir / "errors.log"
        resolved_error_log_path = error_log_path.resolve()
        has_errors_log_handler = any(
            isinstance(handler, RotatingFileHandler)
            and Path(getattr(handler, "baseFilename", "")).resolve()
            == resolved_error_log_path
            for handler in root_logger.handlers
        )
        from agent.redact import RedactingFormatter

        if not has_errors_log_handler:
            error_log_dir.mkdir(parents=True, exist_ok=True)
            error_file_handler = RotatingFileHandler(
                error_log_path,
                maxBytes=2 * 1024 * 1024,
                backupCount=2,
            )
            error_file_handler.setLevel(logging.WARNING)
            error_file_handler.setFormatter(
                RedactingFormatter(
                    "%(asctime)s %(levelname)s %(name)s: %(message)s",
                )
            )
            root_logger.addHandler(error_file_handler)

        if self.verbose_logging:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            for handler in logging.getLogger().handlers:
                handler.setFormatter(
                    RedactingFormatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%H:%M:%S",
                    )
                )
            # Keep third-party libraries at WARNING level to reduce noise
            # We have our own retry and error logging that's more informative
            logging.getLogger("openai").setLevel(logging.WARNING)
            logging.getLogger("openai._base_client").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)
            logging.getLogger("asyncio").setLevel(logging.WARNING)
            # Suppress Modal/gRPC related debug spam
            logging.getLogger("hpack").setLevel(logging.WARNING)
            logging.getLogger("hpack.hpack").setLevel(logging.WARNING)
            logging.getLogger("grpc").setLevel(logging.WARNING)
            logging.getLogger("modal").setLevel(logging.WARNING)
            logging.getLogger("rex-deploy").setLevel(
                logging.INFO
            )  # Keep INFO for sandbox status
            logger.info("Verbose logging enabled (third-party library logs suppressed)")
        else:
            # Set logging to INFO level for important messages only
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            # Suppress noisy library logging
            logging.getLogger("openai").setLevel(logging.ERROR)
            logging.getLogger("openai._base_client").setLevel(logging.ERROR)
            logging.getLogger("httpx").setLevel(logging.ERROR)
            logging.getLogger("httpcore").setLevel(logging.ERROR)
            if self.quiet_mode:
                # In quiet mode (CLI default), suppress all tool/infra log
                # noise. The TUI has its own rich display for status; logger
                # INFO/WARNING messages just clutter it.
                for quiet_logger in [
                    "tools",  # all tools.* (terminal, browser, web, file, etc.)
                    "minisweagent",  # mini-swe-agent execution backend
                    "run_agent",  # agent runner internals
                    "trajectory_compressor",
                    "cron",  # scheduler (only relevant in daemon mode)
                    "hermes_cli",  # CLI helpers
                ]:
                    logging.getLogger(quiet_logger).setLevel(logging.ERROR)

        # Internal stream callback (set during streaming TTS).
        # Initialized here so _vprint can reference it before run_conversation.
        self._stream_callback = None
        # Deferred paragraph break flag — set after tool iterations so a
        # single "\n\n" is prepended to the next real text delta.
        self._stream_needs_break = False

        # Optional current-turn user-message override used when the API-facing
        # user message intentionally differs from the persisted transcript
        # (e.g. CLI voice mode adds a temporary prefix for the live call only).
        self._persist_user_message_idx = None
        self._persist_user_message_override = None

        # Cache anthropic image-to-text fallbacks per image payload/URL so a
        # single tool loop does not repeatedly re-run auxiliary vision on the
        # same image history.
        self._anthropic_image_fallback_cache: Dict[str, str] = {}

        # Initialize LLM client via centralized provider router.
        # The router handles auth resolution, base URL, headers, and
        # Codex/Anthropic wrapping for all known providers.
        # raw_codex=True because the main agent needs direct responses.stream()
        # access for Codex Responses API streaming.
        self._anthropic_client = None

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import (
                build_anthropic_client,
                resolve_anthropic_token,
            )

            # Only fall back to ANTHROPIC_TOKEN when the provider is actually Anthropic.
            # Other anthropic_messages providers (MiniMax, Alibaba, etc.) must use their own API key.
            # Falling back would send Anthropic credentials to third-party endpoints (Fixes #1739, #minimax-401).
            _is_native_anthropic = self.provider == "anthropic"
            effective_key = (
                (api_key or resolve_anthropic_token() or "")
                if _is_native_anthropic
                else (api_key or "")
            )
            self.api_key = effective_key
            self._anthropic_api_key = effective_key
            self._anthropic_base_url = base_url
            from agent.anthropic_adapter import _is_oauth_token as _is_oat

            self._is_anthropic_oauth = _is_oat(effective_key)
            self._anthropic_client = build_anthropic_client(effective_key, base_url)
            # No OpenAI client needed for Anthropic mode
            self.client = None
            self._client_kwargs = {}
            if not self.quiet_mode:
                print(
                    f"🤖 AI Agent initialized with model: {self.model} (Anthropic native)"
                )
                if effective_key and len(effective_key) > 12:
                    print(f"🔑 Using token: {effective_key[:8]}...{effective_key[-4:]}")
        else:
            if api_key and base_url:
                # Explicit credentials from CLI/gateway — construct directly.
                # The runtime provider resolver already handled auth for us.
                client_kwargs: dict[str, Any] = {}
                client_kwargs["api_key"] = api_key
                client_kwargs["base_url"] = base_url
                if self.provider == "copilot-acp":
                    client_kwargs["acp_command"] = self.acp_command
                    client_kwargs["acp_args"] = self.acp_args
                effective_base = base_url
                if "openrouter" in effective_base.lower():
                    client_kwargs["default_headers"] = {
                        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                        "X-OpenRouter-Title": "Hermes Agent",
                        "X-OpenRouter-Categories": "productivity,cli-agent",
                    }
                elif "api.githubcopilot.com" in effective_base.lower():
                    from hermes_cli.models import copilot_default_headers

                    client_kwargs["default_headers"] = copilot_default_headers()
                elif "api.kimi.com" in effective_base.lower():
                    client_kwargs["default_headers"] = {
                        "User-Agent": "KimiCLI/1.3",
                    }
            else:
                # No explicit creds — use the centralized provider router
                from agent.auxiliary_client import resolve_provider_client

                _routed_client, _ = resolve_provider_client(
                    self.provider or "auto", model=self.model, raw_codex=True
                )
                if _routed_client is not None:
                    client_kwargs = {
                        "api_key": _routed_client.api_key,
                        "base_url": str(_routed_client.base_url),
                    }
                    # Preserve any default_headers the router set
                    if (
                        hasattr(_routed_client, "_default_headers")
                        and _routed_client._default_headers
                    ):
                        client_kwargs["default_headers"] = dict(
                            _routed_client._default_headers
                        )
                else:
                    # When the user explicitly chose a non-OpenRouter provider
                    # but no credentials were found, fail fast with a clear
                    # message instead of silently routing through OpenRouter.
                    _explicit = (self.provider or "").strip().lower()
                    if _explicit and _explicit not in ("auto", "openrouter", "custom"):
                        raise RuntimeError(
                            f"Provider '{_explicit}' is set in config.yaml but no API key "
                            f"was found. Set the {_explicit.upper()}_API_KEY environment "
                            f"variable, or switch to a different provider with `hermes model`."
                        )
                    # Final fallback: try raw OpenRouter key
                    client_kwargs = {
                        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
                        "base_url": OPENROUTER_BASE_URL,
                        "default_headers": {
                            "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                            "X-OpenRouter-Title": "Hermes Agent",
                            "X-OpenRouter-Categories": "productivity,cli-agent",
                        },
                    }

            self._client_kwargs = client_kwargs  # stored for rebuilding after interrupt
            self.api_key = client_kwargs.get("api_key", "")
            try:
                self.client = self._create_openai_client(
                    client_kwargs, reason="agent_init", shared=True
                )
                if not self.quiet_mode:
                    print(f"🤖 AI Agent initialized with model: {self.model}")
                    if base_url:
                        print(f"🔗 Using custom base URL: {base_url}")
                    # Always show API key info (masked) for debugging auth issues
                    key_used = client_kwargs.get("api_key", "none")
                    if key_used and key_used != "dummy-key" and len(key_used) > 12:
                        print(f"🔑 Using API key: {key_used[:8]}...{key_used[-4:]}")
                    else:
                        print(
                            f"⚠️  Warning: API key appears invalid or missing (got: '{key_used[:20] if key_used else 'none'}...')"
                        )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

        # Provider fallback — a single backup model/provider tried when the
        # primary is exhausted (rate-limit, overload, connection failure).
        # Config shape: {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}
        self._fallback_model = (
            fallback_model if isinstance(fallback_model, dict) else None
        )
        self._fallback_activated = False
        if self._fallback_model:
            fb_p = self._fallback_model.get("provider", "")
            fb_m = self._fallback_model.get("model", "")
            if fb_p and fb_m and not self.quiet_mode:
                print(f"🔄 Fallback model: {fb_m} ({fb_p})")

        # Get available tools with filtering
        self.tools = get_tool_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=self.quiet_mode,
        )

        # Show tool configuration and store valid tool names for validation
        self.valid_tool_names = set()
        if self.tools:
            self.valid_tool_names = {tool["function"]["name"] for tool in self.tools}
            tool_names = sorted(self.valid_tool_names)
            if not self.quiet_mode:
                print(f"🛠️  Loaded {len(self.tools)} tools: {', '.join(tool_names)}")

                # Show filtering info if applied
                if enabled_toolsets:
                    print(f"   ✅ Enabled toolsets: {', '.join(enabled_toolsets)}")
                if disabled_toolsets:
                    print(f"   ❌ Disabled toolsets: {', '.join(disabled_toolsets)}")
        elif not self.quiet_mode:
            print("🛠️  No tools loaded (all tools filtered out or unavailable)")

        # Check tool requirements
        if self.tools and not self.quiet_mode:
            requirements = check_toolset_requirements()
            missing_reqs = [
                name for name, available in requirements.items() if not available
            ]
            if missing_reqs:
                print(
                    f"⚠️  Some tools may not work due to missing requirements: {missing_reqs}"
                )

        # Show trajectory saving status
        if self.save_trajectories and not self.quiet_mode:
            print("📝 Trajectory saving enabled")

        # Show ephemeral system prompt status
        if self.ephemeral_system_prompt and not self.quiet_mode:
            prompt_preview = (
                self.ephemeral_system_prompt[:60] + "..."
                if len(self.ephemeral_system_prompt) > 60
                else self.ephemeral_system_prompt
            )
            print(
                f"🔒 Ephemeral system prompt: '{prompt_preview}' (not saved to trajectories)"
            )

        # Show prompt caching status
        if self._use_prompt_caching and not self.quiet_mode:
            source = (
                "native Anthropic" if is_native_anthropic else "Claude via OpenRouter"
            )
            print(f"💾 Prompt caching: ENABLED ({source}, {self._cache_ttl} TTL)")

        # Session logging setup - auto-save conversation trajectories for debugging
        self.session_start = datetime.now()
        if session_id:
            # Use provided session ID (e.g., from CLI)
            self.session_id = session_id
        else:
            # Generate a new session ID
            timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:6]
            self.session_id = f"{timestamp_str}_{short_uuid}"

        # Session logs go into ~/.hermes/sessions/ alongside gateway sessions
        hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
        self.logs_dir = hermes_home / "sessions"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"

        # Track conversation messages for session logging
        self._session_messages: List[Dict[str, Any]] = []

        # Cached system prompt -- built once per session, only rebuilt on compression
        self._cached_system_prompt: Optional[str] = None

        # Filesystem checkpoint manager (transparent — not a tool)
        from tools.checkpoint_manager import CheckpointManager

        self._checkpoint_mgr = CheckpointManager(
            enabled=checkpoints_enabled,
            max_snapshots=checkpoint_max_snapshots,
        )

        # SQLite session store (optional -- provided by CLI or gateway)
        self._session_db = session_db
        self._last_flushed_db_idx = (
            0  # tracks DB-write cursor to prevent duplicate writes
        )
        if self._session_db:
            try:
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or "cli",
                    model=self.model,
                    model_config={
                        "max_iterations": self.max_iterations,
                        "reasoning_config": reasoning_config,
                        "max_tokens": max_tokens,
                    },
                    user_id=None,
                )
            except Exception as e:
                logger.debug("Session DB create_session failed: %s", e)

        # In-memory todo list for task planning (one per agent/session)
        from tools.todo_tool import TodoStore

        self._todo_store = TodoStore()

        # Token efficiency managers
        self._prompt_cache = PromptCache()
        self._rate_limiter = CoordinatedRateLimiter(
            provider=self.provider, coordinated=True
        )
        self._context_compactor = ContextCompactionManager()

        # Load config once for memory, skills, and compression sections
        try:
            from hermes_cli.config import load_config as _load_agent_config

            _agent_cfg = _load_agent_config()
        except Exception:
            _agent_cfg = {}

        # Persistent memory (MEMORY.md + USER.md) -- loaded from disk
        self._memory_store = None
        self._memory_enabled = False
        self._user_profile_enabled = False
        self._memory_nudge_interval = 10
        self._memory_flush_min_turns = 6
        self._turns_since_memory = 0
        self._iters_since_skill = 0
        if not skip_memory:
            try:
                mem_config = _agent_cfg.get("memory", {})
                self._memory_enabled = mem_config.get("memory_enabled", False)
                self._user_profile_enabled = mem_config.get(
                    "user_profile_enabled", False
                )
                self._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))
                self._memory_flush_min_turns = int(mem_config.get("flush_min_turns", 6))
                if self._memory_enabled or self._user_profile_enabled:
                    from tools.memory_tool import MemoryStore

                    self._memory_store = MemoryStore(
                        memory_char_limit=mem_config.get("memory_char_limit", 2200),
                        user_char_limit=mem_config.get("user_char_limit", 1375),
                    )
                    self._memory_store.load_from_disk()
            except Exception:
                pass  # Memory is optional -- don't break agent init

        # Honcho AI-native memory (cross-session user modeling)
        # Reads $HERMES_HOME/honcho.json (instance) or ~/.honcho/config.json (global).
        self._honcho = None  # HonchoSessionManager | None
        self._honcho_session_key = honcho_session_key
        self._honcho_config = None  # HonchoClientConfig | None
        self._honcho_exit_hook_registered = False
        if not skip_memory:
            try:
                if honcho_manager is not None:
                    hcfg = honcho_config or getattr(honcho_manager, "_config", None)
                    self._honcho_config = hcfg
                    if hcfg and self._honcho_should_activate(hcfg):
                        self._honcho = honcho_manager
                        self._activate_honcho(
                            hcfg,
                            enabled_toolsets=enabled_toolsets,
                            disabled_toolsets=disabled_toolsets,
                            session_db=session_db,
                        )
                else:
                    from honcho_integration.client import (
                        HonchoClientConfig,
                        get_honcho_client,
                    )

                    hcfg = HonchoClientConfig.from_global_config()
                    self._honcho_config = hcfg
                    if self._honcho_should_activate(hcfg):
                        from honcho_integration.session import HonchoSessionManager

                        client = get_honcho_client(hcfg)
                        self._honcho = HonchoSessionManager(
                            honcho=client,
                            config=hcfg,
                            context_tokens=hcfg.context_tokens,
                        )
                        self._activate_honcho(
                            hcfg,
                            enabled_toolsets=enabled_toolsets,
                            disabled_toolsets=disabled_toolsets,
                            session_db=session_db,
                        )
                    else:
                        if not hcfg.enabled:
                            logger.debug("Honcho disabled in global config")
                        elif not hcfg.api_key:
                            logger.debug("Honcho enabled but no API key configured")
                        else:
                            logger.debug(
                                "Honcho enabled but missing API key or disabled in config"
                            )
            except Exception as e:
                logger.warning("Honcho init failed — memory disabled: %s", e)
                print(f"  Honcho init failed: {e}")
                print("  Run 'hermes honcho setup' to reconfigure.")
                self._honcho = None

        # Tools are initially discovered before Honcho activation. If Honcho
        # stays inactive, remove any stale honcho_* tools from prior process state.
        if not self._honcho:
            self._strip_honcho_tools_from_surface()

        # Gate local memory writes based on per-peer memory modes.
        # AI peer governs MEMORY.md; user peer governs USER.md.
        # "honcho" = Honcho only, disable local writes.
        if self._honcho_config and self._honcho:
            _hcfg = self._honcho_config
            _agent_mode = _hcfg.peer_memory_mode(_hcfg.ai_peer)
            _user_mode = _hcfg.peer_memory_mode(_hcfg.peer_name or "user")
            if _agent_mode == "honcho":
                self._memory_flush_min_turns = 0
                self._memory_enabled = False
                logger.debug(
                    "peer %s memory_mode=honcho: local MEMORY.md writes disabled",
                    _hcfg.ai_peer,
                )
            if _user_mode == "honcho":
                self._user_profile_enabled = False
                logger.debug(
                    "peer %s memory_mode=honcho: local USER.md writes disabled",
                    _hcfg.peer_name or "user",
                )

        # Skills config: nudge interval for skill creation reminders
        self._skill_nudge_interval = 10
        try:
            skills_config = _agent_cfg.get("skills", {})
            self._skill_nudge_interval = int(
                skills_config.get("creation_nudge_interval", 10)
            )
        except Exception:
            pass

        # Initialize context compressor for automatic context management
        # Compresses conversation when approaching model's context limit
        # Configuration via config.yaml (compression section)
        _compression_cfg = _agent_cfg.get("compression", {})
        if not isinstance(_compression_cfg, dict):
            _compression_cfg = {}
        compression_threshold = float(_compression_cfg.get("threshold", 0.50))
        compression_enabled = str(_compression_cfg.get("enabled", True)).lower() in (
            "true",
            "1",
            "yes",
        )
        compression_summary_model = _compression_cfg.get("summary_model") or None

        # Read explicit context_length override from model config
        _model_cfg = _agent_cfg.get("model", {})
        if isinstance(_model_cfg, dict):
            _config_context_length = _model_cfg.get("context_length")
        else:
            _config_context_length = None
        if _config_context_length is not None:
            try:
                _config_context_length = int(_config_context_length)
            except (TypeError, ValueError):
                _config_context_length = None

        # Check custom_providers per-model context_length
        if _config_context_length is None:
            _custom_providers = _agent_cfg.get("custom_providers")
            if isinstance(_custom_providers, list):
                for _cp_entry in _custom_providers:
                    if not isinstance(_cp_entry, dict):
                        continue
                    _cp_url = (_cp_entry.get("base_url") or "").rstrip("/")
                    if _cp_url and _cp_url == self.base_url.rstrip("/"):
                        _cp_models = _cp_entry.get("models", {})
                        if isinstance(_cp_models, dict):
                            _cp_model_cfg = _cp_models.get(self.model, {})
                            if isinstance(_cp_model_cfg, dict):
                                _cp_ctx = _cp_model_cfg.get("context_length")
                                if _cp_ctx is not None:
                                    try:
                                        _config_context_length = int(_cp_ctx)
                                    except (TypeError, ValueError):
                                        pass
                        break

        try:
            self.context_compressor = ContextCompressor(
                model=self.model,
                threshold_percent=compression_threshold,
                protect_first_n=3,
                protect_last_n=4,
                summary_target_tokens=500,
                summary_model_override=compression_summary_model,
                quiet_mode=self.quiet_mode,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                config_context_length=_config_context_length,
                provider=self.provider,
            )
        except Exception as e:
            logger.warning("Failed to initialize context compressor: %s", e)
            self.context_compressor = None
            compression_enabled = False
        self.compression_enabled = compression_enabled
        self._user_turn_count = 0

        # Cumulative token usage for the session
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_api_calls = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"

        if not self.quiet_mode:
            context_length = getattr(self.context_compressor, "context_length", 128000)
            threshold_tokens = getattr(
                self.context_compressor, "threshold_tokens", 64000
            )
            if compression_enabled:
                print(
                    f"📊 Context limit: {context_length:,} tokens (compress at {int(compression_threshold*100)}% = {threshold_tokens:,})"
                )
            else:
                print(
                    f"📊 Context limit: {context_length:,} tokens (auto-compression disabled)"
                )

    def _safe_print(self, *args, **kwargs):
        """Print that silently handles broken pipes / closed stdout.

        In headless environments (systemd, Docker, nohup) stdout may become
        unavailable mid-session.  A raw ``print()`` raises ``OSError`` which
        can crash cron jobs and lose completed work.

        Internally routes through ``self._print_fn`` (default: builtin
        ``print``) so callers such as the CLI can inject a renderer that
        handles ANSI escape sequences properly (e.g. prompt_toolkit's
        ``print_formatted_text(ANSI(...))``) without touching this method.
        """
        try:
            fn = self._print_fn or print
            fn(*args, **kwargs)
        except OSError:
            pass

    def _vprint(self, *args, force: bool = False, **kwargs):
        """Verbose print — suppressed when actively streaming tokens.

        Pass ``force=True`` for error/warning messages that should always be
        shown even during streaming playback (TTS or display).

        During tool execution (``_executing_tools`` is True), printing is
        allowed even with stream consumers registered because no tokens
        are being streamed at that point.

        After the main response has been delivered and the remaining tool
        calls are post-response housekeeping (``_mute_post_response``),
        all non-forced output is suppressed.
        """
        if not force and getattr(self, "_mute_post_response", False):
            return
        if not force and self._has_stream_consumers() and not self._executing_tools:
            return
        self._safe_print(*args, **kwargs)

    def _is_direct_openai_url(self, base_url: Optional[str] = None) -> bool:
        """Return True when a base URL targets OpenAI's native API."""
        url = (base_url or self._base_url_lower).lower()
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

    def _has_content_after_think_block(self, content: str) -> bool:
        """
        Check if content has actual text after any reasoning/thinking blocks.

        This detects cases where the model only outputs reasoning but no actual
        response, which indicates an incomplete generation that should be retried.
        Must stay in sync with _strip_think_blocks() tag variants.

        Args:
            content: The assistant message content to check

        Returns:
            True if there's meaningful content after think blocks, False otherwise
        """
        if not content:
            return False

        # Remove all reasoning tag variants (must match _strip_think_blocks)
        cleaned = self._strip_think_blocks(content)

        # Check if there's any non-whitespace content remaining
        return bool(cleaned.strip())

    def _strip_think_blocks(self, content: str) -> str:
        """Remove reasoning/thinking blocks from content, returning only visible text."""
        if not content:
            return ""
        # Strip all reasoning tag variants: <think>, <thinking>, <THINKING>,
        # <reasoning>, <REASONING_SCRATCHPAD>
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = re.sub(
            r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL | re.IGNORECASE
        )
        content = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)
        content = re.sub(
            r"<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>",
            "",
            content,
            flags=re.DOTALL,
        )
        return content

    def _looks_like_codex_intermediate_ack(
        self,
        user_message: str,
        assistant_content: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        """Detect a planning/ack message that should continue instead of ending the turn."""
        if any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages):
            return False

        assistant_text = (
            self._strip_think_blocks(assistant_content or "").strip().lower()
        )
        if not assistant_text:
            return False
        if len(assistant_text) > 1200:
            return False

        has_future_ack = bool(
            re.search(
                r"\b(i['’]ll|i will|let me|i can do that|i can help with that)\b",
                assistant_text,
            )
        )
        if not has_future_ack:
            return False

        action_markers = (
            "look into",
            "look at",
            "inspect",
            "scan",
            "check",
            "analyz",
            "review",
            "explore",
            "read",
            "open",
            "run",
            "test",
            "fix",
            "debug",
            "search",
            "find",
            "walkthrough",
            "report back",
            "summarize",
        )
        workspace_markers = (
            "directory",
            "current directory",
            "current dir",
            "cwd",
            "repo",
            "repository",
            "codebase",
            "project",
            "folder",
            "filesystem",
            "file tree",
            "files",
            "path",
        )

        user_text = (user_message or "").strip().lower()
        user_targets_workspace = (
            any(marker in user_text for marker in workspace_markers)
            or "~/" in user_text
            or "/" in user_text
        )
        assistant_mentions_action = any(
            marker in assistant_text for marker in action_markers
        )
        assistant_targets_workspace = any(
            marker in assistant_text for marker in workspace_markers
        )
        return (
            user_targets_workspace or assistant_targets_workspace
        ) and assistant_mentions_action

    def _extract_reasoning(self, assistant_message) -> Optional[str]:
        """
        Extract reasoning/thinking content from an assistant message.

        OpenRouter and various providers can return reasoning in multiple formats:
        1. message.reasoning - Direct reasoning field (DeepSeek, Qwen, etc.)
        2. message.reasoning_content - Alternative field (Moonshot AI, Novita, etc.)
        3. message.reasoning_details - Array of {type, summary, ...} objects (OpenRouter unified)

        Args:
            assistant_message: The assistant message object from the API response

        Returns:
            Combined reasoning text, or None if no reasoning found
        """
        reasoning_parts = []

        # Check direct reasoning field
        if hasattr(assistant_message, "reasoning") and assistant_message.reasoning:
            reasoning_parts.append(assistant_message.reasoning)

        # Check reasoning_content field (alternative name used by some providers)
        if (
            hasattr(assistant_message, "reasoning_content")
            and assistant_message.reasoning_content
        ):
            # Don't duplicate if same as reasoning
            if assistant_message.reasoning_content not in reasoning_parts:
                reasoning_parts.append(assistant_message.reasoning_content)

        # Check reasoning_details array (OpenRouter unified format)
        # Format: [{"type": "reasoning.summary", "summary": "...", ...}, ...]
        if (
            hasattr(assistant_message, "reasoning_details")
            and assistant_message.reasoning_details
        ):
            for detail in assistant_message.reasoning_details:
                if isinstance(detail, dict):
                    # Extract summary from reasoning detail object
                    summary = (
                        detail.get("summary")
                        or detail.get("content")
                        or detail.get("text")
                    )
                    if summary and summary not in reasoning_parts:
                        reasoning_parts.append(summary)

        # Combine all reasoning parts
        if reasoning_parts:
            return "\n\n".join(reasoning_parts)

        return None

    def _cleanup_task_resources(self, task_id: str) -> None:
        """Clean up VM and browser resources for a given task."""
        try:
            cleanup_vm(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup VM for task {task_id}: {e}")
        try:
            cleanup_browser(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup browser for task {task_id}: {e}")

    # ------------------------------------------------------------------
    # Background memory/skill review
    # ------------------------------------------------------------------

    _MEMORY_REVIEW_PROMPT = (
        "Review the conversation above and consider saving to memory if appropriate.\n\n"
        "Focus on:\n"
        "1. Has the user revealed things about themselves — their persona, desires, "
        "preferences, or personal details worth remembering?\n"
        "2. Has the user expressed expectations about how you should behave, their work "
        "style, or ways they want you to operate?\n\n"
        "If something stands out, save it using the memory tool. "
        "If nothing is worth saving, just say 'Nothing to save.' and stop."
    )

    _SKILL_REVIEW_PROMPT = (
        "Review the conversation above and consider saving or updating a skill if appropriate.\n\n"
        "Focus on: was a non-trivial approach used to complete a task that required trial "
        "and error, or changing course due to experiential findings along the way, or did "
        "the user expect or desire a different method or outcome?\n\n"
        "If a relevant skill already exists, update it with what you learned. "
        "Otherwise, create a new skill if the approach is reusable.\n"
        "If nothing is worth saving, just say 'Nothing to save.' and stop."
    )

    _COMBINED_REVIEW_PROMPT = (
        "Review the conversation above and consider two things:\n\n"
        "**Memory**: Has the user revealed things about themselves — their persona, "
        "desires, preferences, or personal details? Has the user expressed expectations "
        "about how you should behave, their work style, or ways they want you to operate? "
        "If so, save using the memory tool.\n\n"
        "**Skills**: Was a non-trivial approach used to complete a task that required trial "
        "and error, or changing course due to experiential findings along the way, or did "
        "the user expect or desire a different method or outcome? If a relevant skill "
        "already exists, update it. Otherwise, create a new one if the approach is reusable.\n\n"
        "Only act if there's something genuinely worth saving. "
        "If nothing stands out, just say 'Nothing to save.' and stop."
    )

    def run_conversation(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        task_id: Optional[str] = None,
        stream_callback: Optional[Callable] = None,
        persist_user_message: Optional[str] = None,
        sync_honcho: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a complete conversation with tool calling until completion.

        Args:
            user_message (str): The user's message/question
            system_message (str): Custom system message (optional, overrides ephemeral_system_prompt if provided)
            conversation_history (List[Dict]): Previous conversation messages (optional)
            task_id (str): Unique identifier for this task to isolate VMs between concurrent tasks (optional, auto-generated if not provided)
            stream_callback: Optional callback invoked with each text delta during streaming.
                Used by the TTS pipeline to start audio generation before the full response.
                When None (default), API calls use the standard non-streaming path.
            persist_user_message: Optional clean user message to store in
                transcripts/history when user_message contains API-only
                synthetic prefixes.
            sync_honcho: When False, skip writing the final synthetic turn back
                to Honcho or queuing follow-up prefetch work.

        Returns:
            Dict: Complete conversation result with final response and message history
        """
        messages, active_system_prompt, effective_task_id, current_turn_user_idx, _should_review_memory, original_user_message = self._initialize_conversation(  # type: ignore
            user_message,
            system_message,
            conversation_history,
            task_id,
            stream_callback,
            persist_user_message,
        )

        messages, active_system_prompt = self._perform_preflight_compression(  # type: ignore
            messages, active_system_prompt, system_message, effective_task_id
        )

        # Main conversation loop
        return self._run_conversation_loop(  # type: ignore
            messages,
            active_system_prompt,
            effective_task_id,
            current_turn_user_idx,
            _should_review_memory,
            original_user_message,
            sync_honcho,
        )

    def _spawn_background_review(
        self,
        messages_snapshot: List[Dict],
        review_memory: bool = False,
        review_skills: bool = False,
    ) -> None:
        """Spawn a background thread to review the conversation for memory/skill saves.

        Creates a full AIAgent fork with the same model, tools, and context as the
        main session. The review prompt is appended as the next user turn in the
        forked conversation. Writes directly to the shared memory/skill stores.
        Never modifies the main conversation history or produces user-visible output.
        """
        import threading

        # Pick the right prompt based on which triggers fired
        if review_memory and review_skills:
            prompt = self._COMBINED_REVIEW_PROMPT
        elif review_memory:
            prompt = self._MEMORY_REVIEW_PROMPT
        else:
            prompt = self._SKILL_REVIEW_PROMPT

        def _run_review():
            import contextlib, os as _os

            review_agent: Optional["AIAgent"] = None
            try:
                with (
                    open(_os.devnull, "w") as _devnull,
                    contextlib.redirect_stdout(_devnull),
                    contextlib.redirect_stderr(_devnull),
                ):
                    review_agent = AIAgent(
                        model=self.model,
                        max_iterations=8,
                        quiet_mode=True,
                        platform=self.platform,
                        provider=self.provider,
                    )
                    review_agent._memory_store = self._memory_store
                    review_agent._memory_enabled = self._memory_enabled
                    review_agent._user_profile_enabled = self._user_profile_enabled
                    review_agent._memory_nudge_interval = 0
                    review_agent._skill_nudge_interval = 0

                    review_agent.run_conversation(  # type: ignore
                        user_message=prompt,
                        conversation_history=messages_snapshot,
                    )

                # Scan the review agent's messages for successful tool actions
                # and surface a compact summary to the user.
                actions = []
                for msg in getattr(review_agent, "_session_messages", []):
                    if not isinstance(msg, dict) or msg.get("role") != "tool":
                        continue
                    try:
                        data = json.loads(msg.get("content", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not data.get("success"):
                        continue
                    message = data.get("message", "")
                    target = data.get("target", "")
                    if "created" in message.lower():
                        actions.append(message)
                    elif "updated" in message.lower():
                        actions.append(message)
                    elif "added" in message.lower() or (
                        target and "add" in message.lower()
                    ):
                        label = (
                            "Memory"
                            if target == "memory"
                            else "User profile" if target == "user" else target
                        )
                        actions.append(f"{label} updated")
                    elif "Entry added" in message:
                        label = (
                            "Memory"
                            if target == "memory"
                            else "User profile" if target == "user" else target
                        )
                        actions.append(f"{label} updated")
                    elif "removed" in message.lower() or "replaced" in message.lower():
                        label = (
                            "Memory"
                            if target == "memory"
                            else "User profile" if target == "user" else target
                        )
                        actions.append(f"{label} updated")

                if actions:
                    summary = " · ".join(dict.fromkeys(actions))
                    self._safe_print(f"  💾 {summary}")

            except Exception as e:
                logger.debug("Background memory/skill review failed: %s", e)
            finally:
                # Explicitly close the OpenAI/httpx client so GC doesn't
                # try to clean it up on a dead asyncio event loop (which
                # produces "Event loop is closed" errors in the terminal).
                if review_agent is not None:
                    client = getattr(review_agent, "client", None)
                    if client is not None:
                        try:
                            review_agent._close_openai_client(
                                client, reason="bg_review_done", shared=True
                            )
                            review_agent.client = None
                        except Exception:
                            pass

        t = threading.Thread(target=_run_review, daemon=True, name="bg-review")
        t.start()

    def _apply_persist_user_message_override(self, messages: List[Dict]) -> None:
        """Rewrite the current-turn user message before persistence/return.

        Some call paths need an API-only user-message variant without letting
        that synthetic text leak into persisted transcripts or resumed session
        history. When an override is configured for the active turn, mutate the
        in-memory messages list in place so both persistence and returned
        history stay clean.
        """
        idx = getattr(self, "_persist_user_message_idx", None)
        override = getattr(self, "_persist_user_message_override", None)
        if override is None or idx is None:
            return
        if 0 <= idx < len(messages):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg["content"] = override

    def _get_messages_up_to_last_assistant(self, messages: List[Dict]) -> List[Dict]:
        """
        Get messages up to (but not including) the last assistant turn.

        This is used when we need to "roll back" to the last successful point
        in the conversation, typically when the final assistant message is
        incomplete or malformed.

        Args:
            messages: Full message list

        Returns:
            Messages up to the last complete assistant turn (ending with user/tool message)
        """
        if not messages:
            return []

        # Find the index of the last assistant message
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            # No assistant message found, return all messages
            return messages.copy()

        # Return everything up to (not including) the last assistant message
        return messages[:last_assistant_idx]

    def _format_tools_for_system_message(self) -> str:
        """
        Format tool definitions for the system message in the trajectory format.

        Returns:
            str: JSON string representation of tool definitions
        """
        if not self.tools:
            return "[]"

        # Convert tool definitions to the format expected in trajectories
        formatted_tools = []
        for tool in self.tools:
            func = tool["function"]
            formatted_tool = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "required": None,  # Match the format in the example
            }
            formatted_tools.append(formatted_tool)

        return json.dumps(formatted_tools, ensure_ascii=False)

    def _convert_to_trajectory_format(
        self, messages: List[Dict[str, Any]], user_query: str, completed: bool
    ) -> List[Dict[str, Any]]:
        """
        Convert internal message format to trajectory format for saving.

        Args:
            messages (List[Dict]): Internal message history
            user_query (str): Original user query
            completed (bool): Whether the conversation completed successfully

        Returns:
            List[Dict]: Messages in trajectory format
        """
        trajectory = []

        # Add system message with tool definitions
        system_msg = (
            "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
            "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
            "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
            "into functions. After calling & executing the functions, you will be provided with function results within "
            "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
            f"<tools>\n{self._format_tools_for_system_message()}\n</tools>\n"
            "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
            "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
            "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
            "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
            "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
        )

        trajectory.append({"from": "system", "value": system_msg})

        # Add the actual user prompt (from the dataset) as the first human message
        trajectory.append({"from": "human", "value": user_query})

        # Skip the first message (the user query) since we already added it above.
        # Prefill messages are injected at API-call time only (not in the messages
        # list), so no offset adjustment is needed here.
        i = 1

        while i < len(messages):
            msg = messages[i]

            if msg["role"] == "assistant":
                # Check if this message has tool calls
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Format assistant message with tool calls
                    # Add <think> tags around reasoning for trajectory storage
                    content = ""

                    # Prepend reasoning in <think> tags if available (native thinking tokens)
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"

                    if msg.get("content") and msg["content"].strip():
                        # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                        # (used when native thinking is disabled and model reasons via XML)
                        content += convert_scratchpad_to_think(msg["content"]) + "\n"

                    # Add tool calls wrapped in XML tags
                    for tool_call in msg["tool_calls"]:
                        if not tool_call or not isinstance(tool_call, dict):
                            continue
                        # Parse arguments - should always succeed since we validate during conversation
                        # but keep try-except as safety net
                        try:
                            arguments = (
                                json.loads(tool_call["function"]["arguments"])
                                if isinstance(tool_call["function"]["arguments"], str)
                                else tool_call["function"]["arguments"]
                            )
                        except json.JSONDecodeError:
                            # This shouldn't happen since we validate and retry during conversation,
                            # but if it does, log warning and use empty dict
                            logging.warning(
                                f"Unexpected invalid JSON in trajectory conversion: {tool_call['function']['arguments'][:100]}"
                            )
                            arguments = {}

                        tool_call_json = {
                            "name": tool_call["function"]["name"],
                            "arguments": arguments,
                        }
                        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"

                    # Ensure every gpt turn has a <think> block (empty if no reasoning)
                    # so the format is consistent for training data
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content

                    trajectory.append({"from": "gpt", "value": content.rstrip()})

                    # Collect all subsequent tool responses
                    tool_responses = []
                    j = i + 1
                    while j < len(messages) and messages[j]["role"] == "tool":
                        tool_msg = messages[j]
                        # Format tool response with XML tags
                        tool_response = f"<tool_response>\n"

                        # Try to parse tool content as JSON if it looks like JSON
                        tool_content = tool_msg["content"]
                        try:
                            if tool_content.strip().startswith(("{", "[")):
                                tool_content = json.loads(tool_content)
                        except (json.JSONDecodeError, AttributeError):
                            pass  # Keep as string if not valid JSON

                        tool_index = len(tool_responses)
                        tool_name = (
                            msg["tool_calls"][tool_index]["function"]["name"]
                            if tool_index < len(msg["tool_calls"])
                            else "unknown"
                        )
                        tool_response += json.dumps(
                            {
                                "tool_call_id": tool_msg.get("tool_call_id", ""),
                                "name": tool_name,
                                "content": tool_content,
                            },
                            ensure_ascii=False,
                        )
                        tool_response += "\n</tool_response>"
                        tool_responses.append(tool_response)
                        j += 1

                    # Add all tool responses as a single message
                    if tool_responses:
                        trajectory.append(
                            {"from": "tool", "value": "\n".join(tool_responses)}
                        )
                        i = j - 1  # Skip the tool messages we just processed

                else:
                    # Regular assistant message without tool calls
                    # Add <think> tags around reasoning for trajectory storage
                    content = ""

                    # Prepend reasoning in <think> tags if available (native thinking tokens)
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"

                    # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                    # (used when native thinking is disabled and model reasons via XML)
                    raw_content = msg["content"] or ""
                    content += convert_scratchpad_to_think(raw_content)

                    # Ensure every gpt turn has a <think> block (empty if no reasoning)
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content

                    trajectory.append({"from": "gpt", "value": content.strip()})

            elif msg["role"] == "user":
                trajectory.append({"from": "human", "value": msg["content"]})

            i += 1

        return trajectory

    def _save_trajectory(
        self, messages: List[Dict[str, Any]], user_query: str, completed: bool
    ):
        """
        Save conversation trajectory to JSONL file.

        Args:
            messages (List[Dict]): Complete message history
            user_query (str): Original user query
            completed (bool): Whether the conversation completed successfully
        """
        if not self.save_trajectories:
            return

        trajectory = self._convert_to_trajectory_format(messages, user_query, completed)
        _save_trajectory_to_file(trajectory, self.model, completed)

    def _mask_api_key_for_logs(self, key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        if len(key) <= 12:
            return "***"
        return f"{key[:8]}...{key[-4:]}"

    def _dump_api_request_debug(
        self,
        api_kwargs: Dict[str, Any],
        *,
        reason: str,
        error: Optional[Exception] = None,
    ) -> Optional[Path]:
        """
        Dump a debug-friendly HTTP request record for the active inference API.

        Captures the request body from api_kwargs (excluding transport-only keys
        like timeout). Intended for debugging provider-side 4xx failures where
        retries are not useful.
        """
        try:
            body = copy.deepcopy(api_kwargs)
            body.pop("timeout", None)
            body = {k: v for k, v in body.items() if v is not None}

            api_key = None
            try:
                api_key = getattr(self.client, "api_key", None)
            except Exception as e:
                logger.debug("Could not extract API key for debug dump: %s", e)

            dump_payload: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "reason": reason,
                "request": {
                    "method": "POST",
                    "url": f"{self.base_url.rstrip('/')}{'/responses' if self.api_mode == 'codex_responses' else '/chat/completions'}",
                    "headers": {
                        "Authorization": f"Bearer {self._mask_api_key_for_logs(api_key)}",
                        "Content-Type": "application/json",
                    },
                    "body": body,
                },
            }

            if error is not None:
                error_info: Dict[str, Any] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                for attr_name in ("status_code", "request_id", "code", "param", "type"):
                    attr_value = getattr(error, attr_name, None)
                    if attr_value is not None:
                        error_info[attr_name] = attr_value

                body_attr = getattr(error, "body", None)
                if body_attr is not None:
                    error_info["body"] = body_attr

                response_obj = getattr(error, "response", None)
                if response_obj is not None:
                    try:
                        error_info["response_status"] = getattr(
                            response_obj, "status_code", None
                        )
                        error_info["response_text"] = response_obj.text
                    except Exception as e:
                        logger.debug("Could not extract error response details: %s", e)

                dump_payload["error"] = error_info

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dump_file = (
                self.logs_dir / f"request_dump_{self.session_id}_{timestamp}.json"
            )
            dump_file.write_text(
                json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            self._vprint(
                f"{self.log_prefix}🧾 Request debug dump written to: {dump_file}"
            )

            if os.getenv("HERMES_DUMP_REQUEST_STDOUT", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                print(
                    json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str)
                )

            return dump_file
        except Exception as dump_error:
            if self.verbose_logging:
                logging.warning(
                    f"Failed to dump API request debug payload: {dump_error}"
                )
            return None

    @staticmethod
    def interrupt(self, message: Optional[str] = None) -> None:
        """
        Request the agent to interrupt its current tool-calling loop.

        Call this from another thread (e.g., input handler, message receiver)
        to gracefully stop the agent and process a new message.

        Also signals long-running tool executions (e.g. terminal commands)
        to terminate early, so the agent can respond immediately.

        Args:
            message: Optional new message that triggered the interrupt.
                     If provided, the agent will include this in its response context.

        Example (CLI):
            # In a separate input thread:
            if user_typed_something:
                agent.interrupt(user_input)

        Example (Messaging):
            # When new message arrives for active session:
            if session_has_running_agent:
                running_agent.interrupt(new_message.text)
        """
        self._interrupt_requested = True
        self._interrupt_message = message
        # Signal all tools to abort any in-flight operations immediately
        _set_interrupt(True)
        # Propagate interrupt to any running child agents (subagent delegation)
        with self._active_children_lock:
            children_copy = list(self._active_children)
        for child in children_copy:
            try:
                child.interrupt(message)
            except Exception as e:
                logger.debug("Failed to propagate interrupt to child agent: %s", e)
        if not self.quiet_mode:
            print(
                f"\n⚡ Interrupt requested"
                + (
                    f": '{message[:40]}...'"
                    if message and len(message) > 40
                    else f": '{message}'" if message else ""
                )
            )

    def clear_interrupt(self) -> None:
        """Clear any pending interrupt request and the global tool interrupt signal."""
        self._interrupt_requested = False
        self._interrupt_message = None
        _set_interrupt(False)

    def _hydrate_todo_store(self, history: List[Dict[str, Any]]) -> None:
        """
        Recover todo state from conversation history.

        The gateway creates a fresh AIAgent per message, so the in-memory
        TodoStore is empty. We scan the history for the most recent todo
        tool response and replay it to reconstruct the state.
        """
        # Walk history backwards to find the most recent todo tool response
        last_todo_response = None
        for msg in reversed(history):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            # Quick check: todo responses contain "todos" key
            if '"todos"' not in content:
                continue
            try:
                data = json.loads(content)
                if "todos" in data and isinstance(data["todos"], list):
                    last_todo_response = data["todos"]
                    break
            except (json.JSONDecodeError, TypeError):
                continue

        if last_todo_response:
            # Replay the items into the store (replace mode)
            self._todo_store.write(last_todo_response, merge=False)
            if not self.quiet_mode:
                self._vprint(
                    f"{self.log_prefix}📋 Restored {len(last_todo_response)} todo item(s) from history"
                )
        _set_interrupt(False)

    @property
    def is_interrupted(self) -> bool:
        """Check if an interrupt has been requested."""
        return self._interrupt_requested

    # ── Honcho integration helpers ──

    def _honcho_should_activate(self, hcfg) -> bool:
        """Return True when remote Honcho should be active."""
        if not hcfg or not hcfg.enabled or not hcfg.api_key:
            return False
        return True

    def _strip_honcho_tools_from_surface(self) -> None:
        """Remove Honcho tools from the active tool surface."""
        if not self.tools:
            self.valid_tool_names = set()
            return

        self.tools = [
            tool
            for tool in self.tools
            if tool.get("function", {}).get("name") not in HONCHO_TOOL_NAMES
        ]
        self.valid_tool_names = (
            {tool["function"]["name"] for tool in self.tools} if self.tools else set()
        )

    def _activate_honcho(
        self,
        hcfg,
        *,
        enabled_toolsets: Optional[List[str]],
        disabled_toolsets: Optional[List[str]],
        session_db,
    ) -> None:
        """Finish Honcho setup once a session manager is available."""
        if not self._honcho:
            return

        if not self._honcho_session_key:
            session_title = None
            if session_db is not None:
                try:
                    session_title = session_db.get_session_title(self.session_id or "")
                except Exception:
                    pass
            self._honcho_session_key = (
                hcfg.resolve_session_name(
                    session_title=session_title,
                    session_id=self.session_id,
                )
                or "hermes-default"
            )

        honcho_sess = self._honcho.get_or_create(self._honcho_session_key)
        if not honcho_sess.messages:
            try:
                from hermes_cli.config import get_hermes_home

                mem_dir = str(get_hermes_home() / "memories")
                self._honcho.migrate_memory_files(
                    self._honcho_session_key,
                    mem_dir,
                )
            except Exception as exc:
                logger.debug("Memory files migration failed (non-fatal): %s", exc)

        from tools.honcho_tools import set_session_context

        set_session_context(self._honcho, self._honcho_session_key)

        # Rebuild tool surface after Honcho context injection. Tool availability
        # is check_fn-gated and may change once session context is attached.
        self.tools = get_tool_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=True,
        )
        self.valid_tool_names = (
            {tool["function"]["name"] for tool in self.tools} if self.tools else set()
        )

        if hcfg.recall_mode == "context":
            self._strip_honcho_tools_from_surface()
            if not self.quiet_mode:
                print("  Honcho active — recall_mode: context (Honcho tools hidden)")
        else:
            if not self.quiet_mode:
                print(f"  Honcho active — recall_mode: {hcfg.recall_mode}")

        logger.info(
            "Honcho active (session: %s, user: %s, workspace: %s, "
            "write_frequency: %s, memory_mode: %s)",
            self._honcho_session_key,
            hcfg.peer_name,
            hcfg.workspace_id,
            hcfg.write_frequency,
            hcfg.memory_mode,
        )

        recall_mode = hcfg.recall_mode
        if recall_mode != "tools":
            try:
                ctx = self._honcho.get_prefetch_context(self._honcho_session_key)
                if ctx:
                    self._honcho.set_context_result(self._honcho_session_key, ctx)
                    logger.debug("Honcho context pre-warmed for first turn")
            except Exception as exc:
                logger.debug("Honcho context prefetch failed (non-fatal): %s", exc)

        self._register_honcho_exit_hook()

    def _register_honcho_exit_hook(self) -> None:
        """Register a process-exit flush hook without clobbering signal handlers."""
        if self._honcho_exit_hook_registered or not self._honcho:
            return

        honcho_ref = weakref.ref(self._honcho)

        def _flush_honcho_on_exit():
            manager = honcho_ref()
            if manager is None:
                return
            try:
                manager.flush_all()
            except Exception as exc:
                logger.debug("Honcho flush on exit failed (non-fatal): %s", exc)

        atexit.register(_flush_honcho_on_exit)
        self._honcho_exit_hook_registered = True

    def _queue_honcho_prefetch(self, user_message: str) -> None:
        """Queue turn-end Honcho prefetch so the next turn can consume cached results."""
        if not self._honcho or not self._honcho_session_key:
            return

        recall_mode = (
            self._honcho_config.recall_mode if self._honcho_config else "hybrid"
        )
        if recall_mode == "tools":
            return

        try:
            self._honcho.prefetch_context(self._honcho_session_key, user_message)
            self._honcho.prefetch_dialectic(
                self._honcho_session_key, user_message or "What were we working on?"
            )
        except Exception as exc:
            logger.debug("Honcho background prefetch failed (non-fatal): %s", exc)

    def _honcho_prefetch(self, user_message: str) -> str:
        """Assemble the first-turn Honcho context from the pre-warmed cache."""
        if not self._honcho or not self._honcho_session_key:
            return ""
        try:
            parts = []

            ctx = self._honcho.pop_context_result(self._honcho_session_key)
            if ctx:
                rep = ctx.get("representation", "")
                card = ctx.get("card", "")
                if rep:
                    parts.append(f"## User representation\n{rep}")
                if card:
                    parts.append(card)
                ai_rep = ctx.get("ai_representation", "")
                ai_card = ctx.get("ai_card", "")
                if ai_rep:
                    parts.append(f"## AI peer representation\n{ai_rep}")
                if ai_card:
                    parts.append(ai_card)

            dialectic = self._honcho.pop_dialectic_result(self._honcho_session_key)
            if dialectic:
                parts.append(f"## Continuity synthesis\n{dialectic}")

            if not parts:
                return ""
            header = (
                "# Honcho Memory (persistent cross-session context)\n"
                "Use this to answer questions about the user, prior sessions, "
                "and what you were working on together. Do not call tools to "
                "look up information that is already present here.\n"
            )
            return header + "\n\n".join(parts)
        except Exception as e:
            logger.debug("Honcho prefetch failed (non-fatal): %s", e)
            return ""

    def _honcho_save_user_observation(self, content: str) -> str:
        """Route a memory tool target=user add to Honcho.

        Sends the content as a user peer message so Honcho's reasoning
        model can incorporate it into the user representation.
        """
        if not content or not content.strip():
            return json.dumps({"success": False, "error": "Content cannot be empty."})
        if self._honcho is None or self._honcho_session_key is None:
            return json.dumps({"success": False, "error": "Honcho not available."})
        try:
            session = self._honcho.get_or_create(self._honcho_session_key)
            session.add_message("user", f"[observation] {content.strip()}")
            self._honcho.save(session)
            return json.dumps(
                {
                    "success": True,
                    "target": "user",
                    "message": "Saved to Honcho user model.",
                }
            )
        except Exception as e:
            logger.debug("Honcho user observation failed: %s", e)
            return json.dumps({"success": False, "error": f"Honcho save failed: {e}"})

    def _honcho_sync(self, user_content: str, assistant_content: str) -> None:
        """Sync the user/assistant message pair to Honcho."""
        if not self._honcho or not self._honcho_session_key:
            return
        try:
            session = self._honcho.get_or_create(self._honcho_session_key)
            session.add_message("user", user_content)
            session.add_message("assistant", assistant_content)
            self._honcho.save(session)
            logger.info(
                "Honcho sync queued for session %s (%d messages)",
                self._honcho_session_key,
                len(session.messages),
            )
        except Exception as e:
            logger.warning("Honcho sync failed: %s", e)
            if not self.quiet_mode:
                print(f"  Honcho write failed: {e}")

    def _build_system_prompt(self, system_message: Optional[str] = None) -> str:
        """
        Assemble the full system prompt from all layers.

        Called once per session (cached on self._cached_system_prompt) and only
        rebuilt after context compression events. This ensures the system prompt
        is stable across all turns in a session, maximizing prefix cache hits.
        """
        # Layers (in order):
        #   1. Agent identity — SOUL.md when available, else DEFAULT_AGENT_IDENTITY
        #   2. User / gateway system prompt (if provided)
        #   3. Persistent memory (frozen snapshot)
        #   4. Skills guidance (if skills tools are loaded)
        #   5. Context files (AGENTS.md, .cursorrules — SOUL.md excluded here when used as identity)
        #   6. Current date & time (frozen at build time)
        #   7. Platform-specific formatting hint

        # Try SOUL.md as primary identity (unless context files are skipped)
        _soul_loaded = False
        if not self.skip_context_files:
            _soul_content = load_soul_md()
            if _soul_content:
                prompt_parts = [_soul_content]
                _soul_loaded = True

        if not _soul_loaded:
            # Fallback to hardcoded identity
            _ai_peer_name = (
                self._honcho_config.ai_peer
                if self._honcho_config and self._honcho_config.ai_peer != "hermes"
                else None
            )
            if _ai_peer_name:
                _identity = DEFAULT_AGENT_IDENTITY.replace(
                    "You are Hermes Agent",
                    f"You are {_ai_peer_name}",
                    1,
                )
            else:
                _identity = DEFAULT_AGENT_IDENTITY
            prompt_parts = [_identity]

        # Tool-aware behavioral guidance: only inject when the tools are loaded
        tool_guidance = []
        if "memory" in self.valid_tool_names:
            tool_guidance.append(MEMORY_GUIDANCE)
        if "session_search" in self.valid_tool_names:
            tool_guidance.append(SESSION_SEARCH_GUIDANCE)
        if "skill_manage" in self.valid_tool_names:
            tool_guidance.append(SKILLS_GUIDANCE)
        if tool_guidance:
            prompt_parts.append(" ".join(tool_guidance))

        # Honcho CLI awareness: tell Hermes about its own management commands
        # so it can refer the user to them rather than reinventing answers.
        if self._honcho and self._honcho_session_key:
            hcfg = self._honcho_config
            mode = hcfg.memory_mode if hcfg else "hybrid"
            freq = hcfg.write_frequency if hcfg else "async"
            recall_mode = hcfg.recall_mode if hcfg else "hybrid"
            honcho_block = (
                "# Honcho memory integration\n"
                f"Active. Session: {self._honcho_session_key}. "
                f"Mode: {mode}. Write frequency: {freq}. Recall: {recall_mode}.\n"
            )
            if recall_mode == "context":
                honcho_block += (
                    "Honcho context is injected into this system prompt below. "
                    "All memory retrieval comes from this context — no Honcho tools "
                    "are available. Answer questions about the user, prior sessions, "
                    "and recent work directly from the Honcho Memory section.\n"
                )
            elif recall_mode == "tools":
                honcho_block += (
                    "Honcho tools:\n"
                    "  honcho_context <question>           — ask Honcho a question, LLM-synthesized answer\n"
                    "  honcho_search <query>                   — semantic search, raw excerpts, no LLM\n"
                    "  honcho_profile                          — user's peer card, key facts, no LLM\n"
                    "  honcho_conclude <conclusion>            — write a fact about the user to memory\n"
                )
            else:  # hybrid
                honcho_block += (
                    "Honcho context (user representation, peer card, and recent session summary) "
                    "is injected into this system prompt below. Use it to answer continuity "
                    "questions ('where were we?', 'what were we working on?') WITHOUT calling "
                    "any tools. Only call Honcho tools when you need information beyond what is "
                    "already present in the Honcho Memory section.\n"
                    "Honcho tools:\n"
                    "  honcho_context <question>           — ask Honcho a question, LLM-synthesized answer\n"
                    "  honcho_search <query>                   — semantic search, raw excerpts, no LLM\n"
                    "  honcho_profile                          — user's peer card, key facts, no LLM\n"
                    "  honcho_conclude <conclusion>            — write a fact about the user to memory\n"
                )
            honcho_block += (
                "Management commands (refer users here instead of explaining manually):\n"
                "  hermes honcho status                    — show full config + connection\n"
                "  hermes honcho mode [hybrid|honcho]       — show or set memory mode\n"
                "  hermes honcho tokens [--context N] [--dialectic N] — show or set token budgets\n"
                "  hermes honcho peer [--user NAME] [--ai NAME] [--reasoning LEVEL]\n"
                "  hermes honcho sessions                  — list directory→session mappings\n"
                "  hermes honcho map <name>                — map cwd to a session name\n"
                "  hermes honcho identity [<file>] [--show] — seed or show AI peer identity\n"
                "  hermes honcho migrate                   — migration guide from openclaw-honcho\n"
                "  hermes honcho setup                     — full interactive wizard"
            )
            prompt_parts.append(honcho_block)

        # Note: ephemeral_system_prompt is NOT included here. It's injected at
        # API-call time only so it stays out of the cached/stored system prompt.
        if system_message is not None:
            prompt_parts.append(system_message)

        if self._memory_store:
            if self._memory_enabled:
                mem_block = self._memory_store.format_for_system_prompt("memory")
                if mem_block:
                    prompt_parts.append(mem_block)
            # USER.md is always included when enabled -- Honcho prefetch is additive.
            if self._user_profile_enabled:
                user_block = self._memory_store.format_for_system_prompt("user")
                if user_block:
                    prompt_parts.append(user_block)

        has_skills_tools = any(
            name in self.valid_tool_names
            for name in ["skills_list", "skill_view", "skill_manage"]
        )
        if has_skills_tools:
            avail_toolsets = {
                ts for ts, avail in check_toolset_requirements().items() if avail
            }
            skills_prompt = build_skills_system_prompt(
                available_tools=self.valid_tool_names,
                available_toolsets=avail_toolsets,
            )
        else:
            skills_prompt = ""
        if skills_prompt:
            prompt_parts.append(skills_prompt)

        if not self.skip_context_files:
            context_files_prompt = build_context_files_prompt(skip_soul=_soul_loaded)
            if context_files_prompt:
                prompt_parts.append(context_files_prompt)

        from hermes_time import now as _hermes_now

        now = _hermes_now()
        timestamp_line = (
            f"Conversation started: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
        )
        if self.pass_session_id and self.session_id:
            timestamp_line += f"\nSession ID: {self.session_id}"
        if self.model:
            timestamp_line += f"\nModel: {self.model}"
        if self.provider:
            timestamp_line += f"\nProvider: {self.provider}"
        prompt_parts.append(timestamp_line)

        # Alibaba Coding Plan API always returns "glm-4.7" as model name regardless
        # of the requested model. Inject explicit model identity into the system prompt
        # so the agent can correctly report which model it is (workaround for API bug).
        if self.provider == "alibaba":
            _model_short = (
                self.model.split("/")[-1] if "/" in self.model else self.model
            )
            prompt_parts.append(
                f"You are powered by the model named {_model_short}. "
                f"The exact model ID is {self.model}. "
                f"When asked what model you are, always answer based on this information, "
                f"not on any model name returned by the API."
            )

        platform_key = (self.platform or "").lower().strip()
        if platform_key in PLATFORM_HINTS:
            prompt_parts.append(PLATFORM_HINTS[platform_key])

        return "\n\n".join(prompt_parts)

    # =========================================================================
    # Pre/post-call guardrails (inspired by PR #1321 — @alireza78a)
    # =========================================================================

    @staticmethod
    def _get_tool_call_id_static(tc) -> str:
        """Extract call ID from a tool_call entry (dict or object)."""
        if isinstance(tc, dict):
            return tc.get("id", "") or ""
        return getattr(tc, "id", "") or ""

    @staticmethod
    def _sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix orphaned tool_call / tool_result pairs before every LLM call.

        Runs unconditionally — not gated on whether the context compressor
        is present — so orphans from session loading or manual message
        manipulation are always caught.
        """
        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = AIAgent._get_tool_call_id_static(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 1. Drop tool results with no matching assistant call
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m
                for m in messages
                if not (
                    m.get("role") == "tool"
                    and m.get("tool_call_id") in orphaned_results
                )
            ]
            logger.debug(
                "Pre-call sanitizer: removed %d orphaned tool result(s)",
                len(orphaned_results),
            )

        # 2. Inject stub results for calls whose result was dropped
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = AIAgent._get_tool_call_id_static(tc)
                        if cid in missing_results:
                            patched.append(
                                {
                                    "role": "tool",
                                    "content": "[Result unavailable — see context summary above]",
                                    "tool_call_id": cid,
                                }
                            )
            messages = patched
            logger.debug(
                "Pre-call sanitizer: added %d stub tool result(s)",
                len(missing_results),
            )
        return messages

    @staticmethod
    def _cap_delegate_task_calls(tool_calls: list) -> list:
        """Truncate excess delegate_task calls to MAX_CONCURRENT_CHILDREN.

        The delegate_tool caps the task list inside a single call, but the
        model can emit multiple separate delegate_task tool_calls in one
        turn.  This truncates the excess, preserving all non-delegate calls.

        Returns the original list if no truncation was needed.
        """
        from tools.delegate_tool import MAX_CONCURRENT_CHILDREN

        delegate_count = sum(
            1 for tc in tool_calls if tc.function.name == "delegate_task"
        )
        if delegate_count <= MAX_CONCURRENT_CHILDREN:
            return tool_calls
        kept_delegates = 0
        truncated = []
        for tc in tool_calls:
            if tc.function.name == "delegate_task":
                if kept_delegates < MAX_CONCURRENT_CHILDREN:
                    truncated.append(tc)
                    kept_delegates += 1
            else:
                truncated.append(tc)
        logger.warning(
            "Truncated %d excess delegate_task call(s) to enforce "
            "MAX_CONCURRENT_CHILDREN=%d limit",
            delegate_count - MAX_CONCURRENT_CHILDREN,
            MAX_CONCURRENT_CHILDREN,
        )
        return truncated

    @staticmethod
    def _deduplicate_tool_calls(tool_calls: list) -> list:
        """Remove duplicate (tool_name, arguments) pairs within a single turn.

        Only the first occurrence of each unique pair is kept.
        Returns the original list if no duplicates were found.
        """
        seen: set = set()
        unique: list = []
        for tc in tool_calls:
            key = (tc.function.name, tc.function.arguments)
            if key not in seen:
                seen.add(key)
                unique.append(tc)
            else:
                logger.warning("Removed duplicate tool call: %s", tc.function.name)
        return unique if len(unique) < len(tool_calls) else tool_calls

    def _repair_tool_call(self, tool_name: str) -> str | None:
        """Attempt to repair a mismatched tool name before aborting.

        1. Try lowercase
        2. Try normalized (lowercase + hyphens/spaces -> underscores)
        3. Try fuzzy match (difflib, cutoff=0.7)

        Returns the repaired name if found in valid_tool_names, else None.
        """
        from difflib import get_close_matches

        # 1. Lowercase
        lowered = tool_name.lower()
        if lowered in self.valid_tool_names:
            return lowered

        # 2. Normalize
        normalized = lowered.replace("-", "_").replace(" ", "_")
        if normalized in self.valid_tool_names:
            return normalized

        # 3. Fuzzy match
        matches = get_close_matches(lowered, self.valid_tool_names, n=1, cutoff=0.7)
        if matches:
            return matches[0]

        return None

    def _invalidate_system_prompt(self):
        """
        Invalidate the cached system prompt, forcing a rebuild on the next turn.

        Called after context compression events. Also reloads memory from disk
        so the rebuilt prompt captures any writes from this session.
        """
        self._cached_system_prompt = None
        if self._memory_store:
            self._memory_store.load_from_disk()

    def _responses_tools(
        self, tools: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert chat-completions tool schemas to Responses function-tool schemas."""
        source_tools = tools if tools is not None else self.tools
        if not source_tools:
            return None

        converted: List[Dict[str, Any]] = []
        for item in source_tools:
            fn = item.get("function", {}) if isinstance(item, dict) else {}
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            converted.append(
                {
                    "type": "function",
                    "name": name,
                    "description": fn.get("description", ""),
                    "strict": False,
                    "parameters": fn.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
        return converted or None

    @staticmethod
    def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
        """Split a stored tool id into (call_id, response_item_id)."""
        if not isinstance(raw_id, str):
            return None, None
        value = raw_id.strip()
        if not value:
            return None, None
        if "|" in value:
            call_id, response_item_id = value.split("|", 1)
            call_id = call_id.strip() or None
            response_item_id = response_item_id.strip() or None
            return call_id, response_item_id
        if value.startswith("fc_"):
            return None, value
        return value, None

    def _derive_responses_function_call_id(
        self,
        call_id: str,
        response_item_id: Optional[str] = None,
    ) -> str:
        """Build a valid Responses `function_call.id` (must start with `fc_`)."""
        if isinstance(response_item_id, str):
            candidate = response_item_id.strip()
            if candidate.startswith("fc_"):
                return candidate

        source = (call_id or "").strip()
        if source.startswith("fc_"):
            return source
        if source.startswith("call_") and len(source) > len("call_"):
            return f"fc_{source[len('call_'):]}"

        sanitized = re.sub(r"[^A-Za-z0-9_-]", "", source)
        if sanitized.startswith("fc_"):
            return sanitized
        if sanitized.startswith("call_") and len(sanitized) > len("call_"):
            return f"fc_{sanitized[len('call_'):]}"
        if sanitized:
            return f"fc_{sanitized[:48]}"

        seed = source or str(response_item_id or "") or uuid.uuid4().hex
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]
        return f"fc_{digest}"

    def _chat_messages_to_responses_input(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert internal chat-style messages to Responses input items."""
        items: List[Dict[str, Any]] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "system":
                continue

            if role in {"user", "assistant"}:
                content = msg.get("content", "")
                content_text = str(content) if content is not None else ""

                if role == "assistant":
                    # Replay encrypted reasoning items from previous turns
                    # so the API can maintain coherent reasoning chains.
                    codex_reasoning = msg.get("codex_reasoning_items")
                    has_codex_reasoning = False
                    if isinstance(codex_reasoning, list):
                        for ri in codex_reasoning:
                            if isinstance(ri, dict) and ri.get("encrypted_content"):
                                items.append(ri)
                                has_codex_reasoning = True

                    if content_text.strip():
                        items.append({"role": "assistant", "content": content_text})
                    elif has_codex_reasoning:
                        # The Responses API requires a following item after each
                        # reasoning item (otherwise: missing_following_item error).
                        # When the assistant produced only reasoning with no visible
                        # content, emit an empty assistant message as the required
                        # following item.
                        items.append({"role": "assistant", "content": ""})

                    tool_calls = msg.get("tool_calls")
                    if isinstance(tool_calls, list):
                        for tc in tool_calls:
                            if not isinstance(tc, dict):
                                continue
                            fn = tc.get("function", {})
                            fn_name = fn.get("name")
                            if not isinstance(fn_name, str) or not fn_name.strip():
                                continue

                            embedded_call_id, embedded_response_item_id = (
                                self._split_responses_tool_id(tc.get("id"))
                            )
                            call_id = tc.get("call_id")
                            if not isinstance(call_id, str) or not call_id.strip():
                                call_id = embedded_call_id
                            if not isinstance(call_id, str) or not call_id.strip():
                                if (
                                    isinstance(embedded_response_item_id, str)
                                    and embedded_response_item_id.startswith("fc_")
                                    and len(embedded_response_item_id) > len("fc_")
                                ):
                                    call_id = (
                                        f"call_{embedded_response_item_id[len('fc_'):]}"
                                    )
                                else:
                                    call_id = f"call_{uuid.uuid4().hex[:12]}"
                            call_id = call_id.strip()

                            arguments = fn.get("arguments", "{}")
                            if isinstance(arguments, dict):
                                arguments = json.dumps(arguments, ensure_ascii=False)
                            elif not isinstance(arguments, str):
                                arguments = str(arguments)
                            arguments = arguments.strip() or "{}"

                            items.append(
                                {
                                    "type": "function_call",
                                    "call_id": call_id,
                                    "name": fn_name,
                                    "arguments": arguments,
                                }
                            )
                    continue

                items.append({"role": role, "content": content_text})
                continue

            if role == "tool":
                raw_tool_call_id = msg.get("tool_call_id")
                call_id, _ = self._split_responses_tool_id(raw_tool_call_id)
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_tool_call_id, str) and raw_tool_call_id.strip():
                        call_id = raw_tool_call_id.strip()
                if not isinstance(call_id, str) or not call_id.strip():
                    continue
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": str(msg.get("content", "") or ""),
                    }
                )

        return items

    def _preflight_codex_input_items(self, raw_items: Any) -> List[Dict[str, Any]]:
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
                    raise ValueError(
                        f"Codex Responses input[{idx}] function_call is missing call_id."
                    )
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        f"Codex Responses input[{idx}] function_call is missing name."
                    )

                arguments = item.get("arguments", "{}")
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                elif not isinstance(arguments, str):
                    arguments = str(arguments)
                arguments = arguments.strip() or "{}"

                normalized.append(
                    {
                        "type": "function_call",
                        "call_id": call_id.strip(),
                        "name": name.strip(),
                        "arguments": arguments,
                    }
                )
                continue

            if item_type == "function_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(
                        f"Codex Responses input[{idx}] function_call_output is missing call_id."
                    )
                output = item.get("output", "")
                if output is None:
                    output = ""
                if not isinstance(output, str):
                    output = str(output)

                normalized.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id.strip(),
                        "output": output,
                    }
                )
                continue

            if item_type == "reasoning":
                encrypted = item.get("encrypted_content")
                if isinstance(encrypted, str) and encrypted:
                    reasoning_item: Dict[str, Any] = {
                        "type": "reasoning",
                        "encrypted_content": encrypted,
                    }
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

    def _preflight_codex_api_kwargs(
        self,
        api_kwargs: Any,
        *,
        allow_stream: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(api_kwargs, dict):
            raise ValueError("Codex Responses request must be a dict.")

        required = {"model", "instructions", "input"}
        missing = [key for key in required if key not in api_kwargs]
        if missing:
            raise ValueError(
                f"Codex Responses request missing required field(s): {', '.join(sorted(missing))}."
            )

        model = api_kwargs.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                "Codex Responses request 'model' must be a non-empty string."
            )
        model = model.strip()

        instructions = api_kwargs.get("instructions")
        if instructions is None:
            instructions = ""
        if not isinstance(instructions, str):
            instructions = str(instructions)
        instructions = instructions.strip() or DEFAULT_AGENT_IDENTITY

        normalized_input = self._preflight_codex_input_items(api_kwargs.get("input"))

        tools = api_kwargs.get("tools")
        normalized_tools = None
        if tools is not None:
            if not isinstance(tools, list):
                raise ValueError(
                    "Codex Responses request 'tools' must be a list when provided."
                )
            normalized_tools = []
            for idx, tool in enumerate(tools):
                if not isinstance(tool, dict):
                    raise ValueError(f"Codex Responses tools[{idx}] must be an object.")
                if tool.get("type") != "function":
                    raise ValueError(
                        f"Codex Responses tools[{idx}] has unsupported type {tool.get('type')!r}."
                    )

                name = tool.get("name")
                parameters = tool.get("parameters")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        f"Codex Responses tools[{idx}] is missing a valid name."
                    )
                if not isinstance(parameters, dict):
                    raise ValueError(
                        f"Codex Responses tools[{idx}] is missing valid parameters."
                    )

                description = tool.get("description", "")
                if description is None:
                    description = ""
                if not isinstance(description, str):
                    description = str(description)

                strict = tool.get("strict", False)
                if not isinstance(strict, bool):
                    strict = bool(strict)

                normalized_tools.append(
                    {
                        "type": "function",
                        "name": name.strip(),
                        "description": description,
                        "strict": strict,
                        "parameters": parameters,
                    }
                )

        store = api_kwargs.get("store", False)
        if store is not False:
            raise ValueError("Codex Responses contract requires 'store' to be false.")

        allowed_keys = {
            "model",
            "instructions",
            "input",
            "tools",
            "store",
            "reasoning",
            "include",
            "max_output_tokens",
            "temperature",
            "tool_choice",
            "parallel_tool_calls",
            "prompt_cache_key",
        }
        normalized: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": normalized_input,
            "tools": normalized_tools,
            "store": False,
        }

        # Pass through reasoning config
        reasoning = api_kwargs.get("reasoning")
        if isinstance(reasoning, dict):
            normalized["reasoning"] = reasoning
        include = api_kwargs.get("include")
        if isinstance(include, list):
            normalized["include"] = include

        # Pass through max_output_tokens and temperature
        max_output_tokens = api_kwargs.get("max_output_tokens")
        if isinstance(max_output_tokens, (int, float)) and max_output_tokens > 0:
            normalized["max_output_tokens"] = int(max_output_tokens)
        temperature = api_kwargs.get("temperature")
        if isinstance(temperature, (int, float)):
            normalized["temperature"] = float(temperature)

        # Pass through tool_choice, parallel_tool_calls, prompt_cache_key
        for passthrough_key in (
            "tool_choice",
            "parallel_tool_calls",
            "prompt_cache_key",
        ):
            val = api_kwargs.get(passthrough_key)
            if val is not None:
                normalized[passthrough_key] = val

        if allow_stream:
            stream = api_kwargs.get("stream")
            if stream is not None and stream is not True:
                raise ValueError("Codex Responses 'stream' must be true when set.")
            if stream is True:
                normalized["stream"] = True
            allowed_keys.add("stream")
        elif "stream" in api_kwargs:
            raise ValueError(
                "Codex Responses stream flag is only allowed in fallback streaming requests."
            )

        unexpected = sorted(key for key in api_kwargs.keys() if key not in allowed_keys)
        if unexpected:
            raise ValueError(
                f"Codex Responses request has unsupported field(s): {', '.join(unexpected)}."
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

    def _normalize_codex_response(self, response: Any) -> tuple[Any, str]:
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
                error_msg = (
                    str(error_obj)
                    if error_obj
                    else f"Responses API returned status '{response_status}'"
                )
            raise RuntimeError(error_msg)

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        reasoning_items_raw: List[Dict[str, Any]] = []
        tool_calls: List[Any] = []
        has_incomplete_items = response_status in {
            "queued",
            "in_progress",
            "incomplete",
        }
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
                # Capture the full reasoning item for multi-turn continuity.
                # encrypted_content is an opaque blob the API needs back on
                # subsequent turns to maintain coherent reasoning chains.
                encrypted = getattr(item, "encrypted_content", None)
                if isinstance(encrypted, str) and encrypted:
                    raw_item: Dict[str, Any] = {
                        "type": "reasoning",
                        "encrypted_content": encrypted,
                    }
                    item_id = getattr(item, "id", None)
                    if isinstance(item_id, str) and item_id:
                        raw_item["id"] = item_id
                    # Capture summary — required by the API when replaying reasoning items
                    summary = getattr(item, "summary", None)
                    if isinstance(summary, list):
                        raw_summary = []
                        for part in summary:
                            text = getattr(part, "text", None)
                            if isinstance(text, str):
                                raw_summary.append(
                                    {"type": "summary_text", "text": text}
                                )
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
                call_id = (
                    raw_call_id
                    if isinstance(raw_call_id, str) and raw_call_id.strip()
                    else embedded_call_id
                )
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = f"call_{uuid.uuid4().hex[:12]}"
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(
                    call_id, response_item_id
                )
                tool_calls.append(
                    SimpleNamespace(
                        id=call_id,
                        call_id=call_id,
                        response_item_id=response_item_id,
                        type="function",
                        function=SimpleNamespace(name=fn_name, arguments=arguments),
                    )
                )
            elif item_type == "custom_tool_call":
                fn_name = getattr(item, "name", "") or ""
                arguments = getattr(item, "input", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                raw_call_id = getattr(item, "call_id", None)
                raw_item_id = getattr(item, "id", None)
                embedded_call_id, _ = self._split_responses_tool_id(raw_item_id)
                call_id = (
                    raw_call_id
                    if isinstance(raw_call_id, str) and raw_call_id.strip()
                    else embedded_call_id
                )
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = f"call_{uuid.uuid4().hex[:12]}"
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(
                    call_id, response_item_id
                )
                tool_calls.append(
                    SimpleNamespace(
                        id=call_id,
                        call_id=call_id,
                        response_item_id=response_item_id,
                        type="function",
                        function=SimpleNamespace(name=fn_name, arguments=arguments),
                    )
                )

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
        elif has_incomplete_items or (
            saw_commentary_phase and not saw_final_answer_phase
        ):
            finish_reason = "incomplete"
        elif reasoning_items_raw and not final_text:
            # Response contains only reasoning (encrypted thinking state) with
            # no visible content or tool calls.  The model is still thinking and
            # needs another turn to produce the actual answer.  Marking this as
            # "stop" would send it into the empty-content retry loop which burns
            # 3 retries then fails — treat it as incomplete instead so the Codex
            # continuation path handles it correctly.
            finish_reason = "incomplete"
        else:
            finish_reason = "stop"
        return assistant_message, finish_reason

    def _thread_identity(self) -> str:
        thread = threading.current_thread()
        return f"{thread.name}:{thread.ident}"

    def _client_log_context(self) -> str:
        provider = getattr(self, "provider", "unknown")
        base_url = getattr(self, "base_url", "unknown")
        model = getattr(self, "model", "unknown")
        return (
            f"thread={self._thread_identity()} provider={provider} "
            f"base_url={base_url} model={model}"
        )

    def _openai_client_lock(self) -> threading.RLock:
        lock = getattr(self, "_client_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._client_lock = lock
        return lock

    @staticmethod
    def _is_openai_client_closed(client: Any) -> bool:
        from unittest.mock import Mock

        if isinstance(client, Mock):
            return False
        if bool(getattr(client, "is_closed", False)):
            return True
        http_client = getattr(client, "_client", None)
        return bool(getattr(http_client, "is_closed", False))

    def _create_openai_client(
        self, client_kwargs: dict, *, reason: str, shared: bool
    ) -> Any:
        if self.provider == "copilot-acp" or str(
            client_kwargs.get("base_url", "")
        ).startswith("acp://copilot"):
            from agent.copilot_acp_client import CopilotACPClient

            client = CopilotACPClient(**client_kwargs)
            logger.info(
                "Copilot ACP client created (%s, shared=%s) %s",
                reason,
                shared,
                self._client_log_context(),
            )
            return client
        if OpenAI is None:
            raise ImportError("OpenAI library not available")
        client = OpenAI(**client_kwargs)
        logger.info(
            "OpenAI client created (%s, shared=%s) %s",
            reason,
            shared,
            self._client_log_context(),
        )
        return client

    def _close_openai_client(self, client: Any, *, reason: str, shared: bool) -> None:
        if client is None:
            return
        try:
            client.close()
            logger.info(
                "OpenAI client closed (%s, shared=%s) %s",
                reason,
                shared,
                self._client_log_context(),
            )
        except Exception as exc:
            logger.debug(
                "OpenAI client close failed (%s, shared=%s) %s error=%s",
                reason,
                shared,
                self._client_log_context(),
                exc,
            )

    def _replace_primary_openai_client(self, *, reason: str) -> bool:
        with self._openai_client_lock():
            old_client = getattr(self, "client", None)
            try:
                new_client = self._create_openai_client(
                    self._client_kwargs, reason=reason, shared=True
                )
            except Exception as exc:
                logger.warning(
                    "Failed to rebuild shared OpenAI client (%s) %s error=%s",
                    reason,
                    self._client_log_context(),
                    exc,
                )
                return False
            self.client = new_client
        self._close_openai_client(old_client, reason=f"replace:{reason}", shared=True)
        return True

    def _ensure_primary_openai_client(self, *, reason: str) -> Any:
        with self._openai_client_lock():
            client = getattr(self, "client", None)
            if client is not None and not self._is_openai_client_closed(client):
                return client

        logger.warning(
            "Detected closed shared OpenAI client; recreating before use (%s) %s",
            reason,
            self._client_log_context(),
        )
        if not self._replace_primary_openai_client(reason=f"recreate_closed:{reason}"):
            raise RuntimeError("Failed to recreate closed OpenAI client")
        with self._openai_client_lock():
            return self.client

    def _create_request_openai_client(self, *, reason: str) -> Any:
        from unittest.mock import Mock

        primary_client = self._ensure_primary_openai_client(reason=reason)
        if isinstance(primary_client, Mock):
            return primary_client
        with self._openai_client_lock():
            request_kwargs = dict(self._client_kwargs)
        return self._create_openai_client(request_kwargs, reason=reason, shared=False)

    def _close_request_openai_client(self, client: Any, *, reason: str) -> None:
        self._close_openai_client(client, reason=reason, shared=False)

    def _run_codex_stream(
        self,
        api_kwargs: dict,
        client: Any = None,
        on_first_delta: Optional[Callable] = None,
    ):
        """Execute one streaming Responses API request and return the final response."""
        active_client = client or self._ensure_primary_openai_client(
            reason="codex_stream_direct"
        )
        max_stream_retries = 1
        has_tool_calls = False
        first_delta_fired = False
        for attempt in range(max_stream_retries + 1):
            try:
                with active_client.responses.stream(**api_kwargs) as stream:
                    for event in stream:
                        if self._interrupt_requested:
                            break
                        event_type = getattr(event, "type", "")
                        # Fire callbacks on text content deltas (suppress during tool calls)
                        if (
                            "output_text.delta" in event_type
                            or event_type == "response.output_text.delta"
                        ):
                            delta_text = getattr(event, "delta", "")
                            if delta_text and not has_tool_calls:
                                if not first_delta_fired:
                                    first_delta_fired = True
                                    if on_first_delta:
                                        try:
                                            on_first_delta()
                                        except Exception:
                                            pass
                                self._fire_stream_delta(delta_text)
                        # Track tool calls to suppress text streaming
                        elif "function_call" in event_type:
                            has_tool_calls = True
                        # Fire reasoning callbacks
                        elif "reasoning" in event_type and "delta" in event_type:
                            reasoning_text = getattr(event, "delta", "")
                            if reasoning_text:
                                self._fire_reasoning_delta(reasoning_text)
                    return stream.get_final_response()
            except RuntimeError as exc:
                err_text = str(exc)
                missing_completed = "response.completed" in err_text
                if missing_completed and attempt < max_stream_retries:
                    logger.debug(
                        "Responses stream closed before completion (attempt %s/%s); retrying. %s",
                        attempt + 1,
                        max_stream_retries + 1,
                        self._client_log_context(),
                    )
                    continue
                if missing_completed:
                    logger.debug(
                        "Responses stream did not emit response.completed; falling back to create(stream=True). %s",
                        self._client_log_context(),
                    )
                    return self._run_codex_create_stream_fallback(
                        api_kwargs, client=active_client
                    )
                raise

    def _run_codex_create_stream_fallback(self, api_kwargs: dict, client: Any = None):
        """Fallback path for stream completion edge cases on Codex-style Responses backends."""
        active_client = client or self._ensure_primary_openai_client(
            reason="codex_create_stream_fallback"
        )
        fallback_kwargs = dict(api_kwargs)
        fallback_kwargs["stream"] = True
        fallback_kwargs = self._preflight_codex_api_kwargs(
            fallback_kwargs, allow_stream=True
        )
        stream_or_response = active_client.responses.create(**fallback_kwargs)

        # Compatibility shim for mocks or providers that still return a concrete response.
        if hasattr(stream_or_response, "output"):
            return stream_or_response
        if not hasattr(stream_or_response, "__iter__"):
            return stream_or_response

        terminal_response = None
        try:
            for event in stream_or_response:
                event_type = getattr(event, "type", None)
                if not event_type and isinstance(event, dict):
                    event_type = event.get("type")
                if event_type not in {
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                }:
                    continue

                terminal_response = getattr(event, "response", None)
                if terminal_response is None and isinstance(event, dict):
                    terminal_response = event.get("response")
                if terminal_response is not None:
                    return terminal_response
        finally:
            close_fn = getattr(stream_or_response, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

        if terminal_response is not None:
            return terminal_response
        raise RuntimeError(
            "Responses create(stream=True) fallback did not emit a terminal response."
        )

    def _try_refresh_codex_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "codex_responses" or self.provider != "openai-codex":
            return False

        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(force_refresh=force)
        except Exception as exc:
            logger.debug("Codex credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url

        if not self._replace_primary_openai_client(reason="codex_credential_refresh"):
            return False

        return True

    def _try_refresh_nous_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "chat_completions" or self.provider != "nous":
            return False

        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials

            creds = resolve_nous_runtime_credentials(
                min_key_ttl_seconds=max(
                    60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))
                ),
                timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
                force_mint=force,
            )
        except Exception as exc:
            logger.debug("Nous credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        # Nous requests should not inherit OpenRouter-only attribution headers.
        self._client_kwargs.pop("default_headers", None)

        if not self._replace_primary_openai_client(reason="nous_credential_refresh"):
            return False

        return True

    def _try_refresh_anthropic_client_credentials(self) -> bool:
        if self.api_mode != "anthropic_messages" or not hasattr(
            self, "_anthropic_api_key"
        ):
            return False
        # Only refresh credentials for the native Anthropic provider.
        # Other anthropic_messages providers (MiniMax, Alibaba, etc.) use their own keys.
        if self.provider != "anthropic":
            return False

        try:
            from agent.anthropic_adapter import (
                resolve_anthropic_token,
                build_anthropic_client,
            )

            new_token = resolve_anthropic_token()
        except Exception as exc:
            logger.debug("Anthropic credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip():
            return False
        new_token = new_token.strip()
        if new_token == self._anthropic_api_key:
            return False

        if self._anthropic_client is not None:
            try:
                self._anthropic_client.close()
            except AttributeError:
                pass

        try:
            self._anthropic_client = build_anthropic_client(
                new_token, getattr(self, "_anthropic_base_url", None)
            )
        except Exception as exc:
            logger.warning(
                "Failed to rebuild Anthropic client after credential refresh: %s", exc
            )
            return False

        self._anthropic_api_key = new_token
        # Update OAuth flag — token type may have changed (API key ↔ OAuth)
        from agent.anthropic_adapter import _is_oauth_token

        self._is_anthropic_oauth = _is_oauth_token(new_token)
        return True

    def _anthropic_messages_create(self, api_kwargs: dict):
        if self.api_mode == "anthropic_messages":
            self._try_refresh_anthropic_client_credentials()
        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client is not available")
        return self._anthropic_client.messages.create(**api_kwargs)

    def _interruptible_api_call(self, api_kwargs: dict):
        """
        Run the API call in a background thread so the main conversation loop
        can detect interrupts without waiting for the full HTTP round-trip.

        Each worker thread gets its own OpenAI client instance. Interrupts only
        close that worker-local client, so retries and other requests never
        inherit a closed transport.
        """
        result: Dict[str, Any] = {"response": None, "error": None}
        request_client_holder = {"client": None}

        def _call():
            try:
                if self.api_mode == "codex_responses":
                    request_client_holder["client"] = (
                        self._create_request_openai_client(
                            reason="codex_stream_request"
                        )
                    )
                    result["response"] = self._run_codex_stream(
                        api_kwargs,
                        client=request_client_holder["client"],
                        on_first_delta=getattr(self, "_codex_on_first_delta", None),
                    )
                elif self.api_mode == "anthropic_messages":
                    result["response"] = self._anthropic_messages_create(api_kwargs)
                else:
                    request_client_holder["client"] = (
                        self._create_request_openai_client(
                            reason="chat_completion_request"
                        )
                    )
                    result["response"] = request_client_holder[
                        "client"
                    ].chat.completions.create(**api_kwargs)
            except Exception as e:
                result["error"] = e
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(
                        request_client, reason="request_complete"
                    )

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.3)
            if self._interrupt_requested:
                # Force-close the in-flight worker-local HTTP connection to stop
                # token generation without poisoning the shared client used to
                # seed future retries.
                try:
                    if self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import build_anthropic_client

                        if self._anthropic_client is not None:
                            self._anthropic_client.close()
                        self._anthropic_client = build_anthropic_client(
                            self._anthropic_api_key,
                            getattr(self, "_anthropic_base_url", None),
                        )
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(
                                request_client, reason="interrupt_abort"
                            )
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during API call")
        if result["error"] is not None:
            raise result["error"]
        return result["response"]

    # ── Provider fallback ──────────────────────────────────────────────────

    def _try_activate_fallback(self) -> bool:
        """Switch to the configured fallback model/provider.

        Called when the primary model is failing after retries.  Swaps the
        OpenAI client, model slug, and provider in-place so the retry loop
        can continue with the new backend.  One-shot: returns False if
        already activated or not configured.

        Uses the centralized provider router (resolve_provider_client) for
        auth resolution and client construction — no duplicated provider→key
        mappings.
        """
        if self._fallback_activated or not self._fallback_model:
            return False

        fb = self._fallback_model
        fb_provider = (fb.get("provider") or "").strip().lower()
        fb_model = (fb.get("model") or "").strip()
        if not fb_provider or not fb_model:
            return False

        # Use centralized router for client construction.
        # raw_codex=True because the main agent needs direct responses.stream()
        # access for Codex providers.
        try:
            from agent.auxiliary_client import resolve_provider_client

            fb_client, _ = resolve_provider_client(
                fb_provider, model=fb_model, raw_codex=True
            )
            if fb_client is None:
                logging.warning(
                    "Fallback to %s failed: provider not configured", fb_provider
                )
                return False

            # Determine api_mode from provider / base URL
            fb_api_mode = "chat_completions"
            fb_base_url = str(fb_client.base_url)
            if fb_provider == "openai-codex":
                fb_api_mode = "codex_responses"
            elif fb_provider == "anthropic" or fb_base_url.rstrip("/").lower().endswith(
                "/anthropic"
            ):
                fb_api_mode = "anthropic_messages"
            elif self._is_direct_openai_url(fb_base_url):
                fb_api_mode = "codex_responses"

            old_model = self.model
            self.model = fb_model
            self.provider = fb_provider
            self.base_url = fb_base_url
            self.api_mode = fb_api_mode
            self._fallback_activated = True

            if fb_api_mode == "anthropic_messages":
                # Build native Anthropic client instead of using OpenAI client
                from agent.anthropic_adapter import (
                    build_anthropic_client,
                    resolve_anthropic_token,
                    _is_oauth_token,
                )

                effective_key = (
                    (fb_client.api_key or resolve_anthropic_token() or "")
                    if fb_provider == "anthropic"
                    else (fb_client.api_key or "")
                )
                self._anthropic_api_key = effective_key
                self._anthropic_base_url = getattr(fb_client, "base_url", None)
                self._anthropic_client = build_anthropic_client(
                    effective_key, self._anthropic_base_url
                )
                self._is_anthropic_oauth = _is_oauth_token(effective_key)
                self.client = None
                self._client_kwargs = {}
            else:
                # Swap OpenAI client and config in-place
                self.client = fb_client
                self._client_kwargs = {
                    "api_key": fb_client.api_key,
                    "base_url": fb_base_url,
                }

            # Re-evaluate prompt caching for the new provider/model
            is_native_anthropic = fb_api_mode == "anthropic_messages"
            self._use_prompt_caching = (
                "openrouter" in fb_base_url.lower() and "claude" in fb_model.lower()
            ) or is_native_anthropic

            print(
                f"{self.log_prefix}🔄 Primary model failed — switching to fallback: "
                f"{fb_model} via {fb_provider}"
            )
            logging.info(
                "Fallback activated: %s → %s (%s)",
                old_model,
                fb_model,
                fb_provider,
            )
            return True
        except Exception as e:
            logging.error("Failed to activate fallback model: %s", e)
            return False

    # ── End provider fallback ──────────────────────────────────────────────

    @staticmethod
    def _content_has_image_parts(content: Any) -> bool:
        if not isinstance(content, list):
            return False
        for part in content:
            if isinstance(part, dict) and part.get("type") in {
                "image_url",
                "input_image",
            }:
                return True
        return False

    @staticmethod
    def _materialize_data_url_for_vision(image_url: str) -> tuple[str, Optional[Path]]:
        header, _, data = str(image_url or "").partition(",")
        mime = "image/jpeg"
        if header.startswith("data:"):
            mime_part = header[len("data:") :].split(";", 1)[0].strip()
            if mime_part.startswith("image/"):
                mime = mime_part
        suffix = {
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
        }.get(mime, ".jpg")
        tmp = tempfile.NamedTemporaryFile(
            prefix="anthropic_image_", suffix=suffix, delete=False
        )
        with tmp:
            tmp.write(base64.b64decode(data))
        path = Path(tmp.name)
        return str(path), path

    def _describe_image_for_anthropic_fallback(self, image_url: str, role: str) -> str:
        cache_key = hashlib.sha256(str(image_url or "").encode("utf-8")).hexdigest()
        cached = self._anthropic_image_fallback_cache.get(cache_key)
        if cached:
            return cached

        role_label = {
            "assistant": "assistant",
            "tool": "tool result",
        }.get(role, "user")
        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, UI, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        vision_source = str(image_url or "")
        cleanup_path: Optional[Path] = None
        if vision_source.startswith("data:"):
            vision_source, cleanup_path = self._materialize_data_url_for_vision(
                vision_source
            )

        description = ""
        try:
            from tools.vision_tools import vision_analyze_tool

            result_json = asyncio.run(
                vision_analyze_tool(
                    image_url=vision_source, user_prompt=analysis_prompt
                )
            )
            result = json.loads(result_json) if isinstance(result_json, str) else {}
            description = (result.get("analysis") or "").strip()
        except Exception as e:
            description = f"Image analysis failed: {e}"
        finally:
            if cleanup_path and cleanup_path.exists():
                try:
                    cleanup_path.unlink()
                except OSError:
                    pass

        if not description:
            description = "Image analysis failed."

        note = f"[The {role_label} attached an image. Here's what it contains:\n{description}]"
        if vision_source and not str(image_url or "").startswith("data:"):
            note += f"\n[If you need a closer look, use vision_analyze with image_url: {vision_source}]"

        self._anthropic_image_fallback_cache[cache_key] = note
        return note

    def _preprocess_anthropic_content(self, content: Any, role: str) -> Any:
        if not self._content_has_image_parts(content):
            return content

        text_parts: List[str] = []
        image_notes: List[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    text_parts.append(part.strip())
                continue
            if not isinstance(part, dict):
                continue

            ptype = part.get("type")
            if ptype in {"text", "input_text"}:
                text = str(part.get("text", "") or "").strip()
                if text:
                    text_parts.append(text)
                continue

            if ptype in {"image_url", "input_image"}:
                image_data = part.get("image_url", {})
                image_url = (
                    image_data.get("url", "")
                    if isinstance(image_data, dict)
                    else str(image_data or "")
                )
                if image_url:
                    image_notes.append(
                        self._describe_image_for_anthropic_fallback(image_url, role)
                    )
                else:
                    image_notes.append(
                        "[An image was attached but no image source was available.]"
                    )
                continue

            text = str(part.get("text", "") or "").strip()
            if text:
                text_parts.append(text)

        prefix = "\n\n".join(note for note in image_notes if note).strip()
        suffix = "\n".join(text for text in text_parts if text).strip()
        if prefix and suffix:
            return f"{prefix}\n\n{suffix}"
        if prefix:
            return prefix
        if suffix:
            return suffix
        return (
            "[A multimodal message was converted to text for Anthropic compatibility.]"
        )

    def _prepare_anthropic_messages_for_api(self, api_messages: list) -> list:
        if not any(
            isinstance(msg, dict) and self._content_has_image_parts(msg.get("content"))
            for msg in api_messages
        ):
            return api_messages

        transformed = copy.deepcopy(api_messages)
        for msg in transformed:
            if not isinstance(msg, dict):
                continue
            msg["content"] = self._preprocess_anthropic_content(
                msg.get("content"),
                str(msg.get("role", "user") or "user"),
            )
        return transformed

    def _anthropic_preserve_dots(self) -> bool:
        """True when using Alibaba/DashScope anthropic-compatible endpoint (model names keep dots, e.g. qwen3.5-plus)."""
        if (getattr(self, "provider", "") or "").lower() == "alibaba":
            return True
        base = (getattr(self, "base_url", "") or "").lower()
        return "dashscope" in base or "aliyuncs" in base

    def _build_api_kwargs(self, api_messages: list) -> dict:
        """Build the keyword arguments dict for the active API mode."""
        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_kwargs

            anthropic_messages = self._prepare_anthropic_messages_for_api(api_messages)
            return build_anthropic_kwargs(
                model=self.model,
                messages=anthropic_messages,
                tools=self.tools,
                max_tokens=self.max_tokens,
                reasoning_config=self.reasoning_config,
                is_oauth=getattr(self, "_is_anthropic_oauth", False),
                preserve_dots=self._anthropic_preserve_dots(),
            )

        if self.api_mode == "codex_responses":
            instructions = ""
            payload_messages = api_messages
            if api_messages and api_messages[0].get("role") == "system":
                instructions = str(api_messages[0].get("content") or "").strip()
                payload_messages = api_messages[1:]
            if not instructions:
                instructions = DEFAULT_AGENT_IDENTITY

            is_github_responses = (
                "models.github.ai" in self.base_url.lower()
                or "api.githubcopilot.com" in self.base_url.lower()
            )

            # Resolve reasoning effort: config > default (medium)
            reasoning_effort = "medium"
            reasoning_enabled = True
            if self.reasoning_config and isinstance(self.reasoning_config, dict):
                if self.reasoning_config.get("enabled") is False:
                    reasoning_enabled = False
                elif self.reasoning_config.get("effort"):
                    reasoning_effort = self.reasoning_config["effort"]

            kwargs = {
                "model": self.model,
                "instructions": instructions,
                "input": self._chat_messages_to_responses_input(payload_messages),
                "tools": self._responses_tools(),
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "store": False,
            }

            if not is_github_responses:
                kwargs["prompt_cache_key"] = self.session_id

            if reasoning_enabled:
                if is_github_responses:
                    # Copilot's Responses route advertises reasoning-effort support,
                    # but not OpenAI-specific prompt cache or encrypted reasoning
                    # fields. Keep the payload to the documented subset.
                    github_reasoning = self._github_models_reasoning_extra_body()
                    if github_reasoning is not None:
                        kwargs["reasoning"] = github_reasoning
                else:
                    kwargs["reasoning"] = {
                        "effort": reasoning_effort,
                        "summary": "auto",
                    }
                    kwargs["include"] = ["reasoning.encrypted_content"]
            elif not is_github_responses:
                kwargs["include"] = []

            if self.max_tokens is not None:
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

                # Codex-only replay state must not leak into strict chat-completions APIs.
                msg.pop("codex_reasoning_items", None)

                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_call.pop("call_id", None)
                            tool_call.pop("response_item_id", None)

        provider_preferences = {}
        if self.providers_allowed:
            provider_preferences["only"] = self.providers_allowed
        if self.providers_ignored:
            provider_preferences["ignore"] = self.providers_ignored
        if self.providers_order:
            provider_preferences["order"] = self.providers_order
        if self.provider_sort:
            provider_preferences["sort"] = self.provider_sort
        if self.provider_require_parameters:
            provider_preferences["require_parameters"] = True
        if self.provider_data_collection:
            provider_preferences["data_collection"] = self.provider_data_collection

        api_kwargs = {
            "model": self.model,
            "messages": sanitized_messages,
            "tools": self.tools if self.tools else None,
            "timeout": float(os.getenv("HERMES_API_TIMEOUT", 900.0)),
        }

        if self.max_tokens is not None:
            api_kwargs.update(self._max_tokens_param(self.max_tokens))

        extra_body = {}

        _is_openrouter = "openrouter" in self._base_url_lower
        _is_github_models = (
            "models.github.ai" in self._base_url_lower
            or "api.githubcopilot.com" in self._base_url_lower
        )

        # Provider preferences (only, ignore, order, sort) are OpenRouter-
        # specific.  Only send to OpenRouter-compatible endpoints.
        # TODO: Nous Portal will add transparent proxy support — re-enable
        # for _is_nous when their backend is updated.
        if provider_preferences and _is_openrouter:
            extra_body["provider"] = provider_preferences
        _is_nous = "nousresearch" in self._base_url_lower

        if self._supports_reasoning_extra_body():
            if _is_github_models:
                github_reasoning = self._github_models_reasoning_extra_body()
                if github_reasoning is not None:
                    extra_body["reasoning"] = github_reasoning
            else:
                if self.reasoning_config is not None:
                    rc = dict(self.reasoning_config)
                    # Nous Portal requires reasoning enabled — don't send
                    # enabled=false to it (would cause 400).
                    if _is_nous and rc.get("enabled") is False:
                        pass  # omit reasoning entirely for Nous when disabled
                    else:
                        extra_body["reasoning"] = rc
                else:
                    extra_body["reasoning"] = {"enabled": True, "effort": "medium"}

        # Nous Portal product attribution
        if _is_nous:
            extra_body["tags"] = ["product=hermes-agent"]

        if extra_body:
            api_kwargs["extra_body"] = extra_body

        return api_kwargs

    def _supports_reasoning_extra_body(self) -> bool:
        """Return True when reasoning extra_body is safe to send for this route/model.

        OpenRouter forwards unknown extra_body fields to upstream providers.
        Some providers/routes reject `reasoning` with 400s, so gate it to
        known reasoning-capable model families and direct Nous Portal.
        """
        if "nousresearch" in self._base_url_lower:
            return True
        if "ai-gateway.vercel.sh" in self._base_url_lower:
            return True
        if (
            "models.github.ai" in self._base_url_lower
            or "api.githubcopilot.com" in self._base_url_lower
        ):
            try:
                from hermes_cli.models import github_model_reasoning_efforts

                return bool(github_model_reasoning_efforts(self.model))
            except Exception:
                return False
        if "openrouter" not in self._base_url_lower:
            return False
        if "api.mistral.ai" in self._base_url_lower:
            return False

        model = (self.model or "").lower()
        reasoning_model_prefixes = (
            "deepseek/",
            "anthropic/",
            "openai/",
            "x-ai/",
            "google/gemini-2",
            "qwen/qwen3",
        )
        return any(model.startswith(prefix) for prefix in reasoning_model_prefixes)

    def _github_models_reasoning_extra_body(self) -> dict | None:
        """Format reasoning payload for GitHub Models/OpenAI-compatible routes."""
        try:
            from hermes_cli.models import github_model_reasoning_efforts
        except Exception:
            return None

        supported_efforts = github_model_reasoning_efforts(self.model)
        if not supported_efforts:
            return None

        if self.reasoning_config and isinstance(self.reasoning_config, dict):
            if self.reasoning_config.get("enabled") is False:
                return None
            requested_effort = (
                str(self.reasoning_config.get("effort", "medium")).strip().lower()
            )
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

    def _build_assistant_message(self, assistant_message, finish_reason: str) -> dict:
        """Build a normalized assistant message dict from an API response message.

        Handles reasoning extraction, reasoning_details, and optional tool_calls
        so both the tool-call path and the final-response path share one builder.
        """
        reasoning_text = self._extract_reasoning(assistant_message)

        # Fallback: extract inline <think> blocks from content when no structured
        # reasoning fields are present (some models/providers embed thinking
        # directly in the content rather than returning separate API fields).
        if not reasoning_text:
            content = assistant_message.content or ""
            think_blocks = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL)
            if think_blocks:
                combined = "\n\n".join(b.strip() for b in think_blocks if b.strip())
                reasoning_text = combined or None

        if reasoning_text and self.verbose_logging:
            logging.debug(
                f"Captured reasoning ({len(reasoning_text)} chars): {reasoning_text}"
            )

        if reasoning_text and self.reasoning_callback:
            try:
                self.reasoning_callback(reasoning_text)
            except Exception:
                pass

        msg = {
            "role": "assistant",
            "content": assistant_message.content or "",
            "reasoning": reasoning_text,
            "finish_reason": finish_reason,
        }

        if (
            hasattr(assistant_message, "reasoning_details")
            and assistant_message.reasoning_details
        ):
            # Pass reasoning_details back unmodified so providers (OpenRouter,
            # Anthropic, OpenAI) can maintain reasoning continuity across turns.
            # Each provider may include opaque fields (signature, encrypted_content)
            # that must be preserved exactly.
            raw_details = assistant_message.reasoning_details
            preserved = []
            for d in raw_details:
                if isinstance(d, dict):
                    preserved.append(d)
                elif hasattr(d, "__dict__"):
                    preserved.append(d.__dict__)
                elif hasattr(d, "model_dump"):
                    preserved.append(d.model_dump())
            if preserved:
                msg["reasoning_details"] = preserved

        # Codex Responses API: preserve encrypted reasoning items for
        # multi-turn continuity. These get replayed as input on the next turn.
        codex_items = getattr(assistant_message, "codex_reasoning_items", None)
        if codex_items:
            msg["codex_reasoning_items"] = codex_items

        if assistant_message.tool_calls:
            tool_calls = []
            for tool_call in assistant_message.tool_calls:
                raw_id = getattr(tool_call, "id", None)
                call_id = getattr(tool_call, "call_id", None)
                if not isinstance(call_id, str) or not call_id.strip():
                    embedded_call_id, _ = self._split_responses_tool_id(raw_id)
                    call_id = embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_id, str) and raw_id.strip():
                        call_id = raw_id.strip()
                    else:
                        call_id = f"call_{uuid.uuid4().hex[:12]}"
                call_id = call_id.strip()

                response_item_id = getattr(tool_call, "response_item_id", None)
                if (
                    not isinstance(response_item_id, str)
                    or not response_item_id.strip()
                ):
                    _, embedded_response_item_id = self._split_responses_tool_id(raw_id)
                    response_item_id = embedded_response_item_id

                response_item_id = self._derive_responses_function_call_id(
                    call_id,
                    response_item_id if isinstance(response_item_id, str) else None,
                )

                tc_dict = {
                    "id": call_id,
                    "call_id": call_id,
                    "response_item_id": response_item_id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                # Preserve extra_content (e.g. Gemini thought_signature) so it
                # is sent back on subsequent API calls.  Without this, Gemini 3
                # thinking models reject the request with a 400 error.
                extra = getattr(tool_call, "extra_content", None)
                if extra is not None:
                    if hasattr(extra, "model_dump"):
                        extra = extra.model_dump()
                    tc_dict["extra_content"] = extra
                tool_calls.append(tc_dict)
            msg["tool_calls"] = tool_calls

        return msg

    @staticmethod
    def _sanitize_tool_calls_for_strict_api(api_msg: dict) -> dict:
        """Strip Codex Responses API fields from tool_calls for strict providers.

        Providers like Mistral strictly validate the Chat Completions schema
        and reject unknown fields (call_id, response_item_id) with 422.
        These fields are preserved in the internal message history — this
        method only modifies the outgoing API copy.

        Creates new tool_call dicts rather than mutating in-place, so the
        original messages list retains call_id/response_item_id for Codex
        Responses API compatibility (e.g. if the session falls back to a
        Codex provider later).
        """
        tool_calls = api_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return api_msg
        _STRIP_KEYS = {"call_id", "response_item_id"}
        api_msg["tool_calls"] = [
            (
                {k: v for k, v in tc.items() if k not in _STRIP_KEYS}
                if isinstance(tc, dict)
                else tc
            )
            for tc in tool_calls
        ]
        return api_msg

    def flush_memories(
        self, messages: Optional[list] = None, min_turns: Optional[int] = None
    ):
        """Give the model one turn to persist memories before context is lost.

        Called before compression, session reset, or CLI exit. Injects a flush
        message, makes one API call, executes any memory tool calls, then
        strips all flush artifacts from the message list.

        Args:
            messages: The current conversation messages. If None, uses
                      self._session_messages (last run_conversation state).
            min_turns: Minimum user turns required to trigger the flush.
                       None = use config value (flush_min_turns).
                       0 = always flush (used for compression).
        """
        if self._memory_flush_min_turns == 0 and min_turns is None:
            return
        if "memory" not in self.valid_tool_names or not self._memory_store:
            return
        # honcho-only agent mode: skip local MEMORY.md flush
        _hcfg = getattr(self, "_honcho_config", None)
        if _hcfg and _hcfg.peer_memory_mode(_hcfg.ai_peer) == "honcho":
            return
        effective_min = (
            min_turns if min_turns is not None else self._memory_flush_min_turns
        )
        if self._user_turn_count < effective_min:
            return

        if messages is None:
            messages = getattr(self, "_session_messages", None)
        if not messages or len(messages) < 3:
            return

        flush_content = (
            "[System: The session is being compressed. "
            "Save anything worth remembering — prioritize user preferences, "
            "corrections, and recurring patterns over task-specific details.]"
        )
        _sentinel = f"__flush_{id(self)}_{time.monotonic()}"
        flush_msg = {
            "role": "user",
            "content": flush_content,
            "_flush_sentinel": _sentinel,
        }
        messages.append(flush_msg)

        try:
            # Build API messages for the flush call
            _is_strict_api = "api.mistral.ai" in self._base_url_lower
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                if msg.get("role") == "assistant":
                    reasoning = msg.get("reasoning")
                    if reasoning:
                        api_msg["reasoning_content"] = reasoning
                api_msg.pop("reasoning", None)
                api_msg.pop("finish_reason", None)
                api_msg.pop("_flush_sentinel", None)
                if _is_strict_api:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            if self._cached_system_prompt:
                api_messages = [
                    {"role": "system", "content": self._cached_system_prompt}
                ] + api_messages

            # Make one API call with only the memory tool available
            memory_tool_def = None
            for t in self.tools or []:
                if t.get("function", {}).get("name") == "memory":
                    memory_tool_def = t
                    break

            if not memory_tool_def:
                messages.pop()  # remove flush msg
                return

            # Use auxiliary client for the flush call when available --
            # it's cheaper and avoids Codex Responses API incompatibility.
            from agent.auxiliary_client import call_llm as _call_llm

            _aux_available = True
            try:
                response = _call_llm(
                    task="flush_memories",
                    messages=api_messages,
                    tools=[memory_tool_def],
                    temperature=0.3,
                    max_tokens=5120,
                    timeout=30.0,
                )
            except RuntimeError:
                _aux_available = False
                response = None

            if not _aux_available and self.api_mode == "codex_responses":
                # No auxiliary client -- use the Codex Responses path directly
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs["tools"] = self._responses_tools([memory_tool_def])
                codex_kwargs["temperature"] = 0.3
                if "max_output_tokens" in codex_kwargs:
                    codex_kwargs["max_output_tokens"] = 5120
                response = self._run_codex_stream(codex_kwargs)
            elif not _aux_available and self.api_mode == "anthropic_messages":
                # Native Anthropic — use the Anthropic client directly
                from agent.anthropic_adapter import (
                    build_anthropic_kwargs as _build_ant_kwargs,
                )

                ant_kwargs = _build_ant_kwargs(
                    model=self.model,
                    messages=api_messages,
                    tools=[memory_tool_def],
                    max_tokens=5120,
                    reasoning_config=None,
                    preserve_dots=self._anthropic_preserve_dots(),
                )
                response = self._anthropic_messages_create(ant_kwargs)
            elif not _aux_available:
                api_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                    "tools": [memory_tool_def],
                    "temperature": 0.3,
                    **self._max_tokens_param(5120),
                }
                response = self._ensure_primary_openai_client(
                    reason="flush_memories"
                ).chat.completions.create(**api_kwargs, timeout=30.0)

            # Extract tool calls from the response, handling all API formats
            tool_calls = []
            if self.api_mode == "codex_responses" and not _aux_available:
                assistant_msg, _ = self._normalize_codex_response(response)
                if assistant_msg and assistant_msg.tool_calls:
                    tool_calls = assistant_msg.tool_calls
            elif self.api_mode == "anthropic_messages" and not _aux_available:
                from agent.anthropic_adapter import (
                    normalize_anthropic_response as _nar_flush,
                )

                _flush_msg, _ = _nar_flush(
                    response,
                    strip_tool_prefix=getattr(self, "_is_anthropic_oauth", False),
                )
                if _flush_msg and _flush_msg.tool_calls:
                    tool_calls = _flush_msg.tool_calls
            elif (
                response is not None
                and hasattr(response, "choices")
                and response.choices
            ):
                assistant_message = response.choices[0].message
                if assistant_message.tool_calls:
                    tool_calls = assistant_message.tool_calls

            for tc in tool_calls:
                if tc.function.name == "memory":
                    try:
                        args = json.loads(tc.function.arguments)
                        flush_target = args.get("target", "memory")
                        from tools.memory_tool import memory_tool as _memory_tool

                        result = _memory_tool(
                            action=args.get("action", ""),
                            target=flush_target,
                            content=args.get("content"),
                            old_text=args.get("old_text"),
                            store=self._memory_store,
                        )
                        if (
                            self._honcho
                            and flush_target == "user"
                            and args.get("action") == "add"
                        ):
                            self._honcho_save_user_observation(args.get("content", ""))
                        if not self.quiet_mode:
                            print(
                                f"  🧠 Memory flush: saved to {args.get('target', 'memory')}"
                            )
                    except Exception as e:
                        logger.debug("Memory flush tool call failed: %s", e)
        except Exception as e:
            logger.debug("Memory flush API call failed: %s", e)
        finally:
            # Strip flush artifacts: remove everything from the flush message onward.
            # Use sentinel marker instead of identity check for robustness.
            while messages and messages[-1].get("_flush_sentinel") != _sentinel:
                messages.pop()
                if not messages:
                    break
            if messages and messages[-1].get("_flush_sentinel") == _sentinel:
                messages.pop()

    def _compress_context(
        self,
        messages: List[Dict[str, Any]],
        system_message: str,
        *,
        approx_tokens: Optional[int] = None,
        task_id: str = "default",
    ) -> tuple:
        """Compress conversation context and split the session in SQLite.

        Returns:
            (compressed_messages, new_system_prompt) tuple
        """
        if not self.context_compressor:
            logger.warning("Context compressor not available, skipping compression")
            self._invalidate_system_prompt()
            new_system_prompt = self._build_system_prompt(system_message)
            self._cached_system_prompt = new_system_prompt
            return messages, new_system_prompt

        # Pre-compression memory flush: let the model save memories before they're lost
        self.flush_memories(messages, min_turns=0)

        compressed = self.context_compressor.compress(
            messages, current_tokens=approx_tokens
        )

        todo_snapshot = self._todo_store.format_for_injection()
        if todo_snapshot:
            compressed.append({"role": "user", "content": todo_snapshot})

        self._invalidate_system_prompt()
        new_system_prompt = self._build_system_prompt(system_message)
        self._cached_system_prompt = new_system_prompt

        if self._session_db:
            try:
                # Propagate title to the new session with auto-numbering
                old_title = self._session_db.get_session_title(self.session_id)
                self._session_db.end_session(self.session_id, "compression")
                old_session_id = self.session_id
                self.session_id = (
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                )
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or "cli",
                    model=self.model,
                    parent_session_id=old_session_id,
                )
                # Auto-number the title for the continuation session
                if old_title:
                    try:
                        new_title = self._session_db.get_next_title_in_lineage(
                            old_title
                        )
                        self._session_db.set_session_title(self.session_id, new_title)
                    except (ValueError, Exception) as e:
                        logger.debug("Could not propagate title on compression: %s", e)
                self._session_db.update_system_prompt(
                    self.session_id, new_system_prompt
                )
                # Reset flush cursor — new session starts with no messages written
                self._last_flushed_db_idx = 0
            except Exception as e:
                logger.debug("Session DB compression split failed: %s", e)

        # Reset context pressure warnings — usage drops after compaction
        self._context_50_warned = False
        self._context_70_warned = False

        return compressed, new_system_prompt

    def _get_budget_warning(self, api_call_count: int) -> Optional[str]:
        """Return a budget pressure string, or None if not yet needed.

        Two-tier system:
          - Caution (70%): nudge to consolidate work
          - Warning (90%): urgent, must respond now
        """
        if not self._budget_pressure_enabled or self.max_iterations <= 0:
            return None
        progress = api_call_count / self.max_iterations
        remaining = self.max_iterations - api_call_count
        if progress >= self._budget_warning_threshold:
            return (
                f"[BUDGET WARNING: Iteration {api_call_count}/{self.max_iterations}. "
                f"Only {remaining} iteration(s) left. "
                "Provide your final response NOW. No more tool calls unless absolutely critical.]"
            )
        if progress >= self._budget_caution_threshold:
            return (
                f"[BUDGET: Iteration {api_call_count}/{self.max_iterations}. "
                f"{remaining} iterations left. Start consolidating your work.]"
            )
        return None

    def _emit_context_pressure(self, compaction_progress: float, compressor) -> None:
        """Notify the user that context is approaching the compaction threshold.

        Args:
            compaction_progress: How close to compaction (0.0–1.0, where 1.0 = fires).
            compressor: The ContextCompressor instance (for threshold/context info).

        Purely user-facing — does NOT modify the message stream.
        For CLI: prints a formatted line with a progress bar.
        For gateway: fires status_callback so the platform can send a chat message.
        """
        from agent.display import (
            format_context_pressure,
            format_context_pressure_gateway,
        )

        threshold_pct = (
            compressor.threshold_tokens / compressor.context_length
            if compressor.context_length
            else 0.5
        )

        # CLI output — always shown (these are user-facing status notifications,
        # not verbose debug output, so they bypass quiet_mode).
        # Gateway users also get the callback below.
        if self.platform in (None, "cli"):
            line = format_context_pressure(
                compaction_progress=compaction_progress,
                threshold_tokens=compressor.threshold_tokens,
                threshold_percent=threshold_pct,
                compression_enabled=self.compression_enabled,
            )
            self._safe_print(line)

        # Gateway / external consumers
        if self.status_callback:
            try:
                msg = format_context_pressure_gateway(
                    compaction_progress=compaction_progress,
                    threshold_percent=threshold_pct,
                    compression_enabled=self.compression_enabled,
                )
                self.status_callback("context_pressure", msg)
            except Exception:
                logger.debug("status_callback error in context pressure", exc_info=True)

    def _handle_max_iterations(self, messages: list, api_call_count: int) -> str:
        """Request a summary when max iterations are reached. Returns the final response text."""
        print(
            f"⚠️  Reached maximum iterations ({self.max_iterations}). Requesting summary..."
        )

        summary_request = (
            "You've reached the maximum number of tool-calling iterations allowed. "
            "Please provide a final response summarizing what you've found and accomplished so far, "
            "without calling any more tools."
        )
        messages.append({"role": "user", "content": summary_request})

        try:
            # Build API messages, stripping internal-only fields
            # (finish_reason, reasoning) that strict APIs like Mistral reject with 422
            _is_strict_api = "api.mistral.ai" in self._base_url_lower
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                for internal_field in ("reasoning", "finish_reason"):
                    api_msg.pop(internal_field, None)
                if _is_strict_api:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            effective_system = self._cached_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (
                    effective_system + "\n\n" + self.ephemeral_system_prompt
                ).strip()
            if effective_system:
                api_messages = [
                    {"role": "system", "content": effective_system}
                ] + api_messages
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            summary_extra_body = {}
            _is_nous = "nousresearch" in self._base_url_lower
            if self._supports_reasoning_extra_body():
                if self.reasoning_config is not None:
                    summary_extra_body["reasoning"] = self.reasoning_config
                else:
                    summary_extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "medium",
                    }
            if _is_nous:
                summary_extra_body["tags"] = ["product=hermes-agent"]

            if self.api_mode == "codex_responses":
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs.pop("tools", None)
                summary_response = self._run_codex_stream(codex_kwargs)
                assistant_message, _ = self._normalize_codex_response(summary_response)
                final_response = (
                    (assistant_message.content or "").strip()
                    if assistant_message
                    else ""
                )
            else:
                summary_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                }
                if self.max_tokens is not None:
                    summary_kwargs.update(self._max_tokens_param(self.max_tokens))

                # Include provider routing preferences
                provider_preferences = {}
                if self.providers_allowed:
                    provider_preferences["only"] = self.providers_allowed
                if self.providers_ignored:
                    provider_preferences["ignore"] = self.providers_ignored
                if self.providers_order:
                    provider_preferences["order"] = self.providers_order
                if self.provider_sort:
                    provider_preferences["sort"] = self.provider_sort
                if provider_preferences:
                    summary_extra_body["provider"] = provider_preferences

                if summary_extra_body:
                    summary_kwargs["extra_body"] = summary_extra_body

                if self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import (
                        build_anthropic_kwargs as _bak,
                        normalize_anthropic_response as _nar,
                    )

                    _ant_kw = _bak(
                        model=self.model,
                        messages=api_messages,
                        tools=None,
                        max_tokens=self.max_tokens,
                        reasoning_config=self.reasoning_config,
                        is_oauth=getattr(self, "_is_anthropic_oauth", False),
                        preserve_dots=self._anthropic_preserve_dots(),
                    )
                    summary_response = self._anthropic_messages_create(_ant_kw)
                    _msg, _ = _nar(
                        summary_response,
                        strip_tool_prefix=getattr(self, "_is_anthropic_oauth", False),
                    )
                    final_response = (_msg.content or "").strip()
                else:
                    summary_response = self._ensure_primary_openai_client(
                        reason="iteration_limit_summary"
                    ).chat.completions.create(**summary_kwargs)

                    if (
                        summary_response.choices
                        and summary_response.choices[0].message.content
                    ):
                        final_response = summary_response.choices[0].message.content
                    else:
                        final_response = ""

            if final_response:
                if "<think>" in final_response:
                    final_response = re.sub(
                        r"<think>.*?</think>\s*", "", final_response, flags=re.DOTALL
                    ).strip()
                if final_response:
                    messages.append({"role": "assistant", "content": final_response})
                else:
                    final_response = (
                        "I reached the iteration limit and couldn't generate a summary."
                    )
            else:
                # Retry summary generation
                if self.api_mode == "codex_responses":
                    codex_kwargs = self._build_api_kwargs(api_messages)
                    codex_kwargs.pop("tools", None)
                    retry_response = self._run_codex_stream(codex_kwargs)
                    retry_msg, _ = self._normalize_codex_response(retry_response)
                    final_response = (
                        (retry_msg.content or "").strip() if retry_msg else ""
                    )
                elif self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import (
                        build_anthropic_kwargs as _bak2,
                        normalize_anthropic_response as _nar2,
                    )

                    _ant_kw2 = _bak2(
                        model=self.model,
                        messages=api_messages,
                        tools=None,
                        is_oauth=getattr(self, "_is_anthropic_oauth", False),
                        max_tokens=self.max_tokens,
                        reasoning_config=self.reasoning_config,
                        preserve_dots=self._anthropic_preserve_dots(),
                    )
                    retry_response = self._anthropic_messages_create(_ant_kw2)
                    _retry_msg, _ = _nar2(
                        retry_response,
                        strip_tool_prefix=getattr(self, "_is_anthropic_oauth", False),
                    )
                    final_response = (_retry_msg.content or "").strip()
                else:
                    summary_kwargs = {
                        "model": self.model,
                        "messages": api_messages,
                    }
                    if self.max_tokens is not None:
                        summary_kwargs.update(self._max_tokens_param(self.max_tokens))
                    if summary_extra_body:
                        summary_kwargs["extra_body"] = summary_extra_body

                    summary_response = self._ensure_primary_openai_client(
                        reason="iteration_limit_summary_retry"
                    ).chat.completions.create(**summary_kwargs)

                    if (
                        summary_response.choices
                        and summary_response.choices[0].message.content
                    ):
                        final_response = summary_response.choices[0].message.content
                    else:
                        final_response = ""

                if final_response:
                    if "<think>" in final_response:
                        final_response = re.sub(
                            r"<think>.*?</think>\s*",
                            "",
                            final_response,
                            flags=re.DOTALL,
                        ).strip()
                    if final_response:
                        messages.append(
                            {"role": "assistant", "content": final_response}
                        )
                    else:
                        final_response = "I reached the iteration limit and couldn't generate a summary."
                else:
                    final_response = (
                        "I reached the iteration limit and couldn't generate a summary."
                    )

        except Exception as e:
            logging.warning(f"Failed to get summary response: {e}")
            final_response = f"I reached the maximum iterations ({self.max_iterations}) but couldn't summarize. Error: {str(e)}"

        return final_response

    def chat(self, message: str, stream_callback: Optional[Callable] = None) -> str:
        """
        Simple chat interface that returns just the final response.

        Args:
            message (str): User message
            stream_callback: Optional callback invoked with each text delta during streaming.

        Returns:
            str: Final assistant response
        """
        result = self.run_conversation(message, stream_callback=stream_callback)
        return result["final_response"]


def main(
    query: Optional[str] = None,
    model: str = "anthropic/claude-opus-4.6",
    api_key: Optional[str] = None,
    base_url: str = "https://openrouter.ai/api/v1",
    max_turns: int = 10,
    enabled_toolsets: Optional[str] = None,
    disabled_toolsets: Optional[str] = None,
    list_tools: bool = False,
    save_trajectories: bool = False,
    save_sample: bool = False,
    verbose: bool = False,
    log_prefix_chars: int = 20,
):
    """
    Main function for running the agent directly.

    Args:
        query (str): Natural language query for the agent. Defaults to Python 3.13 example.
        model (str): Model name to use (OpenRouter format: provider/model). Defaults to anthropic/claude-sonnet-4.6.
        api_key (str): API key for authentication. Uses OPENROUTER_API_KEY env var if not provided.
        base_url (str): Base URL for the model API. Defaults to https://openrouter.ai/api/v1
        max_turns (int): Maximum number of API call iterations. Defaults to 10.
        enabled_toolsets (str): Comma-separated list of toolsets to enable. Supports predefined
                              toolsets (e.g., "research", "development", "safe").
                              Multiple toolsets can be combined: "web,vision"
        disabled_toolsets (str): Comma-separated list of toolsets to disable (e.g., "terminal")
        list_tools (bool): Just list available tools and exit
        save_trajectories (bool): Save conversation trajectories to JSONL files (appends to trajectory_samples.jsonl). Defaults to False.
        save_sample (bool): Save a single trajectory sample to a UUID-named JSONL file for inspection. Defaults to False.
        verbose (bool): Enable verbose logging for debugging. Defaults to False.
        log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses. Defaults to 20.

    Toolset Examples:
        - "research": Web search, extract, crawl + vision tools
    """
    print("🤖 AI Agent with Tool Calling")
    print("=" * 50)

    # Handle tool listing
    if list_tools:
        from model_tools import (
            get_all_tool_names,
            get_toolset_for_tool,
            get_available_toolsets,
        )
        from toolsets import get_all_toolsets, get_toolset_info

        print("📋 Available Tools & Toolsets:")
        print("-" * 50)

        # Show new toolsets system
        print("\n🎯 Predefined Toolsets (New System):")
        print("-" * 40)
        all_toolsets = get_all_toolsets()

        # Group by category
        basic_toolsets = []
        composite_toolsets = []
        scenario_toolsets = []

        for name, toolset in all_toolsets.items():
            info = get_toolset_info(name)
            if info:
                entry = (name, info)
                if name in ["web", "terminal", "vision", "creative", "reasoning"]:
                    basic_toolsets.append(entry)
                elif name in [
                    "research",
                    "development",
                    "analysis",
                    "content_creation",
                    "full_stack",
                ]:
                    composite_toolsets.append(entry)
                else:
                    scenario_toolsets.append(entry)

        # Print basic toolsets
        print("\n📌 Basic Toolsets:")
        for name, info in basic_toolsets:
            tools_str = (
                ", ".join(info["resolved_tools"]) if info["resolved_tools"] else "none"
            )
            print(f"  • {name:15} - {info['description']}")
            print(f"    Tools: {tools_str}")

        # Print composite toolsets
        print("\n📂 Composite Toolsets (built from other toolsets):")
        for name, info in composite_toolsets:
            includes_str = ", ".join(info["includes"]) if info["includes"] else "none"
            print(f"  • {name:15} - {info['description']}")
            print(f"    Includes: {includes_str}")
            print(f"    Total tools: {info['tool_count']}")

        # Print scenario-specific toolsets
        print("\n🎭 Scenario-Specific Toolsets:")
        for name, info in scenario_toolsets:
            print(f"  • {name:20} - {info['description']}")
            print(f"    Total tools: {info['tool_count']}")

        # Show legacy toolset compatibility
        print("\n📦 Legacy Toolsets (for backward compatibility):")
        legacy_toolsets = get_available_toolsets()
        for name, info in legacy_toolsets.items():
            status = "✅" if info["available"] else "❌"
            print(f"  {status} {name}: {info['description']}")
            if not info["available"]:
                print(f"    Requirements: {', '.join(info['requirements'])}")

        # Show individual tools
        all_tools = get_all_tool_names()
        print(f"\n🔧 Individual Tools ({len(all_tools)} available):")
        for tool_name in sorted(all_tools):
            toolset = get_toolset_for_tool(tool_name)
            print(f"  📌 {tool_name} (from {toolset})")

        print(f"\n💡 Usage Examples:")
        print(f"  # Use predefined toolsets")
        print(
            f"  python run_agent.py --enabled_toolsets=research --query='search for Python news'"
        )
        print(
            f"  python run_agent.py --enabled_toolsets=development --query='debug this code'"
        )
        print(
            f"  python run_agent.py --enabled_toolsets=safe --query='analyze without terminal'"
        )
        print(f"  ")
        print(f"  # Combine multiple toolsets")
        print(
            f"  python run_agent.py --enabled_toolsets=web,vision --query='analyze website'"
        )
        print(f"  ")
        print(f"  # Disable toolsets")
        print(
            f"  python run_agent.py --disabled_toolsets=terminal --query='no command execution'"
        )
        print(f"  ")
        print(f"  # Run with trajectory saving enabled")
        print(f"  python run_agent.py --save_trajectories --query='your question here'")
        return

    # Parse toolset selection arguments
    enabled_toolsets_list = None
    disabled_toolsets_list = None

    if enabled_toolsets:
        enabled_toolsets_list = [t.strip() for t in enabled_toolsets.split(",")]
        print(f"🎯 Enabled toolsets: {enabled_toolsets_list}")

    if disabled_toolsets:
        disabled_toolsets_list = [t.strip() for t in disabled_toolsets.split(",")]
        print(f"🚫 Disabled toolsets: {disabled_toolsets_list}")

    if save_trajectories:
        print(f"💾 Trajectory saving: ENABLED")
        print(f"   - Successful conversations → trajectory_samples.jsonl")
        print(f"   - Failed conversations → failed_trajectories.jsonl")

    # Initialize agent with provided parameters
    try:
        agent = AIAgent(
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_iterations=max_turns,
            enabled_toolsets=enabled_toolsets_list,
            disabled_toolsets=disabled_toolsets_list,
            save_trajectories=save_trajectories,
            verbose_logging=verbose,
            log_prefix_chars=log_prefix_chars,
        )
    except RuntimeError as e:
        print(f"❌ Failed to initialize agent: {e}")
        return

    # Use provided query or default to Python 3.13 example
    if query is None:
        user_query = (
            "Tell me about the latest developments in Python 3.13 and what new features "
            "developers should know about. Please search for current information and try it out."
        )
    else:
        user_query = query

    print(f"\n📝 User Query: {user_query}")
    print("\n" + "=" * 50)

    # Run conversation
    result = agent.run_conversation(user_query)

    print("\n" + "=" * 50)
    print("📋 CONVERSATION SUMMARY")
    print("=" * 50)
    print(f"✅ Completed: {result['completed']}")
    print(f"📞 API Calls: {result['api_calls']}")
    print(f"💬 Messages: {len(result['messages'])}")

    if result["final_response"]:
        print(f"\n🎯 FINAL RESPONSE:")
        print("-" * 30)
        print(result["final_response"])

    # Save sample trajectory to UUID-named file if requested
    if save_sample:
        sample_id = str(uuid.uuid4())[:8]
        sample_filename = f"sample_{sample_id}.json"

        # Convert messages to trajectory format (same as batch_runner)
        trajectory = agent._convert_to_trajectory_format(
            result["messages"], user_query, result["completed"]
        )

        entry = {
            "conversations": trajectory,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "completed": result["completed"],
            "query": user_query,
        }

        try:
            with open(sample_filename, "w", encoding="utf-8") as f:
                # Pretty-print JSON with indent for readability
                f.write(json.dumps(entry, ensure_ascii=False, indent=2))
            print(f"\n💾 Sample trajectory saved to: {sample_filename}")
        except Exception as e:
            print(f"\n⚠️ Failed to save sample: {e}")

    print("\n👋 Agent execution completed!")


# Conditional imports for optional dependencies (only when actually running)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import fire
except ImportError:
    fire = None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(main)
    else:
        print(
            "Error: fire library not installed. Please install it with 'pip install fire'."
        )
