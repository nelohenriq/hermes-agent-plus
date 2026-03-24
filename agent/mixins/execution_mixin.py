"""Execution mixin for HermesAgent - handles tool call execution."""

import concurrent.futures
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

# ==============================================================================
# CONSTANTS FOR PARALLEL TOOL EXECUTION
# ==============================================================================

_NEVER_PARALLEL_TOOLS = frozenset({"clarify"})

# Read-only tools with no shared mutable session state.
_PARALLEL_SAFE_TOOLS = frozenset({
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
})

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
_REDIRECT_OVERWRITE = re.compile(r'[^>]>[^>]|^>[^>]')


# ==============================================================================
# HELPER FUNCTIONS FOR TOOL EXECUTION
# ==============================================================================

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
            if any(_paths_overlap(scoped_path, existing) for existing in reserved_paths):
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
        # Empty paths should not reach here (guarded upstream), but be safe.
        return bool(left_parts) == bool(right_parts) and bool(left_parts)
    common_len = min(len(left_parts), len(right_parts))
    return left_parts[:common_len] == right_parts[:common_len]


# ==============================================================================
# EXECUTION MIXIN CLASS
# ==============================================================================

class ExecutionMixin:
    """Mixin providing tool execution methods for AIAgent."""

    def _execute_tool_calls(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute tool calls from the assistant message and append results to messages.

        Dispatches to concurrent execution only for batches that look
        independent: read-only tools may always share the parallel path, while
        file reads/writes may do so only when their target paths do not overlap.
        """
        tool_calls = assistant_message.tool_calls

        # Allow _vprint during tool execution even with stream consumers
        self._executing_tools = True
        try:
            if not _should_parallelize_tool_batch(tool_calls):
                return self._execute_tool_calls_sequential(
                    assistant_message, messages, effective_task_id, api_call_count
                )

            return self._execute_tool_calls_concurrent(
                assistant_message, messages, effective_task_id, api_call_count
            )
        finally:
            self._executing_tools = False

    def _invoke_tool(self, function_name: str, function_args: dict, effective_task_id: str) -> str:
        """Invoke a single tool and return the result string. No display logic.

        Handles both agent-level tools (todo, memory, etc.) and registry-dispatched
        tools. Used by the concurrent execution path; the sequential path retains
        its own inline invocation for backward-compatible display handling.
        """
        if function_name == "todo":
            from tools.todo_tool import todo_tool as _todo_tool
            return _todo_tool(
                todos=function_args.get("todos"),
                merge=function_args.get("merge", False),
                store=self._todo_store,
            )
        elif function_name == "session_search":
            if not self._session_db:
                return json.dumps({"success": False, "error": "Session database not available."})
            from tools.session_search_tool import session_search as _session_search
            return _session_search(
                query=function_args.get("query", ""),
                role_filter=function_args.get("role_filter", ""),
                limit=function_args.get("limit", 3),
                db=self._session_db,
                current_session_id=self.session_id,
            )
        elif function_name == "memory":
            target = function_args.get("target", "memory")
            from tools.memory_tool import memory_tool as _memory_tool
            result = _memory_tool(
                action=function_args.get("action", ""),
                target=target,
                content=function_args.get("content"),
                old_text=function_args.get("old_text"),
                store=self._memory_store,
            )
            # Also send user observations to Honcho when active
            if self._honcho and target == "user" and function_args.get("action") == "add":
                self._honcho_save_user_observation(function_args.get("content", ""))
            return result
        elif function_name == "clarify":
            from tools.clarify_tool import clarify_tool as _clarify_tool
            return _clarify_tool(
                question=function_args.get("question", ""),
                choices=function_args.get("choices"),
                callback=self.clarify_callback,
            )
        elif function_name == "delegate_task":
            from tools.delegate_tool import delegate_task as _delegate_task
            return _delegate_task(
                goal=function_args.get("goal"),
                context=function_args.get("context"),
                toolsets=function_args.get("toolsets"),
                tasks=function_args.get("tasks"),
                max_iterations=function_args.get("max_iterations"),
                parent_agent=self,
            )
        else:
            return handle_function_call(
                function_name, function_args, effective_task_id,
                enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                honcho_manager=self._honcho,
                honcho_session_key=self._honcho_session_key,
            )

    def _execute_tool_calls_concurrent(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute multiple tool calls concurrently using a thread pool.

        Results are collected in the original tool-call order and appended to
        messages list atomically after all workers complete.
        """
        tool_calls = assistant_message.tool_calls
        num_tools = len(tool_calls)
        results = [None] * num_tools

        # Import display helpers (conditional import handled at module level)
        from hermes_cli.display_output import (
            KawaiiSpinner,
            build_tool_preview as _build_tool_preview,
            get_cute_tool_message as _get_cute_tool_message_impl,
            _detect_tool_failure,
            get_tool_emoji as _get_tool_emoji,
        )

        # Preview each tool call
        if self.quiet_mode:
            for i, tc in enumerate(tool_calls):
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                preview = _build_tool_preview(name, args)
                emoji = _get_tool_emoji(name)
                if len(preview) > 30:
                    preview = preview[:27] + "..."
                self._vprint(f"  {emoji} {preview}")

        # Spinner for concurrent execution
        spinner = None
        if self.quiet_mode:
            face = random.choice(KawaiiSpinner.KAWAII_WAITING)
            spinner = KawaiiSpinner(f"{face} ⚡ running {num_tools} tools concurrently", spinner_type='dots')
        if spinner:
            spinner.start()

        def _run_single_tool(tc, idx):
            """Worker function for thread pool."""
            tool_start_time = time.time()
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            try:
                result = self._invoke_tool(name, args, effective_task_id)
                is_error, _ = _detect_tool_failure(name, result)
                return (idx, name, args, result, time.time() - tool_start_time, is_error)
            except Exception as e:
                return (idx, name, args, str(e), time.time() - tool_start_time, True)

        # Execute concurrently
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_tools, _MAX_TOOL_WORKERS)) as executor:
                futures = [executor.submit(_run_single_tool, tc, i) for i, tc in enumerate(tool_calls)]
                for future in concurrent.futures.as_completed(futures):
                    idx, name, args, result, duration, is_error = future.result()
                    results[idx] = (name, args, result, duration, is_error)
        finally:
            if spinner:
                spinner.stop()

        # Process results in order
        for i, (name, args, result, duration, is_error) in enumerate(results):
            if result is None:
                continue
            cute_msg = _get_cute_tool_message_impl(name, args, duration, result=result)
            if self.quiet_mode:
                self._vprint(f" {cute_msg}")

            # Truncate large results
            MAX_TOOL_RESULT_CHARS = 100_000
            if len(result) > MAX_TOOL_RESULT_CHARS:
                original_len = len(result)
                result = (
                    result[:MAX_TOOL_RESULT_CHARS]
                    + f"\n\n[Truncated: tool response was {original_len:,} chars, "
                    f"exceeding the {MAX_TOOL_RESULT_CHARS:,} char limit]"
                )

            tool_msg = {
                "role": "tool",
                "content": result,
                "tool_call_id": tool_calls[i].id
            }
            messages.append(tool_msg)

        # Budget warning injection
        budget_warning = self._get_budget_warning(api_call_count)
        if budget_warning and messages and messages[-1].get("role") == "tool":
            last_content = messages[-1]["content"]
            try:
                parsed = json.loads(last_content)
                if isinstance(parsed, dict):
                    parsed["_budget_warning"] = budget_warning
                    messages[-1]["content"] = json.dumps(parsed, ensure_ascii=False)
                else:
                    messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            except (json.JSONDecodeError, TypeError):
                messages[-1]["content"] = last_content + f"\n\n{budget_warning}"

    def _execute_tool_calls_sequential(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute tool calls one at a time with full display handling."""
        tool_calls = assistant_message.tool_calls

        # Import display helpers
        from hermes_cli.display_output import (
            KawaiiSpinner,
            build_tool_preview as _build_tool_preview,
            get_cute_tool_message as _get_cute_tool_message_impl,
            _detect_tool_failure,
            get_tool_emoji as _get_tool_emoji,
        )

        for i, tool_call in enumerate(tool_calls):
            if self._interrupt_requested:
                remaining = len(tool_calls) - i
                self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)", force=True)
                for skipped_tc in tool_calls[i:]:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                        "tool_call_id": skipped_tc.id
                    }
                    messages.append(skip_msg)
                break

            tool_start_time = time.time()
            function_name = tool_call.function.name
            try:
                function_args = json.loads(tool_call.function.arguments)
            except Exception:
                function_args = {}

            # Display preview
            if self.quiet_mode:
                preview = _build_tool_preview(function_name, function_args) or function_name
                emoji = _get_tool_emoji(function_name)
                if len(preview) > 30:
                    preview = preview[:27] + "..."
                self._vprint(f"  {emoji} {preview}")

            # Special handling for tools with spinners
            spinner = None
            if function_name == "todo":
                from tools.todo_tool import todo_tool as _todo_tool
                function_result = _todo_tool(
                    todos=function_args.get("todos"),
                    merge=function_args.get("merge", False),
                    store=self._todo_store,
                )
                tool_duration = time.time() - tool_start_time
                if self.quiet_mode:
                    self._vprint(f" {_get_cute_tool_message_impl('todo', function_args, tool_duration, result=function_result)}")
            elif function_name == "session_search":
                if not self._session_db:
                    function_result = json.dumps({"success": False, "error": "Session database not available."})
                else:
                    from tools.session_search_tool import session_search as _session_search
                    function_result = _session_search(
                        query=function_args.get("query", ""),
                        role_filter=function_args.get("role_filter", ""),
                        limit=function_args.get("limit", 3),
                        db=self._session_db,
                        current_session_id=self.session_id,
                    )
                tool_duration = time.time() - tool_start_time
                if self.quiet_mode:
                    self._vprint(f" {_get_cute_tool_message_impl('session_search', function_args, tool_duration, result=function_result)}")
            elif function_name == "memory":
                target = function_args.get("target", "memory")
                from tools.memory_tool import memory_tool as _memory_tool
                function_result = _memory_tool(
                    action=function_args.get("action", ""),
                    target=target,
                    content=function_args.get("content"),
                    old_text=function_args.get("old_text"),
                    store=self._memory_store,
                )
                if self._honcho and target == "user" and function_args.get("action") == "add":
                    self._honcho_save_user_observation(function_args.get("content", ""))
                tool_duration = time.time() - tool_start_time
                if self.quiet_mode:
                    self._vprint(f" {_get_cute_tool_message_impl('memory', function_args, tool_duration, result=function_result)}")
            elif function_name == "clarify":
                from tools.clarify_tool import clarify_tool as _clarify_tool
                function_result = _clarify_tool(
                    question=function_args.get("question", ""),
                    choices=function_args.get("choices"),
                    callback=self.clarify_callback,
                )
                tool_duration = time.time() - tool_start_time
                if self.quiet_mode:
                    self._vprint(f" {_get_cute_tool_message_impl('clarify', function_args, tool_duration, result=function_result)}")
            elif function_name == "delegate_task":
                from tools.delegate_tool import delegate_task as _delegate_task
                tasks_arg = function_args.get("tasks")
                if tasks_arg and isinstance(tasks_arg, list):
                    spinner_label = f"🔀 delegating {len(tasks_arg)} tasks"
                else:
                    goal_preview = (function_args.get("goal") or "")[:30]
                    spinner_label = f"🔀 {goal_preview}" if goal_preview else "🔀 delegating"
                spinner = None
                if self.quiet_mode:
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    spinner = KawaiiSpinner(f"{face} {spinner_label}", spinner_type='dots')
                if spinner:
                    spinner.start()
                    self._delegate_spinner = spinner
                _delegate_result = None
                try:
                    function_result = _delegate_task(
                        goal=function_args.get("goal"),
                        context=function_args.get("context"),
                        toolsets=function_args.get("toolsets"),
                        tasks=tasks_arg,
                        max_iterations=function_args.get("max_iterations"),
                        parent_agent=self,
                    )
                    _delegate_result = function_result
                finally:
                    self._delegate_spinner = None
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl('delegate_task', function_args, tool_duration, result=_delegate_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self.quiet_mode:
                        self._vprint(f" {cute_msg}")
            else:
                # Default: use registry dispatch
                if self.quiet_mode:
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    if len(preview) > 30:
                        preview = preview[:27] + "..."
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots')
                    spinner.start()

                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                        honcho_manager=self._honcho,
                        honcho_session_key=self._honcho_session_key,
                    )
                finally:
                    tool_duration = time.time() - tool_start_time
                    if spinner:
                        spinner.stop(_get_cute_tool_message_impl(function_name, function_args, tool_duration, result=function_result))

            tool_duration = time.time() - tool_start_time
            if self.verbose_logging:
                self._vprint(f"  Tool {function_name} completed in {tool_duration:.2f}s")
                self._vprint(f"  Result ({len(function_result)} chars): {function_result}")

            # Truncate large results
            MAX_TOOL_RESULT_CHARS = 100_000
            if len(function_result) > MAX_TOOL_RESULT_CHARS:
                original_len = len(function_result)
                function_result = (
                    function_result[:MAX_TOOL_RESULT_CHARS]
                    + f"\n\n[Truncated: tool response was {original_len:,} chars, "
                    f"exceeding the {MAX_TOOL_RESULT_CHARS:,} char limit]"
                )

            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tool_call.id
            }
            messages.append(tool_msg)

            if not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                    print(f"   Result: {function_result}")
                else:
                    response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")

            if self._interrupt_requested and i < len(tool_calls):
                remaining = len(tool_calls) - i - 1
                self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)", force=True)
                for skipped_tc in tool_calls[i+1:]:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                        "tool_call_id": skipped_tc.id
                    }
                    messages.append(skip_msg)
                break

            if self.tool_delay > 0 and i < len(tool_calls) - 1:
                time.sleep(self.tool_delay)

        # Budget pressure injection
        budget_warning = self._get_budget_warning(api_call_count)
        if budget_warning and messages and messages[-1].get("role") == "tool":
            last_content = messages[-1]["content"]
            try:
                parsed = json.loads(last_content)
                if isinstance(parsed, dict):
                    parsed["_budget_warning"] = budget_warning
                    messages[-1]["content"] = json.dumps(parsed, ensure_ascii=False)
                else:
                    messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            except (json.JSONDecodeError, TypeError):
                messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            if not self.quiet_mode:
                remaining = self.max_iterations - api_call_count
                tier = "⚠️ WARNING" if remaining <= self.max_iterations * 0.1 else "💡 CAUTION"
                print(f"{self.log_prefix}{tier}: {remaining} iterations remaining")
