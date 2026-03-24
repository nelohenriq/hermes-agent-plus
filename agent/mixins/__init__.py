"""Mixin classes for modular AIAgent composition.

This module provides mixin classes that encapsulate distinct
capabilities of the AIAgent class. Each mixin handles a specific
concern (conversation, execution, streaming, session), enabling
clean separation of concerns and easier testing.

Mixins are composed into AIAgent using multiple inheritance.
"""

# Import ExecutionMixin and its helper functions
from .execution_mixin import (
    ExecutionMixin,
    # Parallelization constants
    _NEVER_PARALLEL_TOOLS,
    _PARALLEL_SAFE_TOOLS,
    _PATH_SCOPED_TOOLS,
    _MAX_TOOL_WORKERS,
    # Helper functions
    _is_destructive_command,
    _should_parallelize_tool_batch,
    _extract_parallel_scope_path,
    _paths_overlap,
)

# Import StreamingMixin
from .streaming_mixin import StreamingMixin

# Placeholder imports - other mixins will be added in future phases
# from .conversation_mixin import ConversationMixin
# from .session_mixin import SessionMixin

__all__ = [
    'ExecutionMixin',
    'StreamingMixin',
    '_NEVER_PARALLEL_TOOLS',
    '_PARALLEL_SAFE_TOOLS',
    '_PATH_SCOPED_TOOLS',
    '_MAX_TOOL_WORKERS',
    '_is_destructive_command',
    '_should_parallelize_tool_batch',
    '_extract_parallel_scope_path',
    '_paths_overlap',
]
