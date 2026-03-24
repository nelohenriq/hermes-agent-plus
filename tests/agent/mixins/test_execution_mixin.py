"""Tests for ExecutionMixin.

ExecutionMixin provides tool execution methods for AIAgent.
"""

import pytest
from unittest.mock import Mock, patch


class TestExecutionMixin:
    """Test suite for ExecutionMixin methods."""

    @pytest.fixture
    def mixin(self):
        """Create a mock AIAgent with ExecutionMixin."""
        from agent.mixins import ExecutionMixin
        
        instance = ExecutionMixin()
        instance.valid_tool_names = {"read_file", "write_file", "terminal", "clarify"}
        return instance

    # ==================================================================
    # Tool Call Repair Tests
    # ==================================================================

    def test_repair_tool_call_exact_match(self, mixin):
        """Test _repair_tool_call with exact match."""
        result = mixin._repair_tool_call("read_file")
        assert result == "read_file"

    def test_repair_tool_call_lowercase(self, mixin):
        """Test _repair_tool_call repairs lowercase."""
        result = mixin._repair_tool_call("READ_FILE")
        assert result == "read_file"

    def test_repair_tool_call_normalized(self, mixin):
        """Test _repair_tool_call repairs normalized names."""
        result = mixin._repair_tool_call("read-file")
        assert result == "read_file"

    def test_repair_tool_call_with_spaces(self, mixin):
        """Test _repair_tool_call repairs names with spaces."""
        # Add a tool with underscore
        mixin.valid_tool_names = {"test_tool"}
        result = mixin._repair_tool_call("test tool")
        assert result == "test_tool"

    def test_repair_tool_call_fuzzy_match(self, mixin):
        """Test _repair_tool_call uses fuzzy matching."""
        result = mixin._repair_tool_call("read_fle")  # Typo
        assert result == "read_file"

    def test_repair_tool_call_no_match(self, mixin):
        """Test _repair_tool_call returns None for no match."""
        result = mixin._repair_tool_call("nonexistent_tool_xyz")
        assert result is None

    # ==================================================================
    # Tool Formatting Tests
    # ==================================================================

    def test_format_tools_for_system_message_empty(self, mixin):
        """Test _format_tools_for_system_message with no tools."""
        mixin.tools = []
        result = mixin._format_tools_for_system_message()
        assert result == "[]"

    def test_format_tools_for_system_message_with_tools(self, mixin):
        """Test _format_tools_for_system_message formats tools correctly."""
        mixin.tools = [
            {
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
                }
            }
        ]
        result = mixin._format_tools_for_system_message()
        import json
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "read_file"
        assert parsed[0]["description"] == "Read a file"

    # ==================================================================
    # Parallelization Helper Tests
    # ==================================================================

    def test_is_destructive_command_rm(self, mixin):
        """Test _is_destructive_command detects rm."""
        from agent.mixins.execution_mixin import _is_destructive_command
        assert _is_destructive_command("rm -rf /") is True
        assert _is_destructive_command("rm file.txt") is True

    def test_is_destructive_command_del(self, mixin):
        """Test _is_destructive_command detects del."""
        from agent.mixins.execution_mixin import _is_destructive_command
        assert _is_destructive_command("del /f file.txt") is True

    def test_is_destructive_command_safe(self, mixin):
        """Test _is_destructive_command allows safe commands."""
        from agent.mixins.execution_mixin import _is_destructive_command
        assert _is_destructive_command("ls -la") is False
        assert _is_destructive_command("echo hello") is False

    # ==================================================================
    # Path Overlap Tests
    # ==================================================================

    def test_paths_overlap_same_path(self, mixin):
        """Test _paths_overlap with same path."""
        from agent.mixins.execution_mixin import _paths_overlap
        assert _paths_overlap("/path/to/file", "/path/to/file") is True

    def test_paths_overlap_different_paths(self, mixin):
        """Test _paths_overlap with different paths."""
        from agent.mixins.execution_mixin import _paths_overlap
        assert _paths_overlap("/path/to/file1", "/path/to/file2") is False

    def test_paths_overlap_subdirectory(self, mixin):
        """Test _paths_overlap with subdirectory."""
        from agent.mixins.execution_mixin import _paths_overlap
        assert _paths_overlap("/path/to/dir", "/path/to/dir/file") is True
