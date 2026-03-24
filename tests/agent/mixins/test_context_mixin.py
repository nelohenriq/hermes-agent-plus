"""Tests for ContextMixin.

ContextMixin provides context and API-related methods for AIAgent.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from types import SimpleNamespace


class TestContextMixin:
    """Test suite for ContextMixin methods."""

    @pytest.fixture
    def mixin(self):
        """Create a mock AIAgent with ContextMixin."""
        from agent.mixins import ContextMixin

        # Create a mock instance
        instance = ContextMixin()

        # Setup required attributes
        instance._base_url_lower = "https://openrouter.ai/api/v1"
        instance.model = "anthropic/claude-sonnet-4"
        instance.provider = "openrouter"
        instance.api_mode = "chat_completions"
        instance.max_tokens = 4096
        instance.valid_tool_names = {"test_tool"}
        instance.tools = [{"function": {"name": "test_tool"}}]
        instance.session_id = "test_session"
        instance.skip_context_files = False
        instance._cached_system_prompt = None

        return instance

    # ==================================================================
    # URL Detection Tests
    # ==================================================================

    def test_is_direct_openai_url_true(self, mixin):
        """Test detection of direct OpenAI URLs."""
        mixin._base_url_lower = "https://api.openai.com/v1"
        assert mixin._is_direct_openai_url() is True

    def test_is_direct_openai_url_false_openrouter(self, mixin):
        """Test that OpenRouter URLs are not detected as direct OpenAI."""
        mixin._base_url_lower = "https://openrouter.ai/api/v1"
        assert mixin._is_direct_openai_url() is False

    def test_is_direct_openai_url_custom_base(self, mixin):
        """Test with custom base_url parameter."""
        mixin._base_url_lower = "https://example.com/v1"
        assert mixin._is_direct_openai_url("https://api.openai.com/v1") is True

    # ==================================================================
    # Max Tokens Parameter Tests
    # ==================================================================

    def test_max_tokens_param_direct_openai(self, mixin):
        """Test max_tokens_param returns max_completion_tokens for direct OpenAI."""
        mixin._base_url_lower = "https://api.openai.com/v1"
        result = mixin._max_tokens_param(4096)
        assert result == {"max_completion_tokens": 4096}

    def test_max_tokens_param_openrouter(self, mixin):
        """Test max_tokens_param returns max_tokens for OpenRouter."""
        mixin._base_url_lower = "https://openrouter.ai/api/v1"
        result = mixin._max_tokens_param(4096)
        assert result == {"max_tokens": 4096}

    # ==================================================================
    # Think Block Tests
    # ==================================================================

    def test_strip_think_blocks_empty(self, mixin):
        """Test _strip_think_blocks with empty content."""
        assert mixin._strip_think_blocks("") == ""
        assert mixin._strip_think_blocks(None) == ""

    def test_strip_think_blocks_no_tags(self, mixin):
        """Test _strip_think_blocks with no think tags."""
        content = "Hello world"
        assert mixin._strip_think_blocks(content) == "Hello world"

    def test_strip_think_blocks_with_think(self, mixin):
        """Test _strip_think_blocks removes <think> tags."""
        content = "<think>This is reasoning</think>Hello world"
        assert mixin._strip_think_blocks(content) == "Hello world"

    def test_strip_think_blocks_with_thinking(self, mixin):
        """Test _strip_think_blocks removes <thinking> tags."""
        content = "<thinking>This is reasoning</thinking>Hello world"
        assert mixin._strip_think_blocks(content) == "Hello world"

    def test_strip_think_blocks_with_thinking_uppercase(self, mixin):
        """Test _strip_think_blocks removes <THINKING> tags (case insensitive)."""
        content = "<THINKING>This is reasoning</THINKING>Hello world"
        assert mixin._strip_think_blocks(content) == "Hello world"

    def test_strip_think_blocks_with_reasoning(self, mixin):
        """Test _strip_think_blocks removes <reasoning> tags."""
        content = "<reasoning>This is reasoning</reasoning>Hello world"
        assert mixin._strip_think_blocks(content) == "Hello world"

    def test_strip_think_blocks_with_scratchpad(self, mixin):
        """Test _strip_think_blocks removes <REASONING_SCRATCHPAD> tags."""
        content = (
            "<REASONING_SCRATCHPAD>This is reasoning</REASONING_SCRATCHPAD>Hello world"
        )
        assert mixin._strip_think_blocks(content) == "Hello world"

    def test_strip_think_blocks_multiline(self, mixin):
        """Test _strip_think_blocks with multiline content."""
        content = """<think>
This is multiline
reasoning content
</think>
Hello world
This is the answer"""
        result = mixin._strip_think_blocks(content)
        assert "Hello world" in result
        assert "multilinereasoning" not in result

    # ==================================================================
    # Content After Think Block Tests
    # ==================================================================

    def test_has_content_after_think_block_empty(self, mixin):
        """Test _has_content_after_think_block with empty content."""
        assert mixin._has_content_after_think_block("") is False

    def test_has_content_after_think_block_only_think(self, mixin):
        """Test _has_content_after_think_block with only think block."""
        content = "<think>This is reasoning</think>"
        assert mixin._has_content_after_think_block(content) is False

    def test_has_content_after_think_block_with_content(self, mixin):
        """Test _has_content_after_think_block with content after think block."""
        content = "<think>This is reasoning</think>Hello world"
        assert mixin._has_content_after_think_block(content) is True

    def test_has_content_after_think_block_whitespace_only(self, mixin):
        """Test _has_content_after_think_block with only whitespace after."""
        content = "<think>This is reasoning</think>   \n\t  "
        assert mixin._has_content_after_think_block(content) is False

    # ==================================================================
    # Reasoning Extraction Tests
    # ==================================================================

    def test_extract_reasoning_from_reasoning_field(self, mixin):
        """Test _extract_reasoning extracts from reasoning field."""
        msg = SimpleNamespace(
            reasoning="This is reasoning",
            reasoning_content=None,
            reasoning_details=None,
        )
        result = mixin._extract_reasoning(msg)
        assert result == "This is reasoning"

    def test_extract_reasoning_from_reasoning_content_field(self, mixin):
        """Test _extract_reasoning extracts from reasoning_content field."""
        msg = SimpleNamespace(
            reasoning=None,
            reasoning_content="This is reasoning content",
            reasoning_details=None,
        )
        result = mixin._extract_reasoning(msg)
        assert result == "This is reasoning content"

    def test_extract_reasoning_from_reasoning_details(self, mixin):
        """Test _extract_reasoning extracts from reasoning_details array."""
        msg = SimpleNamespace(
            reasoning=None,
            reasoning_content=None,
            reasoning_details=[
                {"type": "reasoning.summary", "summary": "Detail 1"},
                {"type": "reasoning.summary", "summary": "Detail 2"},
            ],
        )
        result = mixin._extract_reasoning(msg)
        assert result == "Detail 1\n\nDetail 2"

    def test_extract_reasoning_no_reasoning(self, mixin):
        """Test _extract_reasoning returns None when no reasoning."""
        msg = SimpleNamespace(
            reasoning=None, reasoning_content=None, reasoning_details=None
        )
        result = mixin._extract_reasoning(msg)
        assert result is None

    def test_extract_reasoning_combined(self, mixin):
        """Test _extract_reasoning combines multiple sources."""
        msg = SimpleNamespace(
            reasoning="Primary reasoning",
            reasoning_content="Secondary reasoning",
            reasoning_details=[{"type": "reasoning.summary", "summary": "Detail"}],
        )
        result = mixin._extract_reasoning(msg)
        assert "Primary reasoning" in result
        assert "Secondary reasoning" in result
        assert "Detail" in result

    # ==================================================================
    # Tool Call ID Splitting Tests
    # ==================================================================

    def test_split_responses_tool_id_with_separator(self, mixin):
        """Test _split_responses_tool_id with __ separator."""
        result = mixin._split_responses_tool_id("call_123__fc_456")
        assert result == ("call_123", "fc_456")

    def test_split_responses_tool_id_no_separator(self, mixin):
        """Test _split_responses_tool_id without separator."""
        result = mixin._split_responses_tool_id("call_123")
        assert result == ("call_123", None)

    def test_split_responses_tool_id_none(self, mixin):
        """Test _split_responses_tool_id with None input."""
        result = mixin._split_responses_tool_id(None)
        assert result == (None, None)

    def test_split_responses_tool_id_empty(self, mixin):
        """Test _split_responses_tool_id with empty string."""
        result = mixin._split_responses_tool_id("")
        assert result[0] in (None, "")  # Empty string or None
        assert result[1] is None

    # ==================================================================
    # Function Call ID Derivation Tests
    # ==================================================================

    def test_derive_responses_function_call_id_with_response_item_id(self, mixin):
        """Test _derive_responses_function_call_id uses existing response_item_id."""
        result = mixin._derive_responses_function_call_id("call_123", "fc_456")
        assert result == "fc_456"

    def test_derive_responses_function_call_id_without_response_item_id(self, mixin):
        """Test _derive_responses_function_call_id generates new ID."""
        result = mixin._derive_responses_function_call_id("call_123", None)
        assert result.startswith("fc_")
        assert len(result) == 27  # "fc_" + 24 hex chars

    # ==================================================================