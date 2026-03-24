"""Tests for SessionMixin.

SessionMixin provides session management methods for AIAgent.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


class TestSessionMixin:
    """Test suite for SessionMixin methods."""

    @pytest.fixture
    def mixin(self):
        """Create a mock AIAgent with SessionMixin."""
        from agent.mixins import SessionMixin
        
        instance = SessionMixin()
        
        # Setup session state attributes
        instance.session_total_tokens = 1000
        instance.session_input_tokens = 600
        instance.session_output_tokens = 400
        instance.session_prompt_tokens = 600
        instance.session_completion_tokens = 400
        instance.session_cache_read_tokens = 100
        instance.session_cache_write_tokens = 50
        instance.session_api_calls = 5
        instance.session_reasoning_tokens = 200
        instance.session_estimated_cost_usd = 0.05
        instance.session_cost_status = "success"
        instance.session_cost_source = "api"
        
        # Mock context compressor
        instance.context_compressor = Mock()
        instance.context_compressor.last_prompt_tokens = 500
        instance.context_compressor.last_completion_tokens = 300
        instance.context_compressor.last_total_tokens = 800
        instance.context_compressor.compression_count = 2
        instance.context_compressor._context_probed = True
        
        return instance

    # ==================================================================
    # Session State Reset Tests
    # ==================================================================

    def test_reset_session_state_resets_token_counters(self, mixin):
        """Test reset_session_state resets all token counters to 0."""
        mixin.reset_session_state()
        
        assert mixin.session_total_tokens == 0
        assert mixin.session_input_tokens == 0
        assert mixin.session_output_tokens == 0
        assert mixin.session_prompt_tokens == 0
        assert mixin.session_completion_tokens == 0

    def test_reset_session_state_resets_cache_counters(self, mixin):
        """Test reset_session_state resets cache counters."""
        mixin.reset_session_state()
        
        assert mixin.session_cache_read_tokens == 0
        assert mixin.session_cache_write_tokens == 0

    def test_reset_session_state_resets_cost_tracking(self, mixin):
        """Test reset_session_state resets cost tracking."""
        mixin.reset_session_state()
        
        assert mixin.session_estimated_cost_usd == 0.0
        assert mixin.session_cost_status == "unknown"
        assert mixin.session_cost_source == "none"

    def test_reset_session_state_resets_compressor_counters(self, mixin):
        """Test reset_session_state resets context compressor counters."""
        mixin.reset_session_state()
        
        assert mixin.context_compressor.last_prompt_tokens == 0
        assert mixin.context_compressor.last_completion_tokens == 0
        assert mixin.context_compressor.last_total_tokens == 0
        assert mixin.context_compressor.compression_count == 0
        assert mixin.context_compressor._context_probed is False

    def test_reset_session_state_without_compressor(self, mixin):
        """Test reset_session_state works without context compressor."""
        mixin.context_compressor = None
        # Should not raise
        mixin.reset_session_state()

    # ==================================================================
    # Session Content Cleaning Tests
    # ==================================================================

    def test_clean_session_content_empty(self, mixin):
        """Test _clean_session_content with empty content."""
        result = SessionMixin._clean_session_content("")
        assert result == ""
        result = SessionMixin._clean_session_content(None)
        assert result is None

    def test_clean_session_content_no_scratchpad(self, mixin):
        """Test _clean_session_content without scratchpad tags."""
        content = "Hello world"
        result = SessionMixin._clean_session_content(content)
        assert result == "Hello world"

    def test_clean_session_content_with_scratchpad(self, mixin):
        """Test _clean_session_content converts scratchpad to think tags."""
        content = "<REASONING_SCRATCHPAD>This is reasoning</REASONING_SCRATCHPAD>Hello"
        result = SessionMixin._clean_session_content(content)
        assert "<REASONING_SCRATCHPAD>" not in result
        assert "<think>" in result or "think" in result.lower()

    def test_clean_session_content_strips_whitespace(self, mixin):
        """Test _clean_session_content strips whitespace."""
        content = "  Hello world  \n\n"
        result = SessionMixin._clean_session_content(content)
        assert result == "Hello world"

    # ==================================================================
    # Message Rollback Tests
    # ==================================================================

    def test_get_messages_up_to_last_assistant_empty(self, mixin):
        """Test _get_messages_up_to_last_assistant with empty list."""
        result = mixin._get_messages_up_to_last_assistant([])
        assert result == []

    def test_get_messages_up_to_last_assistant_no_assistant(self, mixin):
        """Test _get_messages_up_to_last_assistant with no assistant messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "How are you?"}
        ]
        result = mixin._get_messages_up_to_last_assistant(messages)
        assert result == messages  # Returns all messages

    def test_get_messages_up_to_last_assistant_with_assistant(self, mixin):
        """Test _get_messages_up_to_last_assistant removes last assistant message."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "tool", "content": "Result"},
            {"role": "assistant", "content": "Final answer"}
        ]
        result = mixin._get_messages_up_to_last_assistant(messages)
        assert len(result) == 3
        assert result[-1]["role"] == "tool"

    # ==================================================================
    # Honcho Integration Tests
    # ==================================================================

    def test_honcho_should_activate_enabled(self, mixin):
        """Test _honcho_should_activate returns True when enabled."""
        hcfg = Mock()
        hcfg.enabled = True
        hcfg.api_key = "test_key"
        
        result = mixin._honcho_should_activate(hcfg)
        assert result is True

    def test_honcho_should_activate_disabled(self, mixin):
        """Test _honcho_should_activate returns False when disabled."""
        hcfg = Mock()
        hcfg.enabled = False
        hcfg.api_key = "test_key"
        
        result = mixin._honcho_should_activate(hcfg)
        assert result is False

    def test_honcho_should_activate_no_api_key(self, mixin):
        """Test _honcho_should_activate returns False when no API key."""
        hcfg = Mock()
        hcfg.enabled = True
        hcfg.api_key = None
        
        result = mixin._honcho_should_activate(hcfg)
        assert result is False

    def test_honcho_should_activate_empty_api_key(self, mixin):
        """Test _honcho_should_activate returns False when empty API key."""
        hcfg = Mock()
        hcfg.enabled = True
        hcfg.api_key = ""
        
        result = mixin._honcho_should_activate(hcfg)
        assert result is False

    # ==================================================================
    # Interrupt Property Tests
    # ==================================================================

    def test_is_interrupted_false(self, mixin):
        """Test is_interrupted returns False by default."""
        mixin._interrupt_requested = False
        assert mixin.is_interrupted is False

    def test_is_interrupted_true(self, mixin):
        """Test is_interrupted returns True when requested."""
        mixin._interrupt_requested = True
        assert mixin.is_interrupted is True
