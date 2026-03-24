# Agent Mixins Unit Tests

Test suite for the AIAgent mixin classes.

## Running Tests

```bash
# Run all mixin tests
pytest tests/agent/mixins/ -v

# Run specific mixin tests
pytest tests/agent/mixins/test_context_mixin.py -v
pytest tests/agent/mixins/test_execution_mixin.py -v
pytest tests/agent/mixins/test_session_mixin.py -v
pytest tests/agent/mixins/test_streaming_mixin.py -v

# Run with coverage
pytest tests/agent/mixins/ -v --cov=agent/mixins
```

## Test Files

| File | Mixin | Tests |
|------|-------|-------|
| `test_context_mixin.py` | ContextMixin | 25+ tests |
| `test_execution_mixin.py` | ExecutionMixin | 15+ tests |
| `test_session_mixin.py` | SessionMixin | 15+ tests |
| `test_streaming_mixin.py` | StreamingMixin | 8+ tests |

## Coverage Goals

| Mixin | Target | Status |
|-------|--------|--------|
| ContextMixin | 80% | ✅ |
| ExecutionMixin | 85% | ✅ |
| SessionMixin | 80% | ✅ |
| StreamingMixin | 70% | ⏳ |
| ConversationMixin | TBD | ⏳ |

## Test Categories

### ContextMixin Tests
- URL detection (`_is_direct_openai_url`)
- Max tokens parameter (`_max_tokens_param`)
- Think block handling (`_strip_think_blocks`, `_has_content_after_think_block`)
- Reasoning extraction (`_extract_reasoning`)
- Tool call ID handling (`_split_responses_tool_id`, `_derive_responses_function_call_id`)
- Anthropic integration (`_anthropic_preserve_dots`)

### ExecutionMixin Tests
- Tool call repair (`_repair_tool_call`)
- Tool formatting (`_format_tools_for_system_message`)
- Destructive command detection (`_is_destructive_command`)
- Path overlap detection (`_paths_overlap`)

### SessionMixin Tests
- Session state reset (`reset_session_state`)
- Content cleaning (`_clean_session_content`)
- Message rollback (`_get_messages_up_to_last_assistant`)
- Honcho integration (`_honcho_should_activate`)
- Interrupt handling (`is_interrupted`)

### StreamingMixin Tests
- Stream consumer detection (`_has_stream_consumers`)
- Stream delta firing (`_fire_stream_delta`)
- Interruptible API calls (`_interruptible_streaming_api_call`)

## Adding New Tests

1. Create test methods following the pattern `test_<method_name>_<scenario>`
2. Use the `@pytest.fixture` for common setup
3. Mock external dependencies
4. Assert expected behavior
5. Run tests to verify

## Example Test

```python
def test_strip_think_blocks_with_think(self, mixin):
    """Test _strip_think_blocks removes <think> tags."""
    content = "<think>This is reasoning</think>Hello world"
    result = mixin._strip_think_blocks(content)
    assert result == "Hello world"
```

## Fixtures

Each test file provides a `mixin` fixture that creates a mock instance of the mixin with required attributes initialized.
