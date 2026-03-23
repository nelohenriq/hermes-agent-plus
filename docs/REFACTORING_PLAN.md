# AIAgent Refactoring Plan

## Executive Summary

**File**: `run_agent.py` (7,616 lines)
**Class**: `AIAgent` (~100 methods, ~6,771 lines of method code)
**Problem**: God Class anti-pattern - single class handles too many responsibilities
**Goal**: Decompose into focused modules while maintaining backward compatibility

This plan proposes a phased extraction of responsibilities into dedicated modules under the existing `agent/` directory, using Python mixins and composition patterns to keep `AIAgent` as a facade class.

---

## 1. Current State Analysis

### 1.1 Method Distribution

| Category | Methods | Lines | Risk Level |
|----------|---------|-------|------------|
| Conversation | 16 | ~2,308 | **HIGH** |
| Tool Execution | 11 | ~691 | MEDIUM |
| Session/Init | 9 | ~906 | **HIGH** |
| Compression | 4 | ~325 | LOW |
| Streaming | 3 | ~268 | MEDIUM |
| Context/API | 5 | ~398 | MEDIUM |
| Display/Logging | 12+ | ~400 | LOW |
| Utility | 20+ | ~500 | LOW |

### 1.2 Current Module Dependencies

```
run_agent.py imports:
├── agent/anthropic_adapter.py    (Anthropic API handling)
├── agent/context_compressor.py   (Already extracted compression logic)
├── agent/display.py              (Display utilities)
├── agent/prompt_builder.py       (System prompt construction)
├── agent/rate_limiter.py         (Rate limiting)
├── agent/token_stats.py          (Token counting)
├── model_tools.py                (Tool definitions)
└── tools/*.py                    (Individual tool implementations)
```

### 1.3 Key State Variables (Tight Coupling)

The following instance variables are accessed across multiple method categories, making extraction challenging:

**Core State** (required by most methods):
- `self.client` - OpenAI/Anthropic client
- `self.model` - Current model name
- `self.messages` - Conversation history
- `self.session_id` - Current session identifier

**Session State** (reset per conversation):
- `self.session_total_tokens`, `session_input_tokens`, `session_output_tokens`
- `self.session_api_calls`, `session_estimated_cost_usd`
- `self._interrupt_requested`, `_executing_tools`

**Configuration State** (set in __init__):
- `self.max_iterations`, `self.quiet_mode`, `self.verbose_logging`
- `self.valid_tool_names`, `self.compression_enabled`
- `self.api_mode` (chat_completions/anthropic_messages/codex_responses)

**Internal State** (cross-cutting):
- `self._todo_store`, `self._memory_store`, `self._session_db`
- `self.context_compressor`, `self._checkpoint_mgr`
- `self._stream_callback`, `stream_delta_callback`

---

## 2. Proposed Module Structure

### 2.1 New Modules Under `agent/`

```
agent/
├── __init__.py              (existing - exports)
├── anthropic_adapter.py     (existing - Anthropic API)
├── auxiliary_client.py      (existing - client utilities)
├── context_compressor.py    (existing - compression logic)
├── display.py               (existing - display helpers)
├── prompt_builder.py        (existing - prompt construction)
├── rate_limiter.py          (existing - rate limiting)
├── token_stats.py           (existing - token counting)
│
├── conversation/            (NEW - conversation management
│   ├── __init__.py
│   ├── loop_handler.py      (Main conversation loop logic)
│   ├── initialization.py    (Conversation initialization)
│   ├── termination.py       (Loop termination conditions)
│   └── honcho_integration.py (Honcho client integration)
│
├── execution/               (NEW - tool execution
│   ├── __init__.py
│   ├── tool_executor.py     (Tool dispatch and execution)
│   ├── concurrent.py        (Concurrent execution logic)
│   └── sequential.py        (Sequential execution logic)
│
├── streaming/               (NEW - streaming API
│   ├── __init__.py
│   ├── handler.py           (Streaming API handler)
│   └── delta_dispatcher.py  (Delta event dispatch)
│
├── session/                 (NEW - session management
│   ├── __init__.py
│   ├── state_manager.py     (Session state tracking)
│   ├── db_logger.py         (SQLite session logging)
│   └── metrics.py           (Token/cost tracking)
│
└── mixins/                  (NEW - composition helpers
    ├── __init__.py
    ├── conversation_mixin.py (Conversation methods for AIAgent)
    ├── execution_mixin.py   (Tool execution methods)
    ├── streaming_mixin.py   (Streaming methods)
    └── session_mixin.py     (Session management methods)
```

### 2.2 Module/Class Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           run_agent.py                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                      AIAgent (Facade)                          │  │
│  │  - Public API: run_conversation(), chat()                      │  │
│  │  - Inherits: ConversationMixin, ExecutionMixin,                │  │
│  │              StreamingMixin, SessionMixin                       │  │
│  │  - Composes: ConversationLoopHandler, ToolExecutor,            │  │
│  │              StreamingHandler, SessionStateManager              │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     agent/conversation/                              │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ ConversationLoopHandler                                         │ │
│  │  - _run_conversation_loop() (1617 lines → decomposed)          │ │
│  │  - Dependencies: ToolExecutor, StreamingHandler, SessionDB     │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐ │
│  │ ConversationInitializer       │  │ ConversationTerminator       │ │
│  │  - _initialize_conversation() │  │  - _should_terminate()       │ │
│  │  - Message preparation        │  │  - Interrupt handling        │ │
│  │  - Task ID generation         │  │  - Completion detection      │ │
│  └──────────────────────────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      agent/execution/                                │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ ToolExecutor                                                    │ │
│  │  - _execute_tool_calls()      - dispatch                       │ │
│  │  - _invoke_tool()             - single tool call               │ │
│  │  - Dependencies: TodoStore, MemoryStore, SessionDB             │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐ │
│  │ ConcurrentToolExecutor       │  │ SequentialToolExecutor       │ │
│  │  - Thread pool execution     │  │  - One-at-a-time execution   │ │
│  │  - Result ordering           │  │  - Display integration       │ │
│  └──────────────────────────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      agent/streaming/                                │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ StreamingHandler                                                │ │
│  │  - _interruptible_streaming_api_call()                         │ │
│  │  - Chat/Anthropic/Codex streaming modes                        │ │
│  │  - Dependencies: StreamCallback, DeltaCallback                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────┐                                   │ │
│  │ DeltaDispatcher               │                                   │ │
│  │  - _fire_stream_delta()       │                                   │ │
│  │  - Consumer management        │                                   │ │
│  └──────────────────────────────┘                                   │ │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       agent/session/                                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ SessionStateManager                                             │ │
│  │  - reset_session_state()                                        │ │
│  │  - Token counting, cost tracking                               │ │
│  │  - Session lifecycle management                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐ │
│  │ SessionDBLogger               │  │ SessionMetrics               │ │
│  │  - SQLite operations          │  │  - Token counters            │ │
│  │  - Message persistence        │  │  - Cost calculations         │ │
│  └──────────────────────────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dependency Analysis

### 3.1 Import Dependencies

**Current run_agent.py imports** (to be distributed):
```python
# Standard library (keep in run_agent.py)
import atexit, asyncio, base64, concurrent.futures, copy, hashlib
import json, logging, os, random, re, sys, tempfile, time, threading
import weakref, uuid
from types import SimpleNamespace
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime
from pathlib import Path

# Internal modules (distribute to new modules)
from agent.anthropic_adapter import AnthropicAdapter
from agent.context_compressor import ContextCompressor
from agent.display import safe_print, ToolOutputDisplay
from agent.prompt_builder import build_system_prompt
from agent.rate_limiter import RateLimiter
from agent.token_stats import estimate_tokens_rough, estimate_messages_tokens_rough

# Tool system (keep in run_agent.py or execution module)
from model_tools import get_tool_definitions, handle_function_call, check_toolset_requirements
from tools.todo_tool import todo_tool
from tools.memory_tool import memory_tool
from tools.session_search_tool import session_search
from tools.clarify_tool import clarify_tool
from tools.delegate_tool import delegate_task
from tools.terminal_tool import cleanup_vm
from tools.browser_tool import cleanup_browser
from tools.interrupt import set_interrupt
```

### 3.2 Cross-Reference Map

```
Method                           | Accesses self.*               | Coupling Level
--------------------------------|-------------------------------|---------------
run_conversation()              | All state                     | CRITICAL
_run_conversation_loop()        | All state                     | CRITICAL
_initialize_conversation()      | messages, session_id, stores  | HIGH
_execute_tool_calls()           | client, messages, stores      | HIGH
_invoke_tool()                  | stores, session_id, honcho    | MEDIUM
_interruptible_streaming_api_call() | client, model, callbacks   | HIGH
_compress_context()             | compressor, session_db        | LOW
reset_session_state()           | All token counters            | LOW
_build_api_kwargs()             | model, tools, caching         | MEDIUM
```

### 3.3 State Access Patterns

**Pattern A: Read-Only State Access** (Easy to extract)
- Methods that only read `self.model`, `self.quiet_mode`, `self.verbose_logging`
- Example: `_build_api_kwargs()`

**Pattern B: Shared State Modification** (Requires careful synchronization)
- Methods that modify `self.messages`, token counters
- Example: `_run_conversation_loop()`, `_execute_tool_calls()`

**Pattern C: Callback Integration** (Tight coupling)
- Methods that invoke callbacks stored on `self`
- Example: `_fire_stream_delta()`, `_invoke_tool()`

---

## 4. Proposed Mixin Pattern

### 4.1 Mixin Architecture

The mixin pattern allows `AIAgent` to remain a single public class while delegating implementation to focused modules:

```python
# run_agent.py (after refactoring)
from agent.mixins import (
    ConversationMixin,
    ToolExecutionMixin,
    StreamingMixin,
    SessionMixin,
    ContextMixin,
)

class AIAgent(
    ConversationMixin,
    ToolExecutionMixin,
    StreamingMixin,
    SessionMixin,
    ContextMixin,
):
    """AI Agent with tool calling capabilities.
    
    This class serves as a facade, composing behavior from specialized mixins.
    Each mixin delegates to dedicated handler classes for implementation.
    """
    
    def __init__(self, **kwargs):
        # Initialize all mixins
        self._init_session_state()
        self._init_tool_executor()
        self._init_streaming_handler()
        self._init_conversation_handler()
        
        # ... existing __init__ logic
    
    # Public API remains unchanged
    def run_conversation(self, user_message: str, **kwargs) -> str:
        return self._conversation_handler.run(user_message, **kwargs)
    
    def chat(self, message: str) -> str:
        return self.run_conversation(message)
```

### 4.2 Mixin Definitions

```python
# agent/mixins/conversation_mixin.py
class ConversationMixin:
    """Provides conversation management methods for AIAgent."""
    
    _conversation_handler: 'ConversationLoopHandler'
    
    def _init_conversation_handler(self):
        self._conversation_handler = ConversationLoopHandler(
            agent=self,  # Reference back for state access
            tool_executor=self._tool_executor,
            streaming_handler=self._streaming_handler,
        )
    
    def run_conversation(self, user_message: str, **kwargs) -> str:
        return self._conversation_handler.run(user_message, **kwargs)
    
    # Legacy method - delegates to handler
    def _run_conversation_loop(self, messages, **kwargs):
        return self._conversation_handler.run_loop(messages, **kwargs)
```

```python
# agent/mixins/execution_mixin.py
class ToolExecutionMixin:
    """Provides tool execution methods for AIAgent."""
    
    _tool_executor: 'ToolExecutor'
    
    def _init_tool_executor(self):
        self._tool_executor = ToolExecutor(
            todo_store=self._todo_store,
            memory_store=self._memory_store,
            session_db=self._session_db,
            valid_tool_names=self.valid_tool_names,
        )
    
    def _execute_tool_calls(self, assistant_message, messages, task_id, count):
        return self._tool_executor.execute_calls(
            assistant_message, messages, task_id, count
        )
```

### 4.3 Handler Pattern

Handlers encapsulate complex logic while receiving a reference to the agent for state access:

```python
# agent/conversation/loop_handler.py
class ConversationLoopHandler:
    """Handles the main conversation loop logic."""
    
    def __init__(self, agent, tool_executor, streaming_handler):
        self.agent = agent  # Read-only access to agent state
        self.tool_executor = tool_executor
        self.streaming_handler = streaming_handler
    
    def run(self, user_message: str, **kwargs) -> str:
        messages = self._initialize_messages(user_message, **kwargs)
        
        while True:
            response = self._make_api_call(messages)
            
            if self._should_terminate(response):
                break
            
            if response.tool_calls:
                self.tool_executor.execute_calls(response, messages)
            
        return self._extract_final_response(messages)
```

---

## 5. Test Strategy

### 5.1 Existing Test Coverage

Current tests in `tests/test_run_agent.py`:
- Fixtures with mocked OpenAI client
- Basic conversation flow tests
- Tool invocation tests
- State management tests

### 5.2 Test Migration Strategy

**Phase 1: Regression Prevention**
1. Run full test suite before any changes: `pytest tests/test_run_agent.py -v`
2. Create integration test that exercises full conversation flow
3. Document baseline test coverage: `pytest --cov=run_agent tests/`

**Phase 2: Module-Level Testing**
For each extracted module, create dedicated tests:

```
tests/
├── agent/
│   ├── test_conversation_handler.py    (NEW)
│   ├── test_tool_executor.py           (NEW)
│   ├── test_streaming_handler.py       (NEW)
│   └── test_session_manager.py         (NEW)
```

**Phase 3: Contract Testing**
Use interfaces/protocols to define expected behavior:

```python
# agent/protocols.py
from typing import Protocol, List, Dict, Any

class ToolExecutorProtocol(Protocol):
    def execute_calls(self, assistant_message, messages, task_id, count) -> None: ...
    def invoke_tool(self, name: str, args: dict, task_id: str) -> str: ...

class StreamingHandlerProtocol(Protocol):
    def stream_api_call(self, api_kwargs: dict, on_first_delta: Callable) -> Any: ...
```

### 5.3 Test Coverage Targets

| Module | Target Coverage | Priority |
|--------|-----------------|----------|
| agent/conversation/loop_handler.py | 90% | HIGH |
| agent/execution/tool_executor.py | 95% | HIGH |
| agent/streaming/handler.py | 85% | MEDIUM |
| agent/session/state_manager.py | 90% | MEDIUM |
| run_agent.py (AIAgent) | 80% | LOW (facade only) |

### 5.4 Integration Test Scenarios

1. **Happy Path**: User message → API call → tool execution → response
2. **Interrupt Path**: User message → streaming → interrupt → cleanup
3. **Compression Path**: Large history → compression → continuation
4. **Multi-Tool Path**: Multiple tool calls → concurrent execution → response
5. **Error Recovery**: API error → retry → fallback → success

---

## 6. Implementation Phases

### Phase 1: Foundation (Low Risk)
**Duration**: 1-2 days
**Goal**: Extract simple, isolated methods

**Tasks**:
1. Create directory structure: `agent/conversation/`, `agent/execution/`, `agent/streaming/`, `agent/session/`
2. Extract `reset_session_state()` to `agent/session/state_manager.py`
3. Extract `_fire_stream_delta()` to `agent/streaming/delta_dispatcher.py`
4. Create mixin skeleton files in `agent/mixins/`

**Deliverables**:
- New directory structure
- First extracted module: `SessionStateManager`
- Basic mixin infrastructure

**Risk**: LOW
- No changes to core conversation loop
- Can be tested in isolation

**Verification**:
```bash
pytest tests/test_run_agent.py -v  # All existing tests pass
```

---

### Phase 2: Tool Execution Extraction (Medium Risk)
**Duration**: 2-3 days
**Goal**: Extract tool execution logic

**Tasks**:
1. Create `agent/execution/tool_executor.py`
2. Move `_execute_tool_calls()`, `_invoke_tool()`
3. Move `_execute_tool_calls_concurrent()`, `_execute_tool_calls_sequential()`
4. Create `ToolExecutionMixin` that delegates to `ToolExecutor`
5. Update `AIAgent` to inherit from `ToolExecutionMixin`

**Deliverables**:
- `ToolExecutor` class with full execution logic
- `ToolExecutionMixin` for AIAgent integration
- Tests for tool execution paths

**Risk**: MEDIUM
- Tool execution is critical path
- Thread pool concurrency needs careful testing

**Verification**:
```bash
pytest tests/test_run_agent.py tests/agent/test_tool_executor.py -v
```

**Rollback Plan**:
- Keep original methods in `run_agent.py` with deprecation warnings
- Use feature flag to switch between old/new implementation

---

### Phase 3: Streaming Extraction (Medium Risk)
**Duration**: 2-3 days
**Goal**: Extract streaming API handling

**Tasks**:
1. Create `agent/streaming/handler.py`
2. Move `_interruptible_streaming_api_call()`
3. Move `_has_stream_consumers()`
4. Create `StreamingMixin`
5. Update AIAgent to inherit from `StreamingMixin`

**Deliverables**:
- `StreamingHandler` class
- `StreamingMixin` for integration
- Tests for streaming paths

**Risk**: MEDIUM
- Three API modes (chat_completions, anthropic_messages, codex_responses)
- Callback management is delicate

**Verification**:
```bash
pytest tests/test_streaming.py -v
pytest tests/test_run_agent.py -k stream -v
```

---

### Phase 4: Session Management Extraction (Medium Risk)
**Duration**: 1-2 days
**Goal**: Extract session lifecycle management

**Tasks**:
1. Create `agent/session/db_logger.py`
2. Move session database operations from `AIAgent`
3. Create `agent/session/metrics.py` for token/cost tracking
4. Create `SessionMixin`

**Deliverables**:
- `SessionDBLogger` class
- `SessionMetrics` class
- `SessionMixin` for integration

**Risk**: MEDIUM
- SQLite session persistence is critical for state recovery

**Verification**:
```bash
pytest tests/test_run_agent.py -k session -v
```

---

### Phase 5: Conversation Loop Extraction (HIGH Risk)
**Duration**: 5-7 days
**Goal**: Extract the massive `_run_conversation_loop()` method

**Tasks**:
1. Analyze `_run_conversation_loop()` (1617 lines!) and identify sub-sections:
   - API call preparation
   - Response handling
   - Tool call dispatch
   - Context pressure monitoring
   - Interrupt handling
   - Retry logic

2. Create `agent/conversation/loop_handler.py` with decomposed methods:
   ```python
   class ConversationLoopHandler:
       def run_loop(self, messages, **kwargs) -> str:
           while not self._should_terminate():
               response = self._make_api_call(messages)
               self._handle_response(response, messages)
               if response.tool_calls:
                   self._execute_tools(response, messages)
           return self._extract_result(messages)
   ```

3. Create `agent/conversation/initialization.py`:
   - Move `_initialize_conversation()`
   - Move message preparation logic

4. Create `agent/conversation/termination.py`:
   - Extract termination condition checks
   - Extract interrupt handling

5. Create `ConversationMixin`

**Deliverables**:
- `ConversationLoopHandler` with decomposed loop logic
- `ConversationInitializer`
- `ConversationTerminator`
- `ConversationMixin` for integration

**Risk**: HIGH
- Core conversation loop is the heart of the agent
- Many edge cases and error paths
- 1617 lines of complex branching logic

**Verification**:
```bash
# Run all tests including integration tests
pytest tests/ -v --cov=run_agent --cov=agent/conversation

# Manual smoke test
python -c "from run_agent import AIAgent; a = AIAgent(api_key='test'); print('OK')"
```

**Rollback Plan**:
- Keep original `_run_conversation_loop()` as `_run_conversation_loop_legacy()`
- Use environment variable to toggle: `HERMES_USE_LEGACY_LOOP=1`

---

### Phase 6: Context/API Extraction (Medium Risk)
**Duration**: 2-3 days
**Goal**: Extract API call preparation and context building

**Tasks**:
1. Create `agent/context/api_builder.py`
2. Move `_build_api_kwargs()`
3. Move `_build_system_prompt()` (some already in prompt_builder.py)
4. Create `ContextMixin`

**Deliverables**:
- `APIBuilder` class
- Enhanced `prompt_builder.py` if needed
- `ContextMixin` for integration

**Risk**: MEDIUM
- API kwargs construction affects all API calls

**Verification**:
```bash
pytest tests/test_run_agent.py -k api -v
```

---

### Phase 7: Final Integration (Medium Risk)
**Duration**: 2-3 days
**Goal**: Complete mixin integration and cleanup

**Tasks**:
1. Update `AIAgent.__init__()` to initialize all handlers
2. Remove deprecated methods from `run_agent.py`
3. Update all imports across the codebase
4. Update `agent/__init__.py` exports
5. Documentation updates

**Deliverables**:
- Clean `AIAgent` class (~500 lines)
- Full mixin inheritance chain
- Updated documentation

**Risk**: MEDIUM
- Integration of all components
- Import path updates across codebase

**Verification**:
```bash
# Full test suite
pytest tests/ -v --cov=run_agent --cov=agent

# Type checking
mypy run_agent.py agent/

# Linting
ruff check run_agent.py agent/
```

---

## 7. Risk Assessment

### 7.1 Risk Matrix

| Phase | Risk | Impact | Mitigation |
|-------|------|--------|------------|
| 1: Foundation | LOW | Low | Isolated changes, easy rollback |
| 2: Tool Execution | MEDIUM | High | Feature flag, comprehensive tests |
| 3: Streaming | MEDIUM | High | Three-mode testing, fallback logic |
| 4: Session | MEDIUM | Medium | SQLite transaction safety |
| 5: Conversation | HIGH | Critical | Legacy toggle, phased sub-extraction |
| 6: Context | MEDIUM | Medium | API contract tests |
| 7: Integration | MEDIUM | High | Full regression suite |

### 7.2 High-Risk Extractions

**_run_conversation_loop() (1617 lines)**
- **Risk**: Critical path, many edge cases
- **Mitigation**: 
  1. Extract in sub-phases (API call, response, tools, termination)
  2. Keep legacy method available via flag
  3. Extensive manual testing with real conversations

**__init__ (726 lines)**
- **Risk**: Initialization order dependencies
- **Mitigation**:
  1. Document initialization sequence
  2. Use factory methods for complex setup
  3. Dependency injection for testability

### 7.3 Rollback Strategy

Each phase includes:
1. Git branch per phase: `refactor/phase-N-description`
2. Feature flags for new implementations
3. Legacy method retention for one release cycle
4. Automated rollback tests

---

## 8. Success Criteria

### 8.1 Quantitative Goals

| Metric | Current | Target |
|--------|---------|--------|
| run_agent.py lines | 7,616 | < 800 |
| AIAgent methods | ~100 | < 20 (public API) |
| Max method length | 1,617 | < 100 |
| Test coverage | ~70% | > 85% |
| Files in agent/ | 20 | 35+ |

### 8.2 Qualitative Goals

- [ ] Each module has single responsibility
- [ ] Clear interfaces between modules
- [ ] Testable in isolation
- [ ] Maintained backward compatibility
- [ ] Improved code navigation
- [ ] Reduced cognitive load per file

---

## 9. Implementation Order Summary

```
Week 1: Phase 1 (Foundation) + Phase 4 (Session)
Week 2: Phase 2 (Tool Execution)
Week 3: Phase 3 (Streaming) + Phase 6 (Context)
Week 4-5: Phase 5 (Conversation Loop) - Most complex
Week 6: Phase 7 (Integration) + Buffer
```

---

## 10. Appendix: Method Extraction Checklist

For each method being extracted:

- [ ] Identify all `self.*` attributes accessed
- [ ] Document read vs write access
- [ ] Identify callback/circular dependencies
- [ ] Create tests for method in isolation
- [ ] Create new module file with class
- [ ] Move method to new class (keep original temporarily)
- [ ] Update original method to delegate
- [ ] Add deprecation warning to original
- [ ] Run full test suite
- [ ] Remove original method after validation

---

**Document Version**: 1.0  
**Created**: 2026-03-23  
**Author**: Agent Zero Master Developer  
**Status**: PLANNING COMPLETE - Ready for Phase 1 implementation
