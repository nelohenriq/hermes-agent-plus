"""Streaming mixin for AIAgent class.

This mixin provides streaming response handling capabilities including:
- Stream processing and chunk handling
- Streaming state management
- Response accumulation during streaming
- Stream interruption handling

Methods will be extracted from run_agent.py AIAgent class in Phase 2.
"""


class StreamingMixin:
    """Mixin providing streaming handler methods.
    
    This mixin will be composed into AIAgent to provide
    streaming-related functionality extracted from the monolithic
    run_agent.py implementation.
    
    Future methods (to be added in Phase 2):
    - process_stream_chunk()
    - handle_stream_response()
    - accumulate_response()
    - handle_stream_interrupt()
    - etc.
    """
    
    pass
