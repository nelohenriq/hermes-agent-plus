"""
Context Compaction Manager for Token Efficiency

Provides intelligent context window management with multiple compaction strategies:
- Token window: Keep most recent tokens within a limit
- Turn window: Keep most recent conversation turns
- Summarization: Use LLM to summarize older content
- Hybrid: Combine strategies based on context size

Features:
- Multiple compaction strategies
- Configurable thresholds and limits
- Integration with existing ContextCompressor
- Token savings tracking

Usage:
    from agent.context_compaction import ContextCompactionManager

    manager = ContextCompactionManager()

    # Compact conversation history
    result = manager.compact(messages, target_tokens=4000)

    # Check if compaction is needed
    if manager.should_compact(messages, target_tokens=4000):
        compacted = manager.compact(messages, target_tokens=4000)
        print(f"Saved {compacted.tokens_saved} tokens")
"""

import time
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

try:
    from hermes.agent.context_compressor import ContextCompressor
    from hermes.agent.model_metadata import estimate_messages_tokens_rough
    _FULL_HERMES_AVAILABLE = True
except ImportError:
    # Fallback for standalone usage without full Hermes dependencies
    ContextCompressor = None
    estimate_messages_tokens_rough = lambda msgs: sum(len(str(m.get("content", ""))) // 4 for m in msgs)
    _FULL_HERMES_AVAILABLE = False


class CompactionStrategy(Enum):
    """Available compaction strategies."""
    TOKEN_WINDOW = "token_window"
    TURN_WINDOW = "turn_window"
    SUMMARIZATION = "summarization"
    HYBRID = "hybrid"


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    messages: List[Dict[str, Any]]
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    strategy_used: str
    compression_ratio: float
    metadata: Dict[str, Any]

    @property
    def successful(self) -> bool:
        """Check if compaction was successful."""
        return self.tokens_saved > 0


class ContextCompactionManager:
    """
    Intelligent context compaction with multiple strategies.

    Provides a unified interface for compacting conversation history
    using different strategies based on context size and requirements.
    """

    def __init__(
        self,
        strategy: CompactionStrategy = CompactionStrategy.HYBRID,
        token_window_limit: int = 4000,
        turn_window_limit: int = 10,
        summarization_threshold: int = 8000,
        summary_model: Optional[str] = None,
        llm_callback: Optional[Callable] = None,
    ):
        """
        Initialize the compaction manager.

        Args:
            strategy: Default compaction strategy
            token_window_limit: Max tokens for token window strategy
            turn_window_limit: Max turns for turn window strategy
            summarization_threshold: Token threshold for summarization
            summary_model: Model to use for summarization
            llm_callback: Callback for LLM calls (takes prompt, model, returns response)
        """
        self.strategy = strategy
        self.token_window_limit = token_window_limit
        self.turn_window_limit = turn_window_limit
        self.summarization_threshold = summarization_threshold
        self.summary_model = summary_model
        self.llm_callback = llm_callback

        # Initialize the underlying compressor
        self._compressor = None

    def _get_compressor(self, model: str, base_url: str = "", api_key: str = "", provider: str = ""):
        """Get or create a context compressor instance."""
        if self._compressor is None:
            if not _FULL_HERMES_AVAILABLE or ContextCompressor is None:
                return None

            # Build kwargs to avoid passing None for summary_model_override
            kwargs = {
                "model": model,
                "threshold_percent": 0.50,  # Compress at 50% of context limit
                "protect_first_n": 3,       # Protect first 3 messages
                "protect_last_n": 4,        # Protect last 4 messages
                "summary_target_tokens": 2500,
                "base_url": base_url,
                "api_key": api_key,
                "provider": provider,
            }
            if self.summary_model:
                kwargs["summary_model_override"] = self.summary_model

            try:
                self._compressor = ContextCompressor(**kwargs)
            except Exception:
                # Fallback if ContextCompressor fails
                return None
        return self._compressor

    def compact(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
        model: str = "gpt-4",
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        force_strategy: Optional[CompactionStrategy] = None,
    ) -> CompactionResult:
        """
        Compact conversation history to fit within target token limit.

        Args:
            messages: List of message dictionaries
            target_tokens: Target token count
            model: Model name for context length detection
            base_url: API base URL
            api_key: API key
            provider: Provider name
            force_strategy: Force a specific strategy (otherwise auto-select)

        Returns:
            CompactionResult with compacted messages and statistics
        """
        if not messages:
            return CompactionResult(
                messages=[],
                tokens_before=0,
                tokens_after=0,
                tokens_saved=0,
                strategy_used="none",
                compression_ratio=1.0,
                metadata={"reason": "empty_messages"}
            )

        # Estimate current token count
        compressor = self._get_compressor(model, base_url, api_key, provider)
        tokens_before = estimate_messages_tokens_rough(messages)

        # Check if compaction is needed
        if tokens_before <= target_tokens:
            return CompactionResult(
                messages=messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                tokens_saved=0,
                strategy_used="none",
                compression_ratio=1.0,
                metadata={"reason": "under_target"}
            )

        # Choose strategy
        strategy = force_strategy or self._choose_strategy(tokens_before, target_tokens)

        # Apply compaction
        if strategy == CompactionStrategy.TOKEN_WINDOW:
            result = self._apply_token_window(messages, target_tokens)
        elif strategy == CompactionStrategy.TURN_WINDOW:
            result = self._apply_turn_window(messages)
        elif strategy == CompactionStrategy.SUMMARIZATION:
            result = self._apply_summarization(messages, compressor)
        elif strategy == CompactionStrategy.HYBRID:
            result = self._apply_hybrid(messages, target_tokens, compressor)
        else:
            # Fallback to token window
            result = self._apply_token_window(messages, target_tokens)

        # Calculate final metrics
        tokens_after = estimate_messages_tokens_rough(result["messages"])
        tokens_saved = tokens_before - tokens_after
        compression_ratio = tokens_after / max(1, tokens_before)

        return CompactionResult(
            messages=result["messages"],
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            strategy_used=strategy.value,
            compression_ratio=compression_ratio,
            metadata=result.get("metadata", {})
        )

    def _choose_strategy(self, current_tokens: int, target_tokens: int) -> CompactionStrategy:
        """Choose the best compaction strategy based on context size."""
        if current_tokens < self.summarization_threshold:
            return CompactionStrategy.TOKEN_WINDOW
        elif current_tokens < self.summarization_threshold * 1.5:
            return CompactionStrategy.HYBRID
        else:
            return CompactionStrategy.SUMMARIZATION

    def _apply_token_window(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int
    ) -> Dict[str, Any]:
        """Apply token window compaction."""
        # Simple token-based truncation
        result_messages = []
        total_tokens = 0

        # Always keep system message if present
        system_msg = None
        if messages and messages[0].get("role") == "system":
            system_msg = messages[0]
            total_tokens += estimate_messages_tokens_rough([system_msg])
            result_messages.append(system_msg)
            messages = messages[1:]

        # Add messages from the end until we hit the token limit
        for msg in reversed(messages):
            msg_tokens = estimate_messages_tokens_rough([msg])
            if total_tokens + msg_tokens <= target_tokens:
                result_messages.insert(1 if system_msg else 0, msg)
                total_tokens += msg_tokens
            else:
                break

        return {
            "messages": result_messages,
            "metadata": {
                "strategy": "token_window",
                "target_tokens": target_tokens,
                "messages_kept": len(result_messages)
            }
        }

    def _apply_turn_window(
        self,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply turn window compaction (keep recent turns)."""
        # Count complete turns (user-assistant pairs)
        turns = []
        current_turn = []

        for msg in messages:
            current_turn.append(msg)

            if msg.get("role") == "assistant":
                turns.append(current_turn)
                current_turn = []

        # Keep only the most recent turns
        max_turns = self.turn_window_limit
        recent_turns = turns[-max_turns:] if len(turns) > max_turns else turns

        # Flatten back to messages
        result_messages = []
        for turn in recent_turns:
            result_messages.extend(turn)

        return {
            "messages": result_messages,
            "metadata": {
                "strategy": "turn_window",
                "turns_before": len(turns),
                "turns_after": len(recent_turns),
                "max_turns": max_turns
            }
        }

    def _apply_summarization(
        self,
        messages: List[Dict[str, Any]],
        compressor: Any
    ) -> Dict[str, Any]:
        """Apply summarization-based compaction using ContextCompressor."""
        try:
            # Use the existing ContextCompressor for summarization
            compressed_messages = compressor.compress(messages)

            return {
                "messages": compressed_messages,
                "metadata": {
                    "strategy": "summarization",
                    "used_context_compressor": True
                }
            }
        except Exception as e:
            # Fallback to token window if summarization fails
            print(f"Summarization failed: {e}, falling back to token window")
            return self._apply_token_window(messages, self.token_window_limit)

    def _apply_hybrid(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
        compressor: Any
    ) -> Dict[str, Any]:
        """Apply hybrid compaction (combine strategies)."""
        # First try turn window to reduce to a manageable size
        turn_result = self._apply_turn_window(messages)

        # Then apply token window if still over target
        turn_tokens = estimate_messages_tokens_rough(turn_result["messages"])
        if turn_tokens > target_tokens:
            return self._apply_token_window(turn_result["messages"], target_tokens)

        return turn_result

    def should_compact(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
        model: str = "gpt-4",
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> bool:
        """
        Check if messages should be compacted.

        Args:
            messages: Message list to check
            target_tokens: Target token count
            model: Model name for token estimation
            base_url: API base URL
            api_key: API key
            provider: Provider name

        Returns:
            True if compaction is recommended
        """
        if not messages:
            return False

        compressor = self._get_compressor(model, base_url, api_key, provider)
        current_tokens = estimate_messages_tokens_rough(messages)

        return current_tokens > target_tokens

    def estimate_tokens(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4",
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> int:
        """Estimate token count for messages."""
        compressor = self._get_compressor(model, base_url, api_key, provider)
        return estimate_messages_tokens_rough(messages)

    def get_stats(self) -> Dict[str, Any]:
        """Get compaction statistics."""
        if self._compressor:
            return self._compressor.get_status()
        return {
            "strategy": self.strategy.value,
            "token_window_limit": self.token_window_limit,
            "turn_window_limit": self.turn_window_limit,
            "summarization_threshold": self.summarization_threshold,
        }