"""
Unified Token Efficiency Statistics

Combines reporting from prompt caching, rate limiting, and context compaction
into comprehensive token usage and cost savings analytics.

Features:
- Cross-feature statistics aggregation
- TOON format reporting for compactness
- Cost estimation and savings tracking
- Provider-specific breakdowns
- Time-series analysis

Usage:
    from agent.token_stats import TokenStats

    stats = TokenStats()

    # Record various events
    stats.record_cache_hit("anthropic", 500, "coder")
    stats.record_rate_limit_wait("anthropic", 2.5)
    stats.record_context_compaction(1000, 600)

    # Get comprehensive report
    report = stats.get_unified_stats()
    print(report.to_toon())
"""

import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta

try:
    from hermes.agent.toon import to_toon
except ImportError:
    # Fallback for standalone usage
    def to_toon(data):
        return str(data)


@dataclass
class TokenStatsSummary:
    """Summary of token efficiency metrics."""
    # Cache statistics
    cache_total_hits: int = 0
    cache_total_misses: int = 0
    cache_tokens_saved: int = 0
    cache_hit_rate: float = 0.0

    # Rate limiting statistics
    rate_limit_events: int = 0
    rate_limit_total_wait_time: float = 0.0
    rate_limit_avg_wait_time: float = 0.0

    # Context compaction statistics
    compaction_total_events: int = 0
    compaction_tokens_before: int = 0
    compaction_tokens_after: int = 0
    compaction_tokens_saved: int = 0
    compaction_avg_reduction: float = 0.0

    # Overall efficiency
    total_tokens_saved: int = 0
    estimated_cost_savings_usd: float = 0.0
    efficiency_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for TOON serialization."""
        return {
            "cache": {
                "total_hits": self.cache_total_hits,
                "total_misses": self.cache_total_misses,
                "tokens_saved": self.cache_tokens_saved,
                "hit_rate": round(self.cache_hit_rate, 3),
            },
            "rate_limiting": {
                "events": self.rate_limit_events,
                "total_wait_time": round(self.rate_limit_total_wait_time, 2),
                "avg_wait_time": round(self.rate_limit_avg_wait_time, 2),
            },
            "context_compaction": {
                "total_events": self.compaction_total_events,
                "tokens_before": self.compaction_tokens_before,
                "tokens_after": self.compaction_tokens_after,
                "tokens_saved": self.compaction_tokens_saved,
                "avg_reduction_percent": round(self.compaction_avg_reduction, 2),
            },
            "overall": {
                "total_tokens_saved": self.total_tokens_saved,
                "estimated_cost_savings_usd": round(self.estimated_cost_savings_usd, 4),
                "efficiency_score": round(self.efficiency_score, 2),
            }
        }

    def to_toon(self) -> str:
        """Export as TOON format string."""
        return to_toon({"token_efficiency_stats": self.to_dict()})


class TokenStats:
    """
    Unified token efficiency statistics tracker.

    Aggregates metrics from prompt caching, rate limiting, and context compaction
    to provide comprehensive insights into token usage optimization.
    """

    def __init__(self):
        self._lock = threading.RLock()

        # In-memory storage (could be extended to persistent storage)
        self._cache_hits: Dict[str, int] = {}  # provider -> count
        self._cache_misses: Dict[str, int] = {}  # provider -> count
        self._cache_tokens_saved: Dict[str, int] = {}  # provider -> tokens

        self._rate_limit_waits: List[float] = []  # List of wait times
        self._rate_limit_events: int = 0

        self._compaction_events: List[Dict[str, Any]] = []  # List of compaction results

        # Cost estimation (rough USD per 1K tokens)
        self._cost_per_1k_tokens = {
            "anthropic": 0.015,  # Claude Haiku
            "openai": 0.002,     # GPT-4o mini
            "google": 0.001,     # Gemini Flash
            "openrouter": 0.003, # Mixed providers
        }

    def record_cache_hit(self, provider: str, tokens_saved: int, profile: Optional[str] = None):
        """Record a prompt cache hit."""
        with self._lock:
            self._cache_hits[provider] = self._cache_hits.get(provider, 0) + 1
            self._cache_tokens_saved[provider] = self._cache_tokens_saved.get(provider, 0) + tokens_saved

    def record_cache_miss(self, provider: str):
        """Record a prompt cache miss."""
        with self._lock:
            self._cache_misses[provider] = self._cache_misses.get(provider, 0) + 1

    def record_rate_limit_wait(self, provider: str, wait_time: float):
        """Record a rate limit wait event."""
        with self._lock:
            self._rate_limit_events += 1
            self._rate_limit_waits.append(wait_time)

    def record_context_compaction(self, tokens_before: int, tokens_after: int, provider: Optional[str] = None):
        """Record a context compaction event."""
        with self._lock:
            self._compaction_events.append({
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_saved": tokens_before - tokens_after,
                "reduction_percent": ((tokens_before - tokens_after) / max(1, tokens_before)) * 100,
                "provider": provider,
                "timestamp": time.time()
            })

    def get_unified_stats(self) -> TokenStatsSummary:
        """Get comprehensive unified statistics."""
        with self._lock:
            summary = TokenStatsSummary()

            # Cache statistics
            total_hits = sum(self._cache_hits.values())
            total_misses = sum(self._cache_misses.values())
            total_cache_attempts = total_hits + total_misses

            summary.cache_total_hits = total_hits
            summary.cache_total_misses = total_misses
            summary.cache_tokens_saved = sum(self._cache_tokens_saved.values())
            summary.cache_hit_rate = (total_hits / max(1, total_cache_attempts)) if total_cache_attempts > 0 else 0.0

            # Rate limiting statistics
            summary.rate_limit_events = self._rate_limit_events
            summary.rate_limit_total_wait_time = sum(self._rate_limit_waits)
            summary.rate_limit_avg_wait_time = (
                summary.rate_limit_total_wait_time / max(1, len(self._rate_limit_waits))
                if self._rate_limit_waits else 0.0
            )

            # Context compaction statistics
            if self._compaction_events:
                summary.compaction_total_events = len(self._compaction_events)
                summary.compaction_tokens_before = sum(e["tokens_before"] for e in self._compaction_events)
                summary.compaction_tokens_after = sum(e["tokens_after"] for e in self._compaction_events)
                summary.compaction_tokens_saved = sum(e["tokens_saved"] for e in self._compaction_events)
                summary.compaction_avg_reduction = (
                    sum(e["reduction_percent"] for e in self._compaction_events) / len(self._compaction_events)
                )

            # Overall efficiency
            summary.total_tokens_saved = (
                summary.cache_tokens_saved +
                summary.compaction_tokens_saved
            )

            # Estimate cost savings (rough calculation)
            cost_per_token = sum(self._cost_per_1k_tokens.values()) / len(self._cost_per_1k_tokens) / 1000
            summary.estimated_cost_savings_usd = summary.total_tokens_saved * cost_per_token

            # Efficiency score (0-100, higher is better)
            # Combines cache hit rate, compaction reduction, and rate limit efficiency
            cache_score = summary.cache_hit_rate * 30  # 0-30 points
            compaction_score = min(30, summary.compaction_avg_reduction * 0.3)  # 0-30 points
            rate_limit_score = max(0, 40 - (summary.rate_limit_avg_wait_time * 10))  # 0-40 points

            summary.efficiency_score = cache_score + compaction_score + rate_limit_score

            return summary

    def get_provider_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics broken down by provider."""
        with self._lock:
            breakdown = {}

            # Get all providers that have data
            providers = set()
            providers.update(self._cache_hits.keys())
            providers.update(self._cache_misses.keys())
            providers.update(self._cache_tokens_saved.keys())

            for provider in providers:
                hits = self._cache_hits.get(provider, 0)
                misses = self._cache_misses.get(provider, 0)
                tokens_saved = self._cache_tokens_saved.get(provider, 0)

                total_attempts = hits + misses
                hit_rate = (hits / max(1, total_attempts)) if total_attempts > 0 else 0.0

                breakdown[provider] = {
                    "cache_hits": hits,
                    "cache_misses": misses,
                    "cache_hit_rate": round(hit_rate, 3),
                    "cache_tokens_saved": tokens_saved,
                }

            return breakdown

    def get_recent_activity(self, hours: int = 24) -> Dict[str, Any]:
        """Get statistics for recent activity."""
        cutoff_time = time.time() - (hours * 3600)

        with self._lock:
            recent_compactions = [
                e for e in self._compaction_events
                if e.get("timestamp", 0) > cutoff_time
            ]

            return {
                "hours": hours,
                "cache_hits_recent": sum(
                    hits for p, hits in self._cache_hits.items()
                    if self._is_provider_recent(p, cutoff_time)
                ),
                "compactions_recent": len(recent_compactions),
                "tokens_saved_recent": sum(e["tokens_saved"] for e in recent_compactions),
            }

    def _is_provider_recent(self, provider: str, cutoff_time: float) -> bool:
        """Check if provider has recent activity (simplified implementation)."""
        # In a real implementation, you'd track timestamps per provider
        # For now, assume all providers are recent
        return True

    def reset_stats(self):
        """Reset all statistics."""
        with self._lock:
            self._cache_hits.clear()
            self._cache_misses.clear()
            self._cache_tokens_saved.clear()
            self._rate_limit_waits.clear()
            self._rate_limit_events = 0
            self._compaction_events.clear()

    def export_stats_toon(self, include_breakdown: bool = True) -> str:
        """Export comprehensive statistics in TOON format."""
        summary = self.get_unified_stats()

        data = {"token_efficiency_stats": summary.to_dict()}

        if include_breakdown:
            data["provider_breakdown"] = self.get_provider_breakdown()
            data["recent_activity"] = self.get_recent_activity()

        return to_toon(data)


# Global instance for application-wide statistics
_global_stats = TokenStats()
_stats_lock = threading.Lock()


def get_global_stats() -> TokenStats:
    """Get the global token statistics instance."""
    return _global_stats


def record_cache_hit(provider: str, tokens_saved: int, profile: Optional[str] = None):
    """Record a cache hit event globally."""
    _global_stats.record_cache_hit(provider, tokens_saved, profile)


def record_cache_miss(provider: str):
    """Record a cache miss event globally."""
    _global_stats.record_cache_miss(provider)


def record_rate_limit_wait(provider: str, wait_time: float):
    """Record a rate limit wait event globally."""
    _global_stats.record_rate_limit_wait(provider, wait_time)


def record_context_compaction(tokens_before: int, tokens_after: int, provider: Optional[str] = None):
    """Record a context compaction event globally."""
    _global_stats.record_context_compaction(tokens_before, tokens_after, provider)


def get_efficiency_report() -> str:
    """Get a comprehensive efficiency report in TOON format."""
    return _global_stats.export_stats_toon()