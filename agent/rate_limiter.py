"""
Coordinated Rate Limiter for LLM API Efficiency

Provides cross-process coordination to prevent parallel subagents from
overloading API rate limits. Uses SQLite with file locking for coordination
across multiple processes and threads.

Features:
- Sliding window rate limiting
- Cross-process coordination via SQLite
- Thread-safe operations with file locking
- Provider-specific configurations
- Automatic backoff and queuing
- Token usage tracking

Usage:
    from agent.rate_limiter import get_rate_limiter

    # Get provider-specific limiter
    limiter = get_rate_limiter("anthropic")

    # Check before API call
    if limiter.can_proceed(estimated_tokens=100):
        # Make API call
        limiter.record_request(tokens_used=150)
    else:
        # Wait or use cache
        wait_time = limiter.get_wait_time()
        time.sleep(wait_time)
"""

import os
import sqlite3
import threading
import time
import fcntl
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class BackoffStrategy(Enum):
    """Backoff strategies when rate limited."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


@dataclass
class ProviderLimits:
    """Rate limit configuration for a provider."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    tokens_per_minute: int = 40000
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    max_backoff: float = 60.0
    min_backoff: float = 1.0


class CoordinatedRateLimiter:
    """
    Rate limiter with cross-process coordination.

    Uses SQLite with file locking to coordinate rate limiting across
    multiple processes and threads, preventing parallel subagents from
    overloading API limits.
    """

    DB_PATH = Path.home() / ".hermes" / "cache" / "rate_limits.db"

    # Default provider configurations
    DEFAULT_LIMITS = {
        "anthropic": ProviderLimits(
            requests_per_minute=60,
            requests_per_hour=1000,
            tokens_per_minute=40000,
        ),
        "openai": ProviderLimits(
            requests_per_minute=500,
            requests_per_hour=10000,
            tokens_per_minute=90000,
        ),
        "google": ProviderLimits(
            requests_per_minute=60,
            requests_per_hour=1500,
            tokens_per_minute=60000,
        ),
        "openrouter": ProviderLimits(
            requests_per_minute=100,
            requests_per_hour=2000,
            tokens_per_minute=50000,
        ),
    }

    def __init__(self, provider: str, coordinated: bool = True):
        """
        Initialize rate limiter for a provider.

        Args:
            provider: Provider name (anthropic, openai, etc.)
            coordinated: Whether to use cross-process coordination
        """
        self.provider = provider
        self.coordinated = coordinated
        self.limits = self.DEFAULT_LIMITS.get(provider, ProviderLimits())

        if coordinated:
            self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

        # In-memory state for non-coordinated mode
        self._lock = threading.RLock()
        self._minute_requests = 0
        self._hour_requests = 0
        self._minute_tokens = 0
        self._minute_start = time.time()
        self._hour_start = time.time()
        self._consecutive_limits = 0
        self._last_backoff = 0.0

    def _init_db(self):
        """Initialize SQLite database for coordination."""
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.executescript(f"""
                CREATE TABLE IF NOT EXISTS limits_{self.provider} (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    minute_requests INTEGER DEFAULT 0,
                    hour_requests INTEGER DEFAULT 0,
                    minute_tokens INTEGER DEFAULT 0,
                    minute_start REAL DEFAULT 0,
                    hour_start REAL DEFAULT 0,
                    consecutive_limits INTEGER DEFAULT 0,
                    last_backoff REAL DEFAULT 0
                );

                -- Insert default row if not exists
                INSERT OR IGNORE INTO limits_{self.provider} (id) VALUES (1);
            """)

    def _reset_windows_if_needed(self):
        """Reset time windows if expired."""
        now = time.time()

        if self.coordinated:
            return self._reset_windows_coordinated(now)
        else:
            return self._reset_windows_local(now)

    def _reset_windows_local(self, now: float):
        """Reset windows for local (non-coordinated) mode."""
        with self._lock:
            # Reset minute window
            if now - self._minute_start >= 60:
                self._minute_requests = 0
                self._minute_tokens = 0
                self._minute_start = now

            # Reset hour window
            if now - self._hour_start >= 3600:
                self._hour_requests = 0
                self._hour_start = now

            return {
                "minute_requests": self._minute_requests,
                "hour_requests": self._hour_requests,
                "minute_tokens": self._minute_tokens,
                "consecutive_limits": self._consecutive_limits,
                "last_backoff": self._last_backoff,
            }

    def _reset_windows_coordinated(self, now: float) -> Dict[str, Any]:
        """Reset windows for coordinated mode using SQLite."""
        # File locking for cross-process coordination
        lock_file = self.DB_PATH.with_suffix('.lock')
        lock_file.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_file, 'w') as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

            try:
                with sqlite3.connect(self.DB_PATH) as conn:
                    # Get current state
                    row = conn.execute(f"SELECT * FROM limits_{self.provider} WHERE id = 1").fetchone()
                    if not row:
                        # Initialize if missing
                        conn.execute(f"INSERT INTO limits_{self.provider} (id) VALUES (1)")
                        row = (1, 0, 0, 0, now, now, 0, 0.0)

                    minute_start = row[4] or now
                    hour_start = row[5] or now

                    # Reset windows if expired
                    updates = {}
                    if now - minute_start >= 60:
                        updates["minute_requests"] = 0
                        updates["minute_tokens"] = 0
                        updates["minute_start"] = now

                    if now - hour_start >= 3600:
                        updates["hour_requests"] = 0
                        updates["hour_start"] = now

                    if updates:
                        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
                        values = list(updates.values())
                        conn.execute(f"UPDATE limits_{self.provider} SET {set_clause} WHERE id = 1", values)

                    # Return current state
                    row = conn.execute(f"SELECT * FROM limits_{self.provider} WHERE id = 1").fetchone()
                    return {
                        "minute_requests": row[1],
                        "hour_requests": row[2],
                        "minute_tokens": row[3],
                        "consecutive_limits": row[6],
                        "last_backoff": row[7],
                    }

            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def can_proceed(self, estimated_tokens: int = 0) -> Dict[str, Any]:
        """
        Check if a request can proceed without hitting rate limits.

        Args:
            estimated_tokens: Estimated token usage for this request

        Returns:
            Dict with allowed status and wait time if needed
        """
        state = self._reset_windows_if_needed()
        now = time.time()

        # Check limits
        reasons = []

        if state["minute_requests"] >= self.limits.requests_per_minute:
            reasons.append(f"minute_requests ({state['minute_requests']}/{self.limits.requests_per_minute})")

        if state["hour_requests"] >= self.limits.requests_per_hour:
            reasons.append(f"hour_requests ({state['hour_requests']}/{self.limits.requests_per_hour})")

        if state["minute_tokens"] + estimated_tokens > self.limits.tokens_per_minute:
            reasons.append(f"minute_tokens ({state['minute_tokens']}/{self.limits.tokens_per_minute})")

        allowed = len(reasons) == 0

        if not allowed:
            wait_time = self._calculate_backoff(state["consecutive_limits"])
        else:
            wait_time = 0.0

        return {
            "allowed": allowed,
            "reason": "; ".join(reasons) if reasons else None,
            "wait_time": wait_time,
            "remaining_minute": max(0, self.limits.requests_per_minute - state["minute_requests"]),
            "remaining_hour": max(0, self.limits.requests_per_hour - state["hour_requests"]),
            "remaining_tokens": max(0, self.limits.tokens_per_minute - state["minute_tokens"] - estimated_tokens),
        }

    def _calculate_backoff(self, consecutive_limits: int) -> float:
        """Calculate backoff time based on strategy."""
        if self.limits.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            backoff = min(self.limits.max_backoff, self.limits.min_backoff * (2 ** consecutive_limits))
        elif self.limits.backoff_strategy == BackoffStrategy.LINEAR:
            backoff = min(self.limits.max_backoff, self.limits.min_backoff * (consecutive_limits + 1))
        else:  # FIXED
            backoff = self.limits.min_backoff

        return backoff

    def record_request(self, tokens_used: int = 0):
        """
        Record a completed request.

        Args:
            tokens_used: Actual tokens used in the request
        """
        now = time.time()

        if self.coordinated:
            self._record_request_coordinated(tokens_used, now)
        else:
            self._record_request_local(tokens_used, now)

    def _record_request_local(self, tokens_used: int, now: float):
        """Record request for local mode."""
        with self._lock:
            self._reset_windows_local(now)
            self._minute_requests += 1
            self._hour_requests += 1
            self._minute_tokens += tokens_used
            self._consecutive_limits = 0  # Reset on successful request

    def _record_request_coordinated(self, tokens_used: int, now: float):
        """Record request for coordinated mode."""
        lock_file = self.DB_PATH.with_suffix('.lock')

        with open(lock_file, 'w') as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

            try:
                with sqlite3.connect(self.DB_PATH) as conn:
                    # Update counters
                    conn.execute(f"""
                        UPDATE limits_{self.provider} SET
                            minute_requests = minute_requests + 1,
                            hour_requests = hour_requests + 1,
                            minute_tokens = minute_tokens + ?,
                            consecutive_limits = 0
                        WHERE id = 1
                    """, (tokens_used,))

            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def record_rate_limit(self):
        """Record a rate limit event."""
        if self.coordinated:
            self._record_rate_limit_coordinated()
        else:
            self._record_rate_limit_local()

    def _record_rate_limit_local(self):
        """Record rate limit for local mode."""
        with self._lock:
            self._consecutive_limits += 1
            self._last_backoff = self._calculate_backoff(self._consecutive_limits)

    def _record_rate_limit_coordinated(self):
        """Record rate limit for coordinated mode."""
        lock_file = self.DB_PATH.with_suffix('.lock')

        with open(lock_file, 'w') as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

            try:
                with sqlite3.connect(self.DB_PATH) as conn:
                    # Get current consecutive limits
                    row = conn.execute(f"SELECT consecutive_limits FROM limits_{self.provider} WHERE id = 1").fetchone()
                    consecutive = (row[0] if row else 0) + 1

                    # Update consecutive limits and last backoff
                    backoff = self._calculate_backoff(consecutive)
                    conn.execute(f"""
                        UPDATE limits_{self.provider} SET
                            consecutive_limits = ?,
                            last_backoff = ?
                        WHERE id = 1
                    """, (consecutive, backoff))

            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def get_wait_time(self) -> float:
        """Get recommended wait time before next request."""
        status = self.can_proceed()

        if status["allowed"]:
            return 0.0

        return status["wait_time"]

    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limiter statistics."""
        state = self._reset_windows_if_needed()

        return {
            "provider": self.provider,
            "coordinated": self.coordinated,
            "limits": {
                "requests_per_minute": self.limits.requests_per_minute,
                "requests_per_hour": self.limits.requests_per_hour,
                "tokens_per_minute": self.limits.tokens_per_minute,
            },
            "current": {
                "minute_requests": state["minute_requests"],
                "hour_requests": state["hour_requests"],
                "minute_tokens": state["minute_tokens"],
                "consecutive_limits": state["consecutive_limits"],
            },
            "utilization": {
                "minute_request_percent": (state["minute_requests"] / max(1, self.limits.requests_per_minute)) * 100,
                "hour_request_percent": (state["hour_requests"] / max(1, self.limits.requests_per_hour)) * 100,
                "minute_token_percent": (state["minute_tokens"] / max(1, self.limits.tokens_per_minute)) * 100,
            }
        }


# Global cache of rate limiters to avoid recreating
_limiter_cache: Dict[str, CoordinatedRateLimiter] = {}
_cache_lock = threading.Lock()


def get_rate_limiter(provider: str, coordinated: Optional[bool] = None) -> CoordinatedRateLimiter:
    """
    Get or create a rate limiter for a provider.

    Args:
        provider: Provider name (anthropic, openai, etc.)
        coordinated: Whether to use coordination (defaults to env var)

    Returns:
        CoordinatedRateLimiter instance
    """
    # Default coordinated mode from environment
    if coordinated is None:
        coordinated = os.getenv("HERMES_COORDINATED_RATE_LIMITING", "true").lower() in ("true", "1", "yes")

    cache_key = f"{provider}:{coordinated}"

    with _cache_lock:
        if cache_key not in _limiter_cache:
            _limiter_cache[cache_key] = CoordinatedRateLimiter(provider, coordinated)

        return _limiter_cache[cache_key]