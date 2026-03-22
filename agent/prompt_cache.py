"""
Prompt Caching System for LLM Token Efficiency

Tracks static prompt prefixes with deterministic hashes to optimize
LLM API costs through intelligent prefix caching.

Features:
- SQLite-based persistent storage
- Deterministic SHA-256 hashing of prefixes
- Hit/miss tracking and token savings metrics
- Cross-session prefix reuse
- Thread-safe operations

Usage:
    from agent.prompt_cache import PromptCache

    cache = PromptCache()

    # Check if prompt has cached prefix
    match = cache.find_prefix_match(full_prompt)
    if match:
        # Use cached prefix - saves tokens!
        print(f"Cache hit: {match['token_count']} tokens saved")

    # Record cache result
    cache.record_cache_event(prefix_hash, hit=True, tokens_saved=500)
"""

import hashlib
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    from hermes.agent.toon import to_toon
except ImportError:
    # Fallback for standalone usage
    def to_toon(data):
        return str(data)


@dataclass
class PrefixMatch:
    """Result of a prefix cache lookup."""
    prefix_hash: str
    token_count: int
    profile: Optional[str] = None
    hit_count: int = 0
    matched: bool = False


class PromptCache:
    """
    SQLite-based prompt prefix caching for LLM efficiency.

    Tracks static prompt prefixes with deterministic hashes to detect
    cacheable portions of prompts and measure token savings.
    """

    DB_PATH = Path.home() / ".hermes" / "cache" / "prompt_cache.db"

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize prompt cache with SQLite storage."""
        self.db_path = db_path or self.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database schema."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS prefix_cache (
                    prefix_hash TEXT PRIMARY KEY,
                    prefix_content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    profile TEXT,
                    created_at REAL DEFAULT (julianday('now')),
                    last_used REAL,
                    hit_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS cache_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL DEFAULT (julianday('now')),
                    prefix_hash TEXT,
                    event_type TEXT NOT NULL,  -- 'hit', 'miss', 'register'
                    tokens_saved INTEGER DEFAULT 0,
                    profile TEXT,
                    FOREIGN KEY (prefix_hash) REFERENCES prefix_cache(prefix_hash)
                );

                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                    ON cache_events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_prefix
                    ON cache_events(prefix_hash);
            """)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~4 chars per token."""
        return max(1, len(text) // 4)

    def _hash_prefix(self, prefix: str) -> str:
        """Generate deterministic SHA-256 hash for prefix."""
        return hashlib.sha256(prefix.encode('utf-8')).hexdigest()[:16]

    def register_prefix(self, prefix: str, profile: Optional[str] = None) -> str:
        """
        Register a prefix for caching.

        Args:
            prefix: The static prefix content
            profile: Optional profile name (coder, researcher, etc.)

        Returns:
            The prefix hash
        """
        prefix_hash = self._hash_prefix(prefix)
        token_count = self._estimate_tokens(prefix)

        with self._lock, sqlite3.connect(self.db_path) as conn:
            # Check if exists
            existing = conn.execute(
                "SELECT prefix_hash FROM prefix_cache WHERE prefix_hash = ?",
                (prefix_hash,)
            ).fetchone()

            if not existing:
                # Insert new prefix
                conn.execute(
                    """INSERT INTO prefix_cache
                       (prefix_hash, prefix_content, token_count, profile)
                       VALUES (?, ?, ?, ?)""",
                    (prefix_hash, prefix, token_count, profile)
                )

                # Record registration event
                conn.execute(
                    """INSERT INTO cache_events
                       (prefix_hash, event_type, profile)
                       VALUES (?, 'register', ?)""",
                    (prefix_hash, profile)
                )

        return prefix_hash

    def find_prefix_match(self, prompt: str) -> Optional[PrefixMatch]:
        """
        Check if prompt starts with a known cached prefix.

        Args:
            prompt: Full prompt to check

        Returns:
            PrefixMatch if found, None otherwise
        """
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get all prefixes sorted by length (longest first)
            prefixes = conn.execute(
                """SELECT prefix_hash, prefix_content, token_count, profile, hit_count
                   FROM prefix_cache
                   ORDER BY LENGTH(prefix_content) DESC"""
            ).fetchall()

            for row in prefixes:
                prefix_content = row["prefix_content"]
                if prompt.startswith(prefix_content):
                    return PrefixMatch(
                        prefix_hash=row["prefix_hash"],
                        token_count=row["token_count"],
                        profile=row["profile"],
                        hit_count=row["hit_count"],
                        matched=True
                    )

        return None

    def record_cache_event(
        self,
        prefix_hash: str,
        hit: bool,
        tokens_saved: int = 0,
        profile: Optional[str] = None
    ):
        """
        Record a cache hit or miss event.

        Args:
            prefix_hash: The prefix hash
            hit: True for cache hit, False for miss
            tokens_saved: Estimated tokens saved (for hits)
            profile: Optional profile name
        """
        event_type = "hit" if hit else "miss"

        with self._lock, sqlite3.connect(self.db_path) as conn:
            # Record event
            conn.execute(
                """INSERT INTO cache_events
                   (prefix_hash, event_type, tokens_saved, profile)
                   VALUES (?, ?, ?, ?)""",
                (prefix_hash, event_type, tokens_saved, profile)
            )

            # Update hit count and last_used for hits
            if hit:
                conn.execute(
                    """UPDATE prefix_cache
                       SET last_used = julianday('now'),
                           hit_count = hit_count + 1
                       WHERE prefix_hash = ?""",
                    (prefix_hash,)
                )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            # Total prefixes
            total_prefixes = conn.execute(
                "SELECT COUNT(*) as count FROM prefix_cache"
            ).fetchone()[0]

            # Total hits and misses
            hits = conn.execute(
                "SELECT COUNT(*) as count FROM cache_events WHERE event_type = 'hit'"
            ).fetchone()[0]

            misses = conn.execute(
                "SELECT COUNT(*) as count FROM cache_events WHERE event_type = 'miss'"
            ).fetchone()[0]

            # Total tokens saved
            tokens_saved = conn.execute(
                "SELECT COALESCE(SUM(tokens_saved), 0) as total FROM cache_events WHERE event_type = 'hit'"
            ).fetchone()[0]

            # Hit rate
            total_attempts = hits + misses
            hit_rate = hits / max(1, total_attempts)

            # By profile
            by_profile = conn.execute(
                """SELECT
                     profile,
                     COUNT(*) as hits,
                     SUM(tokens_saved) as saved
                   FROM cache_events
                   WHERE event_type = 'hit'
                   GROUP BY profile"""
            ).fetchall()

            profile_stats = {}
            for row in by_profile:
                profile = row[0] or "unknown"
                profile_stats[profile] = {
                    "hits": row[1],
                    "tokens_saved": row[2] or 0
                }

            # Top prefixes
            top_prefixes = conn.execute(
                """SELECT prefix_hash, hit_count, token_count
                   FROM prefix_cache
                   ORDER BY hit_count DESC
                   LIMIT 10"""
            ).fetchall()

            return {
                "total_prefixes": total_prefixes,
                "total_hits": hits,
                "total_misses": misses,
                "total_attempts": total_attempts,
                "hit_rate": round(hit_rate, 3),
                "total_tokens_saved": tokens_saved,
                "by_profile": profile_stats,
                "top_prefixes": [
                    {
                        "hash": row[0],
                        "hits": row[1],
                        "tokens": row[2]
                    }
                    for row in top_prefixes
                ]
            }

    def list_prefixes(self, profile: Optional[str] = None) -> List[Dict[str, Any]]:
        """List registered prefixes, optionally filtered by profile."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if profile:
                rows = conn.execute(
                    """SELECT prefix_hash, token_count, hit_count, created_at, profile
                       FROM prefix_cache WHERE profile = ?
                       ORDER BY hit_count DESC""",
                    (profile,)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT prefix_hash, token_count, hit_count, created_at, profile
                       FROM prefix_cache
                       ORDER BY hit_count DESC"""
                ).fetchall()

            return [
                {
                    "hash": row["prefix_hash"],
                    "tokens": row["token_count"],
                    "hits": row["hit_count"],
                    "profile": row["profile"],
                    "created": row["created_at"]
                }
                for row in rows
            ]

    def clear_cache(self):
        """Clear all cached prefixes and events."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache_events")
            conn.execute("DELETE FROM prefix_cache")

    def export_stats_toon(self) -> str:
        """Export cache statistics in TOON format."""
        stats = self.get_cache_stats()
        return to_toon({"cache_stats": stats})