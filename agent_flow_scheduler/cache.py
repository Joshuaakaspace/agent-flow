"""
cache.py
--------
SpeculativePrefillCache: warms KV-cache entries for predicted next-step agents.

Core idea:
  When agent A is executing, we know (from the workflow graph) that agent B
  will run next. We proactively pre-compute B's system-prompt prefix tokens
  so that when B's request arrives, the KV cache is already warm.

This mirrors the "proactive prefix caching" optimization in Pythia (arxiv 2604.25899).
In a production system, this hooks into the LLM backend's prefix-cache API
(e.g., vLLM's prefix caching, TensorRT-LLM's KV cache manager).

Here we implement a lightweight in-process simulation that tracks hit/miss rates
and estimated token savings.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CacheEntry:
    prefix_hash: str
    prefix_text: str
    token_count: int              # Estimated tokens in prefix
    created_at: float = field(default_factory=time.monotonic)
    last_hit_at: Optional[float] = None
    hit_count: int = 0
    evicted: bool = False

    def touch(self) -> None:
        self.last_hit_at = time.monotonic()
        self.hit_count += 1

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at


class SpeculativePrefillCache:
    """
    In-process simulation of a speculative KV-cache warmer.

    The cache stores system-prompt prefixes for upcoming agents.
    When an agent's request arrives, we check if the prefix is already
    in-cache (a "hit") and record the token savings.

    In production, replace warm_prefix() / check_hit() with calls to
    the actual LLM backend's prefix cache management API.
    """

    def __init__(self, max_entries: int = 64, ttl_seconds: float = 30.0):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}

        # Stats
        self.total_warms = 0
        self.total_hits = 0
        self.total_misses = 0
        self.total_tokens_saved = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def warm(self, prefix_text: str, estimated_token_count: int = 0) -> CacheEntry:
        """
        Proactively warm the cache for prefix_text.
        In production: triggers async KV-cache pre-computation on the GPU.
        """
        if not prefix_text:
            return None

        key = self._hash(prefix_text)
        if key in self._cache and not self._cache[key].evicted:
            return self._cache[key]

        # Evict stale entries if at capacity
        if len(self._cache) >= self.max_entries:
            self._evict_lru()

        if estimated_token_count == 0:
            # Rough estimate: ~1.3 tokens per word
            estimated_token_count = max(1, int(len(prefix_text.split()) * 1.3))

        entry = CacheEntry(
            prefix_hash=key,
            prefix_text=prefix_text,
            token_count=estimated_token_count,
        )
        self._cache[key] = entry
        self.total_warms += 1
        return entry

    def check_hit(self, prefix_text: str) -> Tuple[bool, Optional[CacheEntry]]:
        """
        Check if prefix_text is in cache. Returns (hit: bool, entry: CacheEntry|None).
        Updates hit/miss counters.
        """
        key = self._hash(prefix_text)
        entry = self._cache.get(key)

        if entry is None or entry.evicted:
            self.total_misses += 1
            return False, None

        if entry.age_seconds > self.ttl_seconds:
            entry.evicted = True
            self.total_misses += 1
            return False, None

        entry.touch()
        self.total_hits += 1
        self.total_tokens_saved += entry.token_count
        return True, entry

    def invalidate(self, prefix_text: str) -> None:
        key = self._hash(prefix_text)
        if key in self._cache:
            self._cache[key].evicted = True

    def clear(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------
    # Stats & Reporting
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    def active_entries(self) -> List[CacheEntry]:
        now = time.monotonic()
        return [
            e for e in self._cache.values()
            if not e.evicted and (now - e.created_at) <= self.ttl_seconds
        ]

    def stats(self) -> Dict:
        return {
            "entries_active": len(self.active_entries()),
            "total_warms": self.total_warms,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "tokens_saved": self.total_tokens_saved,
        }

    def report(self) -> str:
        s = self.stats()
        return (
            f"SpeculativePrefillCache | "
            f"active={s['entries_active']} | "
            f"hit_rate={s['hit_rate']} | "
            f"tokens_saved={s['tokens_saved']:,}"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _evict_lru(self) -> None:
        """Evict the least-recently-used entry."""
        active = self.active_entries()
        if not active:
            return
        lru = min(
            active,
            key=lambda e: e.last_hit_at if e.last_hit_at else e.created_at,
        )
        lru.evicted = True
