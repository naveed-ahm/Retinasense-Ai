"""Simple sliding-window rate limiter.

Process-local by default (adequate for single-worker deployments and tests).
When REDIS_ENABLED and a Redis client are available the limiter can be
swapped for a distributed one without changing call sites.
"""
import threading
import time
from collections import defaultdict, deque

_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)

_MAX_HISTORY = 2000


def check_rate_limit(key: str, limit: int, window_seconds: int) -> bool:
    """Return True when the key is allowed, False when it is rate-limited."""
    now = time.monotonic()
    with _lock:
        bucket = _hits[key]
        while bucket and now - bucket[0] > window_seconds:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        if len(_hits) > _MAX_HISTORY:
            for k in list(_hits.keys()):
                if k != key and not _hits[k]:
                    del _hits[k]
        return True


def clear_rate_limit(key: str) -> None:
    with _lock:
        _hits.pop(key, None)
