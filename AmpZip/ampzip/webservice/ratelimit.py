"""The per-email rate limiter, an in-memory sliding window.

Job submissions and numerical checks share one per-email quota, independent of
the global job queue limit. Counts are held in memory within one process.
Multiple server processes would need a shared store such as Redis.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from .config import CONFIG


class RateLimiter:
    def __init__(self, per_min: int) -> None:
        self._per_min = per_min
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        window_start = now - 60.0
        with self._lock:
            dq = self._hits[key]
            while dq and dq[0] < window_start:
                dq.popleft()
            if len(dq) >= self._per_min:
                return False
            dq.append(now)
            return True


LIMITER = RateLimiter(CONFIG.rate_per_min)
