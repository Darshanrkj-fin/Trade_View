from __future__ import annotations

import time
from typing import Any


class TTLCache:
    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        record = self._store.get(key)
        if record is None:
            return None
        expires_at, value = record
        if time.time() >= expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> Any:
        self._store[key] = (time.time() + self.ttl_seconds, value)
        return value

    def clear(self) -> None:
        self._store.clear()
