"""Small persistent TTL cache backed by SQLite.

Used for HTTP responses so repeated agent calls (common: the model re-asks for
the same UniProt entry several times in one session) cost one network round trip.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


def make_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class TTLCache:
    def __init__(self, path: Path | str | None, ttl_s: int):
        """``path=None`` keeps the cache in memory (used by tests)."""
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path) if path is not None else ":memory:", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT NOT NULL, exp REAL NOT NULL)")
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._db.execute("SELECT v, exp FROM cache WHERE k = ?", (key,)).fetchone()
            if row is None or row[1] < time.time():
                self.misses += 1
                return None
            self.hits += 1
            return json.loads(row[0])

    def set(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        exp = time.time() + (self.ttl_s if ttl_s is None else ttl_s)
        with self._lock:
            self._db.execute("REPLACE INTO cache (k, v, exp) VALUES (?, ?, ?)", (key, json.dumps(value), exp))
            self._db.commit()

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM cache")
            self._db.commit()
