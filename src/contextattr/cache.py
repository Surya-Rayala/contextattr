"""The cache: every model answer stored once, under a key made from its request (docs/interface.md §7).

One SQLite file per run, or memory when no path is given. A run that stops resumes from it, another method on the
same masks reuses its answers, and nothing is asked of the model twice. The first answer stored for a key stays.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable, Mapping

_CHUNK = 500  # keys per SELECT, under SQLite's limit on query parameters


class Cache:
    """Answers by key. `key(request)` makes the key from a JSON-able request; answers are JSON-able values."""

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = None if path is None else os.fspath(path)
        if self.path is not None and os.path.dirname(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._db = sqlite3.connect(":memory:" if self.path is None else self.path)
        self._db.execute("CREATE TABLE IF NOT EXISTS answers "
                         "(key TEXT PRIMARY KEY, kind TEXT NOT NULL, answer TEXT NOT NULL)")
        self._db.commit()

    @staticmethod
    def key(request: Mapping) -> str:
        """sha256 of the request as canonical JSON: the same request always gives the same key."""
        text = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(self, keys: Iterable[str]) -> dict[str, object]:
        """The stored answers among `keys`."""
        keys = list(dict.fromkeys(keys))
        found = {}
        for i in range(0, len(keys), _CHUNK):
            chunk = keys[i:i + _CHUNK]
            rows = self._db.execute(f"SELECT key, answer FROM answers WHERE key IN ({','.join('?' * len(chunk))})",
                                    chunk)
            found.update((k, json.loads(a)) for k, a in rows)
        return found

    def put_many(self, kind: str, answers: Mapping[str, object]) -> None:
        """Store answers and commit at once; a key already stored keeps its first answer."""
        self._db.executemany("INSERT OR IGNORE INTO answers (key, kind, answer) VALUES (?, ?, ?)",
                             [(k, kind, json.dumps(a, ensure_ascii=False, allow_nan=False))
                              for k, a in answers.items()])
        self._db.commit()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Cache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
