from __future__ import annotations

import base64
import json
import pickle
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

from src.utils.logging_config import setup_logger

logger = setup_logger("db_cache")


class SimpleCache:
    """Small SQLite-backed cache for API responses.

    Values are stored as JSON when possible and fall back to pickle for
    pandas/numpy objects returned by market data fetchers.
    """

    def __init__(self, db_path: Union[str, Path] = "src/data/cache.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    encoding TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )

    def get(self, cache_key: str, max_age_hours: int = 24) -> Optional[Any]:
        if max_age_hours <= 0:
            return None

        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT payload, encoding, created_at, expires_at
                    FROM cache_entries
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                ).fetchone()
        except Exception as exc:
            logger.warning(f"读取缓存失败 {cache_key}: {exc}")
            return None

        if row is None:
            return None

        payload, encoding, created_at, expires_at = row
        now = datetime.utcnow()
        try:
            created = datetime.fromisoformat(created_at)
            expires = datetime.fromisoformat(expires_at)
        except ValueError:
            self.delete(cache_key)
            return None

        if now >= expires or now - created > timedelta(hours=max_age_hours):
            self.delete(cache_key)
            return None

        try:
            if encoding == "json":
                return json.loads(payload)
            if encoding == "pickle":
                return pickle.loads(base64.b64decode(payload.encode("ascii")))
        except Exception as exc:
            logger.warning(f"反序列化缓存失败 {cache_key}: {exc}")
            self.delete(cache_key)
            return None

        self.delete(cache_key)
        return None

    def set(self, cache_key: str, value: Any, ttl_hours: int = 24) -> bool:
        try:
            payload, encoding = self._serialize(value)
            now = datetime.utcnow()
            expires = now + timedelta(hours=max(ttl_hours, 0))
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries
                        (cache_key, payload, encoding, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        payload,
                        encoding,
                        now.isoformat(),
                        expires.isoformat(),
                    ),
                )
            return True
        except Exception as exc:
            logger.warning(f"写入缓存失败 {cache_key}: {exc}")
            return False

    def delete(self, cache_key: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM cache_entries WHERE cache_key = ?",
                    (cache_key,),
                )
        except Exception as exc:
            logger.debug(f"删除缓存失败 {cache_key}: {exc}")

    @staticmethod
    def _serialize(value: Any) -> tuple[str, str]:
        try:
            return json.dumps(value, ensure_ascii=False), "json"
        except (TypeError, ValueError):
            payload = base64.b64encode(pickle.dumps(value)).decode("ascii")
            return payload, "pickle"
