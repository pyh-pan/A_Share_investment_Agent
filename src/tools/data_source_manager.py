from __future__ import annotations

from typing import Any, Callable, List, Optional

from src.tools.cache.db_cache import SimpleCache
from src.utils.logging_config import setup_logger

logger = setup_logger("data_source_manager")


class DataSourceManager:
    """Simple multi-source fallback manager with local cache."""

    def __init__(self) -> None:
        self.cache = SimpleCache()

    def fetch_with_fallback(
        self,
        cache_key: str,
        fetchers: List[Callable[[], Any]],
        source_names: List[str],
        cache_ttl_hours: int = 24,
    ) -> Optional[Any]:
        cached = self.cache.get(cache_key, max_age_hours=cache_ttl_hours)
        if cached is not None:
            logger.info(f"缓存命中: {cache_key}")
            return cached

        for source_name, fetcher in zip(source_names, fetchers):
            try:
                value = fetcher()
                if value is None:
                    continue
                self.cache.set(cache_key, value, ttl_hours=cache_ttl_hours)
                logger.info(f"数据源 {source_name} 获取成功: {cache_key}")
                return value
            except Exception as exc:
                logger.warning(f"数据源 {source_name} 获取失败: {exc}")
        return None
