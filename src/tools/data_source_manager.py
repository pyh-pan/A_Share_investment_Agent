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
            logger.info(f"缓存命中: {cache_key} (TTL={cache_ttl_hours}h)")
            return cached

        logger.info(f"缓存未命中: {cache_key} — 尝试 {len(fetchers)} 个数据源")
        for idx, (source_name, fetcher) in enumerate(zip(source_names, fetchers)):
            try:
                value = fetcher()
                if value is None:
                    logger.info(f"数据源 {source_name} 返回 None: {cache_key} (第{idx+1}/{len(fetchers)}个)")
                    continue
                self.cache.set(cache_key, value, ttl_hours=cache_ttl_hours)
                logger.info(f"数据源 {source_name} 获取成功: {cache_key} (TTL={cache_ttl_hours}h)")
                return value
            except Exception as exc:
                if idx < len(fetchers) - 1:
                    logger.warning(f"数据源 {source_name} 获取失败，降级到下一个: {exc}")
                else:
                    logger.error(f"数据源 {source_name} 获取失败（已是最后备选）: {exc}")
        logger.error(f"所有数据源均失败: {cache_key}")
        return None
