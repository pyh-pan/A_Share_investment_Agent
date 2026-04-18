from __future__ import annotations

from typing import Optional

import akshare as ak

from src.tools.data_source_manager import DataSourceManager
from src.utils.logging_config import setup_logger

logger = setup_logger("industry_service")

_CODE_PREFIX_INDUSTRY = {
    "6013": "银行",
    "6012": "银行",
    "6000": "银行",
    "000001": "银行",
    "000002": "房地产",
    "6016": "非银金融",
    "6008": "非银金融",
    "300": "计算机",
}


class IndustryService:
    """Get stock industry with fallback: Tushare -> AKShare -> code prefix."""

    def __init__(self) -> None:
        self.dsm = DataSourceManager()

    @staticmethod
    def _to_ts_code(symbol: str) -> str:
        if symbol.startswith(("5", "6", "9")):
            return f"{symbol}.SH"
        return f"{symbol}.SZ"

    def get_industry(self, symbol: str) -> str:
        cache_key = f"industry:{symbol}"

        def _from_tushare() -> Optional[str]:
            try:
                import tushare as ts

                pro = ts.pro_api()
                df = pro.stock_basic(ts_code=self._to_ts_code(symbol), fields="industry")
                if df is not None and not df.empty:
                    value = str(df.iloc[0].get("industry") or "").strip()
                    return value or None
            except Exception as exc:
                logger.debug(f"Tushare 行业获取失败: {exc}")
            return None

        def _from_akshare() -> Optional[str]:
            try:
                df = ak.stock_individual_info_em(symbol=symbol)
                if df is None or df.empty:
                    return None
                rows = df[df["item"] == "行业"]
                if rows.empty:
                    return None
                value = str(rows.iloc[0].get("value") or "").strip()
                return value or None
            except Exception as exc:
                logger.debug(f"AKShare 行业获取失败: {exc}")
            return None

        industry = self.dsm.fetch_with_fallback(
            cache_key=cache_key,
            fetchers=[_from_tushare, _from_akshare],
            source_names=["tushare", "akshare"],
            cache_ttl_hours=24 * 7,
        )
        if isinstance(industry, str) and industry:
            return industry

        for prefix, guess in _CODE_PREFIX_INDUSTRY.items():
            if symbol.startswith(prefix):
                return guess
        return "default"
