from typing import Dict, Any, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import json
import numpy as np
from src.utils.logging_config import setup_logger

# 设置日志记录
logger = setup_logger('api')
_FETCH_FAILURE_CACHE: Dict[str, float] = {}
_FETCH_FAILURE_COOLDOWN_SECONDS = 180


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert common numeric-like values to float and normalize NaN."""
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_ak_symbol(symbol: str) -> str:
    if symbol.startswith(("5", "6", "9")):
        return f"sh{symbol}"
    return f"sz{symbol}"


def _standardize_price_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    column_mappings = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change_amount",
        "换手率": "turnover",
    }
    normalized = normalized.rename(columns=column_mappings)

    if source == "tencent":
        if "amount" in normalized.columns and "volume" not in normalized.columns:
            normalized["volume"] = pd.to_numeric(
                normalized["amount"], errors="coerce"
            ) * 100
        if "close" in normalized.columns and "amount" in normalized.columns:
            normalized["amount"] = pd.to_numeric(
                normalized["close"], errors="coerce"
            ) * pd.to_numeric(normalized["volume"], errors="coerce")

    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "change_amount",
        "turnover",
    ]
    for col in required_columns:
        if col not in normalized.columns:
            normalized[col] = np.nan

    normalized["date"] = pd.to_datetime(normalized["date"])
    for col in required_columns[1:]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    normalized = normalized.sort_values("date").reset_index(drop=True)
    normalized.attrs["data_source"] = source
    return normalized


def _fetch_with_fallback(
    symbol: str,
    fetchers: List[tuple[str, Callable[[], Any], int]],
    entity_name: str,
) -> tuple[Optional[Any], str, Optional[str]]:
    last_error = None
    for source_name, fetcher, timeout_seconds in fetchers:
        cache_key = f"{entity_name}:{symbol}:{source_name}"
        failed_at = _FETCH_FAILURE_CACHE.get(cache_key)
        if failed_at and (datetime.now().timestamp() - failed_at) < _FETCH_FAILURE_COOLDOWN_SECONDS:
            logger.info(
                f"跳过 {source_name} 获取 {symbol} 的 {entity_name}（近期失败冷却中）"
            )
            continue
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fetcher)
                result = future.result(timeout=timeout_seconds)
            if result is None:
                raise ValueError(f"{source_name} returned None")
            if isinstance(result, pd.DataFrame) and result.empty:
                raise ValueError(f"{source_name} returned empty dataframe")
            _FETCH_FAILURE_CACHE.pop(cache_key, None)
            logger.info(f"使用 {source_name} 获取 {symbol} 的 {entity_name}")
            return result, source_name, None
        except FuturesTimeoutError:
            last_error = f"{source_name} timed out after {timeout_seconds}s"
            _FETCH_FAILURE_CACHE[cache_key] = datetime.now().timestamp()
            logger.warning(
                f"{source_name} 获取 {symbol} 的 {entity_name} 超时 ({timeout_seconds}秒)"
            )
        except Exception as e:
            last_error = str(e)
            _FETCH_FAILURE_CACHE[cache_key] = datetime.now().timestamp()
            logger.warning(
                f"{source_name} 获取 {symbol} 的 {entity_name} 失败: {e}"
            )
    return None, "unavailable", last_error


def _extract_latest_numeric(df: pd.DataFrame, candidates: List[str]) -> Optional[float]:
    if df is None or df.empty:
        return None
    for col in df.columns:
        for candidate in candidates:
            if candidate in str(col):
                value = pd.to_numeric(df[col], errors="coerce").dropna()
                if not value.empty:
                    return float(value.iloc[0])
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if not series.empty:
            return float(series.iloc[0])
    return None


def get_northbound_flow(days: int = 5) -> Dict[str, Any]:
    """Get northbound capital flow signal from AKShare."""
    neutral = {
        "net_inflow_billion": 0.0,
        "trend": "neutral",
        "signal": "neutral",
        "data_available": False,
    }
    try:
        fetch_fn = getattr(ak, "stock_hsgt_north_net_flow_in_em", None)
        if fetch_fn is not None:
            df = fetch_fn(symbol="北上")
        else:
            hist_fn = getattr(ak, "stock_hsgt_hist_em", None)
            if hist_fn is None:
                return neutral
            df = hist_fn(symbol="北向资金")
        if df is None or df.empty:
            return neutral
        flow_cols = [c for c in df.columns if ("净流入" in str(c) or "净买额" in str(c))]
        if not flow_cols:
            return neutral
        col = flow_cols[0]
        recent = pd.to_numeric(df[col], errors="coerce").dropna().tail(days)
        if recent.empty:
            return neutral
        total_inflow_billion = float(recent.sum())
        if total_inflow_billion > 50:
            trend, signal = "strong_inflow", "bullish"
        elif total_inflow_billion > 10:
            trend, signal = "moderate_inflow", "bullish"
        elif total_inflow_billion > -10:
            trend, signal = "neutral", "neutral"
        elif total_inflow_billion > -50:
            trend, signal = "moderate_outflow", "bearish"
        else:
            trend, signal = "strong_outflow", "bearish"
        return {
            "net_inflow_billion": round(total_inflow_billion, 2),
            "trend": trend,
            "signal": signal,
            "data_available": True,
        }
    except Exception as e:
        logger.warning(f"获取北向资金失败: {e}")
        return neutral


def get_macro_indicators() -> Dict[str, Dict[str, Any]]:
    """Get core China macro indicators with graceful degradation."""
    indicators: Dict[str, Dict[str, Any]] = {}

    # Use lambdas so AttributeError is raised inside the try/except loop,
    # not at dict construction time (some akshare functions may not exist)
    fetch_map: Dict[str, Callable[[], pd.DataFrame]] = {
        "gdp_growth": lambda: ak.macro_china_gdp(),
        "cpi": lambda: ak.macro_china_cpi(),
        "pmi": lambda: ak.macro_china_pmi(),
        "lpr": lambda: ak.macro_china_lpr(),
        "m2_growth": lambda: ak.macro_china_m2_yearly(),
    }
    key_words = {
        "gdp_growth": ["同比增长", "增长", "GDP"],
        "cpi": ["同比增长", "CPI"],
        "pmi": ["PMI", "指数"],
        "lpr": ["1Y", "5Y", "LPR", "报价"],
        "m2_growth": ["同比增长", "M2", "增长"],
    }

    for metric, fetcher in fetch_map.items():
        try:
            df = fetcher()
            value = _extract_latest_numeric(df, key_words[metric])
            indicators[metric] = {
                "value": value,
                "data_available": value is not None,
                "source": "akshare",
            }
        except Exception as e:
            logger.warning(f"获取宏观指标 {metric} 失败: {e}")
            indicators[metric] = {
                "value": None,
                "data_available": False,
                "source": "akshare",
            }
    return indicators


def get_industry_news(industry: str, max_news: int = 20) -> List[Dict[str, Any]]:
    """Fetch industry-level news by keyword using the existing news_crawler."""
    if not industry:
        return []
    try:
        from src.tools.news_crawler import get_stock_news
        raw_news = get_stock_news(industry, max_news=max_news)
        if not raw_news:
            return []
        items: List[Dict[str, Any]] = []
        for n in raw_news[:max_news]:
            items.append(
                {
                    "title": str(n.get("title", "") or "").strip(),
                    "content": str(n.get("content", "") or "")[:400].strip(),
                    "publish_time": str(n.get("publish_time", "") or n.get("date", "") or ""),
                    "source": str(n.get("source", "") or "").strip(),
                    "url": str(n.get("url", "") or "").strip(),
                    "keyword": industry,
                }
            )
        return [x for x in items if x.get("title")]
    except Exception as e:
        logger.warning(f"获取行业新闻失败({industry}): {e}")
        return []


def calculate_beta(symbol: str, benchmark: str = "000300", period: int = 252) -> float:
    """Compute stock beta versus benchmark index returns."""
    try:
        stock_df = get_price_history(symbol)
        bench_df = get_price_history(benchmark)
        if stock_df is None or stock_df.empty or bench_df is None or bench_df.empty:
            return 1.0
        stock_returns = stock_df[["date", "close"]].copy()
        bench_returns = bench_df[["date", "close"]].copy()
        stock_returns["stock"] = stock_returns["close"].pct_change()
        bench_returns["bench"] = bench_returns["close"].pct_change()
        merged = stock_returns[["date", "stock"]].merge(
            bench_returns[["date", "bench"]], on="date", how="inner"
        ).dropna()
        merged = merged.tail(period)
        if len(merged) < 30:
            return 1.0
        variance = float(np.var(merged["bench"]))
        if variance <= 0:
            return 1.0
        covariance = float(np.cov(merged["stock"], merged["bench"])[0][1])
        beta = covariance / variance
        return float(max(0.3, min(beta, 2.5)))
    except Exception as e:
        logger.warning(f"计算 Beta 失败: {e}")
        return 1.0


def calculate_wacc(symbol: str, financial_metrics: Dict[str, Any], risk_free_rate: float = 0.03) -> float:
    """Compute dynamic WACC with conservative fallback."""
    try:
        beta = calculate_beta(symbol=symbol)
        market_premium = 0.06
        cost_of_equity = risk_free_rate + beta * market_premium

        debt_ratio = financial_metrics.get("debt_to_equity")
        debt_ratio = float(debt_ratio) if debt_ratio is not None else 0.4
        debt_ratio = max(0.0, min(debt_ratio, 1.0))

        cost_of_debt = risk_free_rate + 0.02
        tax_rate = 0.25
        wacc = cost_of_equity * (1 - debt_ratio) + cost_of_debt * debt_ratio * (1 - tax_rate)
        return float(max(0.08, min(wacc, 0.20)))
    except Exception as e:
        logger.warning(f"计算 WACC 失败: {e}")
        return 0.10


def _fetch_market_snapshot(symbol: str) -> Dict[str, Any]:
    normalized_symbol = _normalize_ak_symbol(symbol)
    snapshot: Dict[str, Any] = {
        "latest_price": None,
        "market_cap": None,
        "float_market_cap": None,
        "total_shares": None,
        "float_shares": None,
        "pe_ratio": None,
        "price_to_book": None,
        "data_source": "unavailable",
        "error": None,
    }

    def _spot_em():
        df = ak.stock_zh_a_spot_em()
        row = df[df["代码"] == symbol]
        if row.empty:
            raise ValueError(f"未找到 {symbol} 的东方财富实时行情数据")
        return row.iloc[0]

    fetchers: List[tuple[str, Callable[[], Any]]] = [
        ("eastmoney_realtime", _spot_em, 15),
        ("eastmoney_info", lambda: ak.stock_individual_info_em(symbol=symbol), 15),
        (
            "sina_daily",
            lambda: ak.stock_zh_a_daily(
                symbol=normalized_symbol,
                start_date=(datetime.now() - timedelta(days=20)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq",
            ),
            15,
        ),
        (
            "tencent_hist",
            lambda: ak.stock_zh_a_hist_tx(
                symbol=normalized_symbol,
                start_date=(datetime.now() - timedelta(days=20)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq",
            ),
            15,
        ),
    ]

    result, source_name, error = _fetch_with_fallback(
        symbol=symbol,
        fetchers=fetchers,
        entity_name="market snapshot",
    )
    snapshot["data_source"] = source_name
    snapshot["error"] = error

    if result is None:
        return snapshot

    if source_name == "eastmoney_realtime":
        stock_data = result
        snapshot.update(
            {
                "latest_price": _safe_float(stock_data.get("最新价")),
                "market_cap": _safe_float(stock_data.get("总市值")),
                "float_market_cap": _safe_float(stock_data.get("流通市值")),
                "pe_ratio": _safe_float(stock_data.get("市盈率-动态")),
                "price_to_book": _safe_float(stock_data.get("市净率")),
            }
        )
        return snapshot

    if source_name == "eastmoney_info":
        info_map = dict(zip(result["item"], result["value"]))
        latest_price = _safe_float(info_map.get("最新"))
        total_shares = _safe_float(info_map.get("总股本"))
        float_shares = _safe_float(info_map.get("流通股"))
        snapshot.update(
            {
                "latest_price": latest_price,
                "market_cap": _safe_float(info_map.get("总市值")),
                "float_market_cap": _safe_float(info_map.get("流通市值")),
                "total_shares": total_shares,
                "float_shares": float_shares,
            }
        )
        if snapshot["market_cap"] is None and latest_price is not None and total_shares is not None:
            snapshot["market_cap"] = latest_price * total_shares
        if (
            snapshot["float_market_cap"] is None
            and latest_price is not None
            and float_shares is not None
        ):
            snapshot["float_market_cap"] = latest_price * float_shares
        return snapshot

    standardized = _standardize_price_dataframe(result, source_name)
    latest_row = standardized.iloc[-1]
    snapshot["latest_price"] = _safe_float(latest_row.get("close"))
    if source_name == "sina" or source_name == "sina_daily":
        float_shares = None
        if "outstanding_share" in result.columns:
            float_shares = _safe_float(result.iloc[-1].get("outstanding_share"))
        snapshot["float_shares"] = float_shares
        if snapshot["latest_price"] is not None and float_shares is not None:
            snapshot["float_market_cap"] = snapshot["latest_price"] * float_shares
            snapshot["market_cap"] = snapshot["float_market_cap"]
    return snapshot


def get_financial_metrics(symbol: str) -> Dict[str, Any]:
    """获取财务指标数据"""
    logger.info(f"正在获取 {symbol} 的财务指标...")
    try:
        market_snapshot = _fetch_market_snapshot(symbol)
        logger.info(
            f"市场快照数据来源 {symbol}: {market_snapshot.get('data_source')}"
        )

        # 获取新浪财务指标
        logger.info("正在获取新浪财务指标...")
        current_year = datetime.now().year
        financial_data = ak.stock_financial_analysis_indicator(
            symbol=symbol, start_year=str(current_year-1))
        if financial_data is None or financial_data.empty:
            logger.warning("无可用财务指标数据")
            return [{
                "_status": "unavailable",
                "_error": "无可用财务指标数据",
                "_data_source": market_snapshot.get("data_source"),
            }]

        # 按日期排序并获取最新的数据
        financial_data['日期'] = pd.to_datetime(financial_data['日期'])
        financial_data = financial_data.sort_values('日期', ascending=False)
        latest_financial = financial_data.iloc[0] if not financial_data.empty else pd.Series(
        )
        logger.info(
            f"✓ 财务指标获取完成 ({len(financial_data)} 条记录)")
        logger.info(f"最新数据日期: {latest_financial.get('日期')}")

        # 获取利润表数据（用于计算 price_to_sales）
        logger.info("正在获取利润表...")
        try:
            income_statement = ak.stock_financial_report_sina(
                stock=_normalize_ak_symbol(symbol), symbol="利润表")
            if not income_statement.empty:
                latest_income = income_statement.iloc[0]
                logger.info("✓ 利润表获取完成")
            else:
                logger.warning("获取利润表失败")
                logger.error("未找到利润表数据")
                latest_income = pd.Series()
        except Exception as e:
            logger.warning("获取利润表失败")
            logger.error(f"获取利润表出错: {e}")
            latest_income = pd.Series()

        # 构建完整指标数据
        logger.info("正在构建指标...")
        try:
            def convert_percentage(value: float) -> Optional[float]:
                """将百分比值转换为小数"""
                try:
                    if value is None or pd.isna(value):
                        return None
                    return float(value) / 100.0
                except Exception:
                    return None

            latest_price = market_snapshot.get("latest_price")
            market_cap = market_snapshot.get("market_cap")
            float_market_cap = market_snapshot.get("float_market_cap")
            revenue = _safe_float(latest_income.get("营业总收入"))
            net_income = _safe_float(latest_income.get("净利润"))
            book_value_per_share = _safe_float(
                latest_financial.get("每股净资产_调整后(元)")
            )
            if book_value_per_share is None:
                book_value_per_share = _safe_float(
                    latest_financial.get("每股净资产_调整前(元)")
                )

            pe_ratio = market_snapshot.get("pe_ratio")
            if pe_ratio is None and market_cap is not None and net_income and net_income > 0:
                pe_ratio = market_cap / net_income

            price_to_book = market_snapshot.get("price_to_book")
            if (
                price_to_book is None
                and latest_price is not None
                and book_value_per_share
                and book_value_per_share > 0
            ):
                price_to_book = latest_price / book_value_per_share

            price_to_sales = None
            if market_cap is not None and revenue and revenue > 0:
                price_to_sales = market_cap / revenue

            all_metrics = {
                # 市场数据
                "market_cap": market_cap,
                "float_market_cap": float_market_cap,

                # 盈利数据
                "revenue": revenue,
                "net_income": net_income,
                "return_on_equity": convert_percentage(latest_financial.get("净资产收益率(%)", 0)),
                "net_margin": convert_percentage(latest_financial.get("销售净利率(%)", 0)),
                "operating_margin": convert_percentage(latest_financial.get("营业利润率(%)", 0)),

                # 增长指标
                "revenue_growth": convert_percentage(latest_financial.get("主营业务收入增长率(%)", 0)),
                "earnings_growth": convert_percentage(latest_financial.get("净利润增长率(%)", 0)),
                "book_value_growth": convert_percentage(latest_financial.get("净资产增长率(%)", 0)),

                # 财务健康指标
                "current_ratio": _safe_float(latest_financial.get("流动比率")),
                "debt_to_equity": convert_percentage(latest_financial.get("资产负债率(%)", 0)),
                "free_cash_flow_per_share": _safe_float(latest_financial.get("每股经营性现金流(元)")),
                "earnings_per_share": _safe_float(latest_financial.get("加权每股收益(元)")),

                # 估值比率
                "pe_ratio": pe_ratio,
                "price_to_book": price_to_book,
                "price_to_sales": price_to_sales,
            }

            # 只返回 agent 需要的指标
            agent_metrics = {
                # 盈利能力指标
                "return_on_equity": all_metrics["return_on_equity"],
                "net_margin": all_metrics["net_margin"],
                "operating_margin": all_metrics["operating_margin"],
                "market_cap": all_metrics["market_cap"],
                "float_market_cap": all_metrics["float_market_cap"],

                # 增长指标
                "revenue_growth": all_metrics["revenue_growth"],
                "earnings_growth": all_metrics["earnings_growth"],
                "book_value_growth": all_metrics["book_value_growth"],

                # 财务健康指标
                "current_ratio": all_metrics["current_ratio"],
                "debt_to_equity": all_metrics["debt_to_equity"],
                "free_cash_flow_per_share": all_metrics["free_cash_flow_per_share"],
                "earnings_per_share": all_metrics["earnings_per_share"],

                # 估值比率
                "pe_ratio": all_metrics["pe_ratio"],
                "price_to_book": all_metrics["price_to_book"],
                "price_to_sales": all_metrics["price_to_sales"],
                "_status": "ok",
                "_data_source": {
                    "market_snapshot": market_snapshot.get("data_source"),
                    "financial_indicator": "sina",
                    "income_statement": "sina",
                },
            }

            logger.info("✓ 指标构建成功")

            # 打印所有获取到的指标数据（用于调试）
            logger.debug("\n获取到的完整指标数据：")
            for key, value in all_metrics.items():
                logger.debug(f"{key}: {value}")

            logger.debug("\n传递给 agent 的指标数据：")
            for key, value in agent_metrics.items():
                logger.debug(f"{key}: {value}")

            return [agent_metrics]

        except Exception as e:
            logger.error(f"构建指标出错: {e}")
            return [{
                "_status": "unavailable",
                "_error": str(e),
                "_data_source": market_snapshot.get("data_source"),
            }]

    except Exception as e:
        logger.error(f"获取财务指标出错: {e}")
        return [{
            "_status": "unavailable",
            "_error": str(e),
            "_data_source": "unavailable",
        }]


def get_financial_statements(symbol: str) -> Dict[str, Any]:
    """获取财务报表数据"""
    logger.info(f"正在获取 {symbol} 的财务报表...")
    try:
        # 获取资产负债表数据
        logger.info("正在获取资产负债表...")
        try:
            balance_sheet = ak.stock_financial_report_sina(
                stock=_normalize_ak_symbol(symbol), symbol="资产负债表")
            if not balance_sheet.empty:
                latest_balance = balance_sheet.iloc[0]
                previous_balance = balance_sheet.iloc[1] if len(
                    balance_sheet) > 1 else balance_sheet.iloc[0]
                logger.info("✓ 资产负债表获取完成")
            else:
                logger.warning("获取资产负债表失败")
                logger.error("未找到资产负债表数据")
                latest_balance = pd.Series()
                previous_balance = pd.Series()
        except Exception as e:
            logger.warning("获取资产负债表失败")
            logger.error(f"获取资产负债表出错: {e}")
            latest_balance = pd.Series()
            previous_balance = pd.Series()

        # 获取利润表数据
        logger.info("正在获取利润表...")
        try:
            income_statement = ak.stock_financial_report_sina(
                stock=_normalize_ak_symbol(symbol), symbol="利润表")
            if not income_statement.empty:
                latest_income = income_statement.iloc[0]
                previous_income = income_statement.iloc[1] if len(
                    income_statement) > 1 else income_statement.iloc[0]
                logger.info("✓ 利润表获取完成")
            else:
                logger.warning("获取利润表失败")
                logger.error("未找到利润表数据")
                latest_income = pd.Series()
                previous_income = pd.Series()
        except Exception as e:
            logger.warning("获取利润表失败")
            logger.error(f"获取利润表出错: {e}")
            latest_income = pd.Series()
            previous_income = pd.Series()

        # 获取现金流量表数据
        logger.info("正在获取现金流量表...")
        try:
            cash_flow = ak.stock_financial_report_sina(
                stock=_normalize_ak_symbol(symbol), symbol="现金流量表")
            if not cash_flow.empty:
                latest_cash_flow = cash_flow.iloc[0]
                previous_cash_flow = cash_flow.iloc[1] if len(
                    cash_flow) > 1 else cash_flow.iloc[0]
                logger.info("✓ 现金流量表获取完成")
            else:
                logger.warning("获取现金流量表失败")
                logger.error("未找到现金流量数据")
                latest_cash_flow = pd.Series()
                previous_cash_flow = pd.Series()
        except Exception as e:
            logger.warning("获取现金流量表失败")
            logger.error(f"获取现金流量表出错: {e}")
            latest_cash_flow = pd.Series()
            previous_cash_flow = pd.Series()

        # 构建财务数据
        line_items = []
        try:
            # 处理最新期间数据
            current_item = {
                # 从利润表获取
                "net_income": float(latest_income.get("净利润", 0)),
                "operating_revenue": float(latest_income.get("营业总收入", 0)),
                "operating_profit": float(latest_income.get("营业利润", 0)),

                # 从资产负债表计算营运资金
                "working_capital": float(latest_balance.get("流动资产合计", 0)) - float(latest_balance.get("流动负债合计", 0)),

                # 从现金流量表获取
                "depreciation_and_amortization": float(latest_cash_flow.get("固定资产折旧、油气资产折耗、生产性生物资产折旧", 0)),
                "capital_expenditure": abs(float(latest_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0))),
                "free_cash_flow": float(latest_cash_flow.get("经营活动产生的现金流量净额", 0)) - abs(float(latest_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0)))
            }
            line_items.append(current_item)
            logger.info("✓ 最新期数据处理成功")

            # 处理上一期间数据
            previous_item = {
                "net_income": float(previous_income.get("净利润", 0)),
                "operating_revenue": float(previous_income.get("营业总收入", 0)),
                "operating_profit": float(previous_income.get("营业利润", 0)),
                "working_capital": float(previous_balance.get("流动资产合计", 0)) - float(previous_balance.get("流动负债合计", 0)),
                "depreciation_and_amortization": float(previous_cash_flow.get("固定资产折旧、油气资产折耗、生产性生物资产折旧", 0)),
                "capital_expenditure": abs(float(previous_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0))),
                "free_cash_flow": float(previous_cash_flow.get("经营活动产生的现金流量净额", 0)) - abs(float(previous_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0)))
            }
            line_items.append(previous_item)
            logger.info("✓ 上期数据处理成功")

        except Exception as e:
            logger.error(f"处理财务数据出错: {e}")
            default_item = {
                "net_income": 0,
                "operating_revenue": 0,
                "operating_profit": 0,
                "working_capital": 0,
                "depreciation_and_amortization": 0,
                "capital_expenditure": 0,
                "free_cash_flow": 0
            }
            line_items = [default_item, default_item]

        return line_items

    except Exception as e:
        logger.error(f"获取财务报表出错: {e}")
        default_item = {
            "net_income": 0,
            "operating_revenue": 0,
            "operating_profit": 0,
            "working_capital": 0,
            "depreciation_and_amortization": 0,
            "capital_expenditure": 0,
            "free_cash_flow": 0
        }
        return [default_item, default_item]


def get_market_data(
    symbol: str,
    price_history: Optional[pd.DataFrame] = None,
    market_cap: Optional[float] = None,
) -> Dict[str, Any]:
    """获取市场数据"""
    try:
        snapshot = {
            "market_cap": market_cap,
            "data_source": "financial_metrics" if market_cap is not None else "unavailable",
        }
        if market_cap is None:
            snapshot = _fetch_market_snapshot(symbol)
        if price_history is None:
            price_history = get_price_history(symbol)

        if price_history is None or price_history.empty:
            latest_volume = None
            avg_volume = None
            high_52w = None
            low_52w = None
        else:
            latest_volume = _safe_float(price_history["volume"].iloc[-1], 0.0)
            avg_volume = _safe_float(price_history["volume"].tail(20).mean(), 0.0)
            high_52w = _safe_float(price_history["high"].max())
            low_52w = _safe_float(price_history["low"].min())

        return {
            "market_cap": snapshot.get("market_cap"),
            "volume": latest_volume,
            "average_volume": avg_volume,
            "fifty_two_week_high": high_52w,
            "fifty_two_week_low": low_52w,
            "_data_source": {
                "snapshot": snapshot.get("data_source"),
                "price_history": getattr(price_history, "attrs", {}).get("data_source", "unavailable"),
            },
            "_status": "ok" if any(
                value is not None for value in [snapshot.get("market_cap"), latest_volume, high_52w, low_52w]
            ) else "unavailable",
        }

    except Exception as e:
        logger.error(f"获取市场数据出错: {e}")
        return {
            "_status": "unavailable",
            "_error": str(e),
        }


def get_price_history(symbol: str, start_date: str = None, end_date: str = None, adjust: str = "qfq") -> pd.DataFrame:
    """获取历史价格数据

    Args:
        symbol: 股票代码
        start_date: 开始日期，格式：YYYY-MM-DD，如果为None则默认获取过去一年的数据
        end_date: 结束日期，格式：YYYY-MM-DD，如果为None则使用昨天作为结束日期
        adjust: 复权类型，可选值：
               - "": 不复权
               - "qfq": 前复权（默认）
               - "hfq": 后复权

    Returns:
        包含以下列的DataFrame：
        - date: 日期
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价
        - volume: 成交量（手）
        - amount: 成交额（元）
        - amplitude: 振幅（%）
        - pct_change: 涨跌幅（%）
        - change_amount: 涨跌额（元）
        - turnover: 换手率（%）

        技术指标：
        - momentum_1m: 1个月动量
        - momentum_3m: 3个月动量
        - momentum_6m: 6个月动量
        - volume_momentum: 成交量动量
        - historical_volatility: 历史波动率
        - volatility_regime: 波动率区间
        - volatility_z_score: 波动率Z分数
        - atr_ratio: 真实波动幅度比率
        - hurst_exponent: 赫斯特指数
        - skewness: 偏度
        - kurtosis: 峰度
    """
    try:
        # 获取当前日期和昨天的日期
        current_date = datetime.now()
        yesterday = current_date - timedelta(days=1)

        # 如果没有提供日期，默认使用昨天作为结束日期
        if not end_date:
            end_date = yesterday  # 使用昨天作为结束日期
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            # 确保end_date不会超过昨天
            if end_date > yesterday:
                end_date = yesterday

        if not start_date:
            start_date = end_date - timedelta(days=365)  # 默认获取一年的数据
        else:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        logger.info(f"\n正在获取 {symbol} 的价格历史...")
        logger.info(f"开始日期: {start_date.strftime('%Y-%m-%d')}")
        logger.info(f"结束日期: {end_date.strftime('%Y-%m-%d')}")

        normalized_symbol = _normalize_ak_symbol(symbol)

        def get_and_process_data(start_date, end_date):
            """获取并处理数据，包括多数据源降级。"""
            fetchers = [
                (
                    "eastmoney",
                    lambda: ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date.strftime("%Y%m%d"),
                        end_date=end_date.strftime("%Y%m%d"),
                        adjust=adjust,
                    ),
                    15,
                ),
                (
                    "sina",
                    lambda: ak.stock_zh_a_daily(
                        symbol=normalized_symbol,
                        start_date=start_date.strftime("%Y%m%d"),
                        end_date=end_date.strftime("%Y%m%d"),
                        adjust=adjust,
                    ),
                    15,
                ),
                (
                    "tencent",
                    lambda: ak.stock_zh_a_hist_tx(
                        symbol=normalized_symbol,
                        start_date=start_date.strftime("%Y%m%d"),
                        end_date=end_date.strftime("%Y%m%d"),
                        adjust=adjust,
                    ),
                    15,
                ),
            ]

            result, source_name, _ = _fetch_with_fallback(
                symbol=symbol,
                fetchers=fetchers,
                entity_name="price history",
            )
            if result is None:
                return pd.DataFrame()
            return _standardize_price_dataframe(result, source_name)

        # 获取历史行情数据
        df = get_and_process_data(start_date, end_date)

        if df is None or df.empty:
            logger.warning(
                f"警告: 未找到 {symbol} 的价格历史数据")
            return pd.DataFrame()

        # 检查数据量是否足够
        min_required_days = 120  # 至少需要120个交易日的数据
        if len(df) < min_required_days:
            logger.warning(
                f"警告: 数据不足 ({len(df)} 天) 无法计算所有技术指标")
            logger.info("尝试获取更多数据...")

            # 扩大时间范围到2年
            start_date = end_date - timedelta(days=730)
            df = get_and_process_data(start_date, end_date)

            if len(df) < min_required_days:
                logger.warning(
                    f"警告: 即使扩展时间范围，数据仍不足 ({len(df)} 天)")

        # 计算动量指标
        df["momentum_1m"] = df["close"].pct_change(periods=20)  # 20个交易日约等于1个月
        df["momentum_3m"] = df["close"].pct_change(periods=60)  # 60个交易日约等于3个月
        df["momentum_6m"] = df["close"].pct_change(
            periods=120)  # 120个交易日约等于6个月

        # 计算成交量动量（相对于20日平均成交量的变化）
        df["volume_ma20"] = df["volume"].rolling(window=20).mean()
        df["volume_momentum"] = df["volume"] / df["volume_ma20"]

        # 计算波动率指标
        # 1. 历史波动率 (20日)
        returns = df["close"].pct_change()
        df["historical_volatility"] = returns.rolling(
            window=20).std() * np.sqrt(252)  # 年化

        # 2. 波动率区间 (相对于过去120天的波动率的位置)
        volatility_120d = returns.rolling(window=120).std() * np.sqrt(252)
        vol_min = volatility_120d.rolling(window=120).min()
        vol_max = volatility_120d.rolling(window=120).max()
        vol_range = vol_max - vol_min
        df["volatility_regime"] = np.where(
            vol_range > 0,
            (df["historical_volatility"] - vol_min) / vol_range,
            0  # 当范围为0时返回0
        )

        # 3. 波动率Z分数
        vol_mean = df["historical_volatility"].rolling(window=120).mean()
        vol_std = df["historical_volatility"].rolling(window=120).std()
        df["volatility_z_score"] = (
            df["historical_volatility"] - vol_mean) / vol_std

        # 4. ATR比率
        tr = pd.DataFrame()
        tr["h-l"] = df["high"] - df["low"]
        tr["h-pc"] = abs(df["high"] - df["close"].shift(1))
        tr["l-pc"] = abs(df["low"] - df["close"].shift(1))
        tr["tr"] = tr[["h-l", "h-pc", "l-pc"]].max(axis=1)
        df["atr"] = tr["tr"].rolling(window=14).mean()
        df["atr_ratio"] = df["atr"] / df["close"]

        # 计算统计套利指标
        # 1. 赫斯特指数 (使用过去120天的数据)
        def calculate_hurst(series):
            """
            计算Hurst指数。

            Args:
                series: 价格序列

            Returns:
                float: Hurst指数，或在计算失败时返回np.nan
            """
            try:
                series = series.dropna()
                if len(series) < 30:  # 降低最小数据点要求
                    return np.nan

                # 使用对数收益率
                log_returns = np.log(series / series.shift(1)).dropna()
                if len(log_returns) < 30:  # 降低最小数据点要求
                    return np.nan

                # 使用更小的lag范围
                # 减少lag范围到2-10天
                lags = range(2, min(11, len(log_returns) // 4))

                # 计算每个lag的标准差
                tau = []
                for lag in lags:
                    # 计算滚动标准差
                    std = log_returns.rolling(window=lag).std().dropna()
                    if len(std) > 0:
                        tau.append(np.mean(std))

                # 基本的数值检查
                if len(tau) < 3:  # 进一步降低最小要求
                    return np.nan

                # 使用对数回归
                lags_log = np.log(list(lags))
                tau_log = np.log(tau)

                # 计算回归系数
                reg = np.polyfit(lags_log, tau_log, 1)
                hurst = reg[0] / 2.0

                # 只保留基本的数值检查
                if np.isnan(hurst) or np.isinf(hurst):
                    return np.nan

                return hurst

            except Exception as e:
                return np.nan

        # 使用对数收益率计算Hurst指数
        log_returns = np.log(df["close"] / df["close"].shift(1))
        df["hurst_exponent"] = log_returns.rolling(
            window=120,
            min_periods=60  # 要求至少60个数据点
        ).apply(calculate_hurst)

        # 2. 偏度 (20日)
        df["skewness"] = returns.rolling(window=20).skew()

        # 3. 峰度 (20日)
        df["kurtosis"] = returns.rolling(window=20).kurt()

        # 按日期升序排序
        df = df.sort_values("date")

        # 重置索引
        df = df.reset_index(drop=True)

        logger.info(
            f"成功获取价格历史数据 ({len(df)} 条记录)")
        logger.info(f"价格历史数据来源 {symbol}: {df.attrs.get('data_source', 'unknown')}")

        # 检查并报告NaN值
        nan_columns = df.isna().sum()
        if nan_columns.any():
            logger.warning(
                "\n警告: 以下指标包含 NaN 值:")
            for col, nan_count in nan_columns[nan_columns > 0].items():
                logger.warning(f"- {col}: {nan_count} 条记录")

        return df

    except Exception as e:
        logger.error(f"获取价格历史出错: {e}")
        return pd.DataFrame()


def prices_to_df(prices):
    """Convert price data to DataFrame with standardized column names"""
    try:
        df = pd.DataFrame(prices)

        # 标准化列名映射
        column_mapping = {
            '收盘': 'close',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'change_percent',
            '涨跌额': 'change_amount',
            '换手率': 'turnover_rate'
        }

        # 重命名列
        for cn, en in column_mapping.items():
            if cn in df.columns:
                df[en] = df[cn]

        # 确保必要的列存在
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0  # 使用0填充缺失的必要列

        return df
    except Exception as e:
        logger.error(f"转换价格数据出错: {str(e)}")
        # 返回一个包含必要列的空DataFrame
        return pd.DataFrame(columns=['close', 'open', 'high', 'low', 'volume'])


def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """获取股票价格数据

    Args:
        ticker: 股票代码
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD

    Returns:
        包含价格数据的DataFrame
    """
    return get_price_history(ticker, start_date, end_date)
