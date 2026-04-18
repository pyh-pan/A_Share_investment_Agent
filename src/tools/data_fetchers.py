from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("data_fetchers")


def _to_ts_code(symbol: str) -> str:
    if symbol.startswith(("5", "6", "9")):
        return f"{symbol}.SH"
    return f"{symbol}.SZ"


def _tushare_pro():
    import tushare as ts

    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        return None
    ts.set_token(token)
    return ts.pro_api()


def fetch_price_history_tushare(
    symbol: str, start_date: str, end_date: str, adjust: str = "qfq"
) -> Optional[List[Dict[str, Any]]]:
    pro = _tushare_pro()
    if pro is None:
        return None
    try:
        ts_code = _to_ts_code(symbol)
        start_fmt = start_date.replace("-", "") if "-" in start_date else start_date
        end_fmt = end_date.replace("-", "") if "-" in end_date else end_date

        adj = "qfq" if adjust == "qfq" else ("hfq" if adjust == "hfq" else None)
        df = pro.daily(
            ts_code=ts_code, start_date=start_fmt, end_date=end_fmt
        )
        if df is None or df.empty:
            return None

        if adj:
            try:
                adj_factor = pro.adj_factor(ts_code=ts_code, start_date=start_fmt, end_date=end_fmt)
                if adj_factor is not None and not adj_factor.empty:
                    df = df.merge(adj_factor[["trade_date", "adj_factor"]], on="trade_date", how="left")
                    latest_factor = df["adj_factor"].iloc[0] if not df["adj_factor"].isna().all() else 1.0
                    for col in ["open", "high", "low", "close"]:
                        df[col] = df[col] * df["adj_factor"] / latest_factor
            except Exception:
                pass

        df = df.sort_values("trade_date")

        records = []
        for _, row in df.iterrows():
            pct_change = _safe_float_val(row.get("pct_chg"))
            pre_close = _safe_float_val(row.get("pre_close"))
            close = _safe_float_val(row.get("close"))
            change_amount = close - pre_close if close is not None and pre_close is not None else None
            high = _safe_float_val(row.get("high"))
            low = _safe_float_val(row.get("low"))
            pre_close_safe = pre_close if pre_close and pre_close != 0 else None
            amplitude = (high - low) / pre_close_safe * 100 if high and low and pre_close_safe else None

            records.append(
                {
                    "日期": row.get("trade_date", ""),
                    "开盘": _safe_float_val(row.get("open")),
                    "最高": high,
                    "最低": low,
                    "收盘": close,
                    "成交量": _safe_float_val(row.get("vol")),
                    "成交额": _safe_float_val(row.get("amount")),
                    "振幅": amplitude,
                    "涨跌幅": pct_change,
                    "涨跌额": change_amount,
                    "换手率": _safe_float_val(row.get("turnover_rate")),
                }
            )
        return records
    except Exception as exc:
        logger.warning(f"Tushare 价格历史获取失败: {exc}")
        return None


def fetch_price_history_baostock(
    symbol: str, start_date: str, end_date: str, adjust: str = "qfq"
) -> Optional[List[Dict[str, Any]]]:
    try:
        import baostock as bs
    except ImportError:
        logger.debug("BaoStock 未安装，跳过")
        return None
    try:
        bs.login()
        adjust_flag = "2" if adjust == "qfq" else ("3" if adjust == "hfq" else "1")
        rs = bs.query_history_k_data_plus(
            symbol,
            fields="date,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date if "-" not in start_date else start_date.replace("-", ""),
            end_date=end_date if "-" not in end_date else end_date.replace("-", ""),
            frequency="d",
            adjustflag=adjust_flag,
        )

        records = []
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            close_val = _safe_float_val(row[4])
            pre_close = _safe_float_val(row[4])
            pct_change = _safe_float_val(row[8])
            high_val = _safe_float_val(row[2])
            low_val = _safe_float_val(row[3])
            change_amount = None
            if close_val is not None and pre_close is not None and pct_change is not None:
                pre_close_calc = close_val / (1 + pct_change / 100) if pct_change != 0 else close_val
                change_amount = close_val - pre_close_calc
            pre_close_safe = pre_close_calc if pre_close is not None else None
            amplitude = None
            if high_val and low_val and pre_close_safe and pre_close_safe != 0:
                amplitude = (high_val - low_val) / pre_close_safe * 100

            records.append(
                {
                    "日期": row[0],
                    "开盘": _safe_float_val(row[1]),
                    "最高": high_val,
                    "最低": low_val,
                    "收盘": close_val,
                    "成交量": _safe_float_val(row[5]),
                    "成交额": _safe_float_val(row[6]),
                    "振幅": amplitude,
                    "涨跌幅": pct_change,
                    "涨跌额": change_amount,
                    "换手率": _safe_float_val(row[7]),
                }
            )
        bs.logout()
        return records if records else None
    except Exception as exc:
        logger.warning(f"BaoStock 价格历史获取失败: {exc}")
        try:
            bs.logout()
        except Exception:
            pass
        return None


def fetch_financial_metrics_tushare(symbol: str) -> Optional[List[Dict[str, Any]]]:
    pro = _tushare_pro()
    if pro is None:
        return None
    try:
        ts_code = _to_ts_code(symbol)
        fina = pro.fina_indicator(ts_code=ts_code)
        if fina is None or fina.empty:
            return None

        fina = fina.sort_values("end_date", ascending=False)
        latest = fina.iloc[0]

        def _pct(val):
            v = _safe_float_val(val)
            return v / 100.0 if v is not None else None

        metrics = {
            "return_on_equity": _pct(latest.get("roe")),
            "net_margin": _pct(latest.get("netprofit_margin")),
            "operating_margin": _pct(latest.get("op_yoy")),
            "revenue_growth": _pct(latest.get("or_yoy")),
            "earnings_growth": _pct(latest.get("netprofit_yoy")),
            "book_value_growth": None,
            "current_ratio": _safe_float_val(latest.get("current_ratio")),
            "debt_to_equity": _pct(latest.get("debt_to_assets")),
            "free_cash_flow_per_share": None,
            "earnings_per_share": _safe_float_val(latest.get("eps")),
            "pe_ratio": None,
            "price_to_book": None,
            "price_to_sales": None,
            "market_cap": None,
            "float_market_cap": None,
            "_status": "ok",
            "_data_source": {
                "market_snapshot": "unavailable",
                "financial_indicator": "tushare",
                "income_statement": "tushare",
            },
        }
        return [metrics]
    except Exception as exc:
        logger.warning(f"Tushare 财务指标获取失败: {exc}")
        return None


def fetch_financial_statements_tushare(symbol: str) -> Optional[List[Dict[str, Any]]]:
    pro = _tushare_pro()
    if pro is None:
        return None
    try:
        ts_code = _to_ts_code(symbol)

        income = pro.income(ts_code=ts_code)
        balancesheet = pro.balancesheet(ts_code=ts_code)
        cashflow = pro.cashflow(ts_code=ts_code)

        if income is None or income.empty:
            return None

        income = income.sort_values("end_date", ascending=False)
        balancesheet = balancesheet.sort_values("end_date", ascending=False) if balancesheet is not None and not balancesheet.empty else pd.DataFrame()
        cashflow = cashflow.sort_values("end_date", ascending=False) if cashflow is not None and not cashflow.empty else pd.DataFrame()

        latest_income = income.iloc[0]
        previous_income = income.iloc[1] if len(income) > 1 else latest_income

        latest_balance = balancesheet.iloc[0] if not balancesheet.empty else pd.Series()
        previous_balance = balancesheet.iloc[1] if len(balancesheet) > 1 else latest_balance

        latest_cashflow = cashflow.iloc[0] if not cashflow.empty else pd.Series()
        previous_cashflow = cashflow.iloc[1] if len(cashflow) > 1 else latest_cashflow

        def _get(series, *keys):
            for k in keys:
                v = series.get(k) if isinstance(series, pd.Series) else None
                if v is not None:
                    return float(v) if not pd.isna(v) else 0.0
            return 0.0

        current_item = {
            "net_income": _get(latest_income, "n_income", "net_profit"),
            "operating_revenue": _get(latest_income, "total_cogs", "revenue"),
            "operating_profit": _get(latest_income, "operate_profit"),
            "working_capital": _get(latest_balance, "total_cur_assets") - _get(latest_balance, "total_cur_liab"),
            "depreciation_and_amortization": _get(latest_cashflow, "c_fix_assets_depr"),
            "capital_expenditure": abs(_get(latest_cashflow, "c_pay_acq_const_fiolta")),
            "free_cash_flow": _get(latest_cashflow, "n_cashflow_act") - abs(_get(latest_cashflow, "c_pay_acq_const_fiolta")),
        }

        previous_item = {
            "net_income": _get(previous_income, "n_income", "net_profit"),
            "operating_revenue": _get(previous_income, "total_cogs", "revenue"),
            "operating_profit": _get(previous_income, "operate_profit"),
            "working_capital": _get(previous_balance, "total_cur_assets") - _get(previous_balance, "total_cur_liab"),
            "depreciation_and_amortization": _get(previous_cashflow, "c_fix_assets_depr"),
            "capital_expenditure": abs(_get(previous_cashflow, "c_pay_acq_const_fiolta")),
            "free_cash_flow": _get(previous_cashflow, "n_cashflow_act") - abs(_get(previous_cashflow, "c_pay_acq_const_fiolta")),
        }

        return [current_item, previous_item]
    except Exception as exc:
        logger.warning(f"Tushare 财务报表获取失败: {exc}")
        return None


def _safe_float_val(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
