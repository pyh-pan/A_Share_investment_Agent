import pandas as pd

from src.tools.api import get_margin_trading_sentiment, prewarm_symbol_cache
from src.tools.data_source_manager import DataSourceManager


def test_get_margin_trading_sentiment_szse_bullish(monkeypatch):
    monkeypatch.setattr(
        DataSourceManager,
        "fetch_with_fallback",
        lambda self, cache_key, fetchers, source_names, cache_ttl_hours=24: fetchers[0](),
    )
    df = pd.DataFrame(
        {
            "融资余额": [1200.0, 1000.0],
            "融券余额": [50.0, 60.0],
        }
    )
    monkeypatch.setattr(
        "akshare.stock_margin_detail_szse",
        lambda symbol: df,
        raising=False,
    )

    result = get_margin_trading_sentiment("300001")

    assert result["data_available"] is True
    assert result["sentiment"] == "strong_bullish"
    assert result["signal"] == "bullish"


def test_get_margin_trading_sentiment_sse_bearish(monkeypatch):
    monkeypatch.setattr(
        DataSourceManager,
        "fetch_with_fallback",
        lambda self, cache_key, fetchers, source_names, cache_ttl_hours=24: fetchers[0](),
    )
    df = pd.DataFrame(
        {
            "融资余额": [900.0, 1000.0],
            "融券余额": [100.0, 100.0],
        }
    )
    monkeypatch.setattr(
        "akshare.stock_margin_detail_sse",
        lambda symbol: df,
        raising=False,
    )

    result = get_margin_trading_sentiment("600001")

    assert result["data_available"] is True
    assert result["signal"] == "bearish"


def test_get_margin_trading_sentiment_returns_neutral_on_missing_data(monkeypatch):
    monkeypatch.setattr(
        DataSourceManager,
        "fetch_with_fallback",
        lambda self, cache_key, fetchers, source_names, cache_ttl_hours=24: None,
    )

    result = get_margin_trading_sentiment("600001")

    assert result["data_available"] is False
    assert result["signal"] == "neutral"


def test_prewarm_symbol_cache_reports_step_status(monkeypatch):
    calls = []

    def _ok(name):
        def inner(*args, **kwargs):
            calls.append(name)
            return {"ok": True}
        return inner

    monkeypatch.setattr("src.tools.api.get_price_history", _ok("price_history"))
    monkeypatch.setattr("src.tools.api.get_financial_metrics", _ok("financial_metrics"))
    monkeypatch.setattr("src.tools.api.get_financial_statements", _ok("financial_statements"))
    monkeypatch.setattr("src.tools.api.get_market_data", _ok("market_data"))
    monkeypatch.setattr("src.tools.api.get_northbound_flow", _ok("northbound_flow"))
    monkeypatch.setattr("src.tools.api.get_macro_indicators", _ok("macro_indicators"))
    monkeypatch.setattr("src.tools.api.get_margin_trading_sentiment", _ok("margin_trading"))

    result = prewarm_symbol_cache("600001", "2025-01-01", "2025-12-31")

    assert set(result.keys()) == {
        "price_history",
        "financial_metrics",
        "financial_statements",
        "market_data",
        "northbound_flow",
        "macro_indicators",
        "margin_trading",
    }
    assert all(item["status"] == "ok" for item in result.values())
    assert "price_history" in calls
