import pandas as pd

from src.tools.api import calculate_wacc, get_northbound_flow
from src.tools.data_source_manager import DataSourceManager


def test_get_northbound_flow_signal_bullish(monkeypatch):
    # Bypass DataSourceManager cache so the akshare mock is actually invoked
    monkeypatch.setattr(
        DataSourceManager, "fetch_with_fallback",
        lambda self, cache_key, fetchers, source_names, cache_ttl_hours=24: fetchers[0](),
    )
    mock_df = pd.DataFrame({"当日净流入": [12, 13, 14, 15, 16]})
    monkeypatch.setattr(
        "akshare.stock_hsgt_north_net_flow_in_em",
        lambda symbol: mock_df,
        raising=False,
    )
    result = get_northbound_flow(days=5)
    assert result["signal"] == "bullish"
    assert result["data_available"] is True


def test_calculate_wacc_fallback_range(monkeypatch):
    monkeypatch.setattr("src.tools.api.calculate_beta", lambda symbol: 1.2)
    wacc = calculate_wacc("600519", {"debt_to_equity": 0.4})
    assert 0.08 <= wacc <= 0.20
