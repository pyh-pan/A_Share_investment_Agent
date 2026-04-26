import pandas as pd

from src.tools.cache.db_cache import SimpleCache
from src.tools.api import calculate_wacc, get_northbound_flow
from src.tools.data_source_manager import DataSourceManager


def test_data_source_manager_import_and_cache_roundtrip(tmp_path):
    cache = SimpleCache(db_path=tmp_path / "cache.db")
    cache.set("k", {"value": 1}, ttl_hours=1)

    assert cache.get("k", max_age_hours=1) == {"value": 1}
    assert DataSourceManager(cache=cache).fetch_with_fallback(
        cache_key="k",
        fetchers=[lambda: {"value": 2}],
        source_names=["fallback"],
        cache_ttl_hours=1,
    ) == {"value": 1}


def test_simple_cache_treats_zero_age_as_expired(tmp_path):
    cache = SimpleCache(db_path=tmp_path / "cache.db")
    cache.set("expired", {"value": 1}, ttl_hours=1)

    assert cache.get("expired", max_age_hours=0) is None


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
