import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    np.random.seed(7)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=80, freq="D")
    close = 50 + np.cumsum(np.random.randn(80) * 0.5)
    return pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "open": close - np.random.rand(80),
            "high": close + np.random.rand(80),
            "low": close - np.random.rand(80),
            "volume": np.random.randint(10_000, 500_000, size=80),
        }
    )


@pytest.fixture
def top_divergence_prices() -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.now(), periods=80, freq="D")
    # Push price to new high in late stage
    close = np.concatenate(
        [
            np.linspace(30, 40, 40),
            np.linspace(40, 35, 20),
            np.linspace(35, 42, 20),
        ]
    )
    open_ = close - 0.2
    high = close + 0.5
    low = close - 0.5
    volume = np.full(80, 100000)
    return pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "open": open_,
            "high": high,
            "low": low,
            "volume": volume,
        }
    )
