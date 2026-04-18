from src.agents.technicals import (
    calculate_kdj,
    calculate_macd,
    detect_macd_divergence,
)


def test_calculate_kdj_returns_fields(sample_prices):
    result = calculate_kdj(sample_prices)
    assert set(result.keys()) == {"k", "d", "j", "signal"}
    assert result["signal"] in {"bullish", "bearish", "neutral"}


def test_detect_macd_divergence_returns_contract(top_divergence_prices):
    macd_line, _ = calculate_macd(top_divergence_prices)
    result = detect_macd_divergence(top_divergence_prices, macd_line=macd_line)
    assert set(result.keys()) == {"divergence", "signal", "reason"}
    assert result["signal"] in {"bullish", "bearish", "neutral"}
