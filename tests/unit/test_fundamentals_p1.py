import json

from src.agents.fundamentals import (
    analyze_metric_trend,
    calculate_peer_percentile_signal,
    fundamentals_agent,
)


def test_analyze_metric_trend_flags_large_decline_as_bearish():
    result = analyze_metric_trend(
        [
            {"return_on_equity": 0.20},
            {"return_on_equity": 0.18},
            {"return_on_equity": 0.15},
        ],
        "return_on_equity",
    )

    assert result["signal"] == "bearish"
    assert result["trend"] == "declining"
    assert result["change_pct"] == -0.25
    assert result["data_available"] is True


def test_analyze_metric_trend_flags_large_improvement_as_bullish():
    result = analyze_metric_trend(
        [
            {"net_margin": 0.10},
            {"net_margin": 0.11},
            {"net_margin": 0.13},
        ],
        "net_margin",
    )

    assert result["signal"] == "bullish"
    assert result["trend"] == "improving"
    assert result["change_pct"] == 0.30


def test_analyze_metric_trend_returns_neutral_for_small_or_missing_history():
    small_move = analyze_metric_trend(
        [{"revenue_growth": 0.10}, {"revenue_growth": 0.11}],
        "revenue_growth",
    )
    missing = analyze_metric_trend([{"revenue_growth": None}], "revenue_growth")

    assert small_move["signal"] == "neutral"
    assert small_move["trend"] == "stable"
    assert missing["signal"] == "neutral"
    assert missing["data_available"] is False


def test_calculate_peer_percentile_signal_higher_is_better():
    top = calculate_peer_percentile_signal(0.18, [0.08, 0.10, 0.12, 0.15, 0.16])
    bottom = calculate_peer_percentile_signal(0.07, [0.08, 0.10, 0.12, 0.15, 0.16])

    assert top["signal"] == "bullish"
    assert top["percentile"] > 0.70
    assert bottom["signal"] == "bearish"
    assert bottom["percentile"] < 0.30


def test_calculate_peer_percentile_signal_lower_is_better_inverts_signal():
    cheap = calculate_peer_percentile_signal(
        8,
        [10, 12, 15, 20, 25],
        higher_is_better=False,
    )
    expensive = calculate_peer_percentile_signal(
        30,
        [10, 12, 15, 20, 25],
        higher_is_better=False,
    )

    assert cheap["signal"] == "bullish"
    assert expensive["signal"] == "bearish"


def test_fundamentals_agent_adds_optional_trend_and_peer_reasoning_without_static_regression():
    state = {
        "messages": [],
        "data": {
            "financial_metrics": [
                {
                    "return_on_equity": 0.18,
                    "net_margin": 0.24,
                    "operating_margin": 0.20,
                    "revenue_growth": 0.08,
                    "earnings_growth": 0.08,
                    "book_value_growth": 0.08,
                    "current_ratio": 1.2,
                    "debt_to_equity": 0.8,
                    "free_cash_flow_per_share": 0.5,
                    "earnings_per_share": 1.0,
                    "pe_ratio": 20,
                    "price_to_book": 2,
                    "price_to_sales": 4,
                }
            ],
            "metrics_history": [
                {"return_on_equity": 0.24, "net_margin": 0.20, "revenue_growth": 0.06},
                {"return_on_equity": 0.22, "net_margin": 0.22, "revenue_growth": 0.08},
                {"return_on_equity": 0.18, "net_margin": 0.24, "revenue_growth": 0.10},
            ],
            "industry_peer_metrics": {
                "return_on_equity": [0.08, 0.10, 0.12, 0.14, 0.16],
                "pe_ratio": [12, 14, 18, 22, 28],
            },
            "industry_classification": "default",
        },
        "metadata": {"show_reasoning": False},
    }

    result = fundamentals_agent(state)
    message_content = json.loads(result["messages"][0].content)
    reasoning = message_content["reasoning"]

    assert reasoning["profitability_signal"]["signal"] == "bullish"
    assert reasoning["trend_analysis"]["return_on_equity"]["signal"] == "bearish"
    assert reasoning["peer_percentile_analysis"]["return_on_equity"]["signal"] == "bullish"
    assert reasoning["peer_percentile_analysis"]["pe_ratio"]["signal"] == "neutral"
