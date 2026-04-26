import json

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.risk_manager import (
    calculate_beta_risk,
    calculate_expected_shortfall,
    calculate_liquidity_risk,
    calculate_t1_settlement_constraint,
    risk_management_agent,
)


def _prices(count=90):
    dates = pd.date_range(end=pd.Timestamp.now(), periods=count, freq="D")
    close = 20 + np.linspace(0, 3, count) + np.sin(np.linspace(0, 8, count))
    return [
        {
            "date": date,
            "open": price * 0.99,
            "high": price * 1.02,
            "low": price * 0.98,
            "close": price,
            "volume": 100_000,
            "amount": price * 100_000,
            "turnover": 1.2,
        }
        for date, price in zip(dates, close)
    ]


def test_expected_shortfall_averages_tail_losses():
    returns = pd.Series([-0.10, -0.05, -0.02, 0.01, 0.03])

    assert calculate_expected_shortfall(returns, confidence=0.8) == -0.10


def test_expected_shortfall_handles_empty_returns():
    assert calculate_expected_shortfall(pd.Series([], dtype=float)) == 0.0
    assert calculate_expected_shortfall(pd.Series([np.nan, np.nan])) == 0.0


def test_liquidity_risk_flags_low_turnover_and_amount():
    result = calculate_liquidity_risk(
        prices_df=pd.DataFrame(
            {
                "turnover": [0.1, 0.2, 0.15],
                "amount": [1_000_000, 900_000, 800_000],
            }
        ),
        max_position_size=5_000_000,
    )

    assert result["score"] >= 2
    assert result["signal"] in {"elevated", "high"}


def test_liquidity_risk_uses_turnover_rate_alias():
    result = calculate_liquidity_risk(
        prices_df=pd.DataFrame(
            {
                "turnover_rate": [0.1, 0.2, 0.15],
                "amount": [2_000_000, 2_000_000, 2_000_000],
            }
        ),
        max_position_size=100_000,
    )

    assert result["avg_turnover"] > 0


def test_liquidity_risk_falls_back_to_close_times_volume():
    result = calculate_liquidity_risk(
        prices_df=pd.DataFrame({"close": [10, 11], "volume": [1000, 1000]}),
        max_position_size=10_000,
    )

    assert result["avg_daily_amount"] > 0


def test_t1_constraint_blocks_same_day_sell_without_prior_position():
    result = calculate_t1_settlement_constraint({"stock": 100, "sellable_stock": 0})

    assert result["sellable_stock"] == 0
    assert result["can_sell"] is False


def test_t1_constraint_defaults_stock_to_sellable_for_legacy_portfolio():
    result = calculate_t1_settlement_constraint({"stock": 100})

    assert result["sellable_stock"] == 100
    assert result["can_sell"] is True


def test_beta_risk_flags_high_beta():
    result = calculate_beta_risk(1.6)

    assert result["score"] > 0
    assert result["signal"] == "high"


def test_risk_agent_emits_new_metrics(monkeypatch):
    monkeypatch.setattr("src.agents.risk_manager.calculate_beta", lambda symbol: 1.5)
    state = {
        "messages": [
            HumanMessage(
                content=json.dumps(
                    {
                        "signal": "bullish",
                        "bull_confidence": 0.7,
                        "bear_confidence": 0.2,
                        "confidence": 0.8,
                    }
                ),
                name="debate_room_agent",
            )
        ],
        "data": {
            "ticker": "600519",
            "prices": _prices(),
            "portfolio": {"cash": 100_000, "stock": 100},
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    metrics = result["data"]["risk_analysis"]["risk_metrics"]

    assert "conditional_value_at_risk_95" in metrics
    assert "liquidity_risk" in metrics
    assert "t1_settlement" in metrics
    assert "beta_risk" in metrics


def test_t1_blocks_sell_action(monkeypatch):
    monkeypatch.setattr("src.agents.risk_manager.calculate_beta", lambda symbol: 1.0)
    state = {
        "messages": [
            HumanMessage(
                content=json.dumps(
                    {
                        "signal": "bearish",
                        "bull_confidence": 0.1,
                        "bear_confidence": 0.8,
                        "confidence": 0.9,
                    }
                ),
                name="debate_room_agent",
            )
        ],
        "data": {
            "ticker": "600519",
            "prices": _prices(),
            "portfolio": {"cash": 100_000, "stock": 100, "sellable_stock": 0},
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)

    assert result["data"]["risk_analysis"]["trading_action"] != "sell"
