from src.agents.portfolio_manager import (
    SIGNAL_WEIGHTS,
    calculate_transaction_costs,
    format_decision,
)


def test_calculate_transaction_costs_buy():
    result = calculate_transaction_costs(
        action="buy",
        quantity=1000,
        price=10,
        expected_return=0.05,
    )

    assert result["cost_rate"] == 0.00027
    assert result["transaction_cost"] == 2.7
    assert result["net_return"] == 497.3
    assert result["profitable"] is True


def test_calculate_transaction_costs_sell_includes_stamp_tax():
    result = calculate_transaction_costs(
        action="sell",
        quantity=1000,
        price=10,
        expected_return=0.05,
    )

    assert result["cost_rate"] == 0.00127
    assert result["transaction_cost"] == 12.7
    assert result["net_return"] == 487.3


def test_calculate_transaction_costs_hold_has_no_cost():
    result = calculate_transaction_costs("hold", 1000, 10, 0.05)

    assert result["transaction_cost"] == 0
    assert result["net_return"] == 500


def test_format_decision_weight_labels_match_signal_weights():
    report = format_decision(
        action="hold",
        quantity=0,
        confidence=0.5,
        agent_signals=[],
        reasoning="test",
    )["分析报告"]

    assert f"估值分析 (权重{int(SIGNAL_WEIGHTS['valuation'] * 100)}%)" in report
    assert f"基本面分析 (权重{int(SIGNAL_WEIGHTS['fundamentals'] * 100)}%)" in report
    assert f"技术分析 (权重{int(SIGNAL_WEIGHTS['technical'] * 100)}%)" in report
    assert f"宏观分析 (综合权重{int(SIGNAL_WEIGHTS['macro'] * 100)}%)" in report
    assert f"情绪分析 (权重{int(SIGNAL_WEIGHTS['sentiment'] * 100)}%)" in report
