from src.agents.valuation import (
    adjust_equity_value_for_net_debt,
    calculate_comparable_value,
    calculate_peg_ratio,
)


def test_calculate_peg_ratio_handles_positive_growth():
    assert calculate_peg_ratio(pe_ratio=20, earnings_growth=0.25) == 0.8


def test_calculate_peg_ratio_returns_none_for_nonpositive_or_missing_growth():
    assert calculate_peg_ratio(pe_ratio=20, earnings_growth=0) is None
    assert calculate_peg_ratio(pe_ratio=20, earnings_growth=-0.1) is None
    assert calculate_peg_ratio(pe_ratio=None, earnings_growth=0.2) is None


def test_comparable_value_uses_pe_pb_average_when_available():
    value = calculate_comparable_value(
        metrics={"earnings_per_share": 2.0, "book_value_per_share": 10.0},
        comparable_multiples={"pe": 15, "pb": 2},
        shares_outstanding=1000,
    )

    assert value == 25_000


def test_comparable_value_falls_back_to_market_cap_multiple_scaling():
    value = calculate_comparable_value(
        metrics={"pe_ratio": 20, "price_to_book": 4, "market_cap": 100_000},
        comparable_multiples={"pe": 15, "pb": 2},
    )

    assert value == 62_500


def test_comparable_value_returns_none_when_no_inputs_usable():
    assert calculate_comparable_value({}, {"pe": 15, "pb": 2}) is None


def test_adjust_equity_value_subtracts_net_debt():
    assert adjust_equity_value_for_net_debt(
        enterprise_value=100,
        cash=20,
        total_debt=40,
    ) == 80


def test_adjust_equity_value_degrades_to_original_value_when_cash_or_debt_missing():
    assert adjust_equity_value_for_net_debt(enterprise_value=100, cash=None, total_debt=40) == 100
    assert adjust_equity_value_for_net_debt(enterprise_value=100, cash=20, total_debt=None) == 100
