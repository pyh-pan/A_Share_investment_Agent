from langchain_core.messages import HumanMessage
from src.utils.logging_config import setup_logger
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint, log_llm_interaction
from src.tools.api import calculate_wacc
from src.agents.fundamentals import INDUSTRY_THRESHOLDS
import json

# 初始化 logger
logger = setup_logger('valuation_agent')


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_peg_ratio(pe_ratio, earnings_growth):
    pe_ratio = _safe_float(pe_ratio)
    earnings_growth = _safe_float(earnings_growth)
    if pe_ratio is None or pe_ratio <= 0:
        return None
    if earnings_growth is None or earnings_growth <= 0:
        return None
    return round(pe_ratio / (earnings_growth * 100), 2)


def calculate_comparable_value(
    metrics: dict,
    comparable_multiples: dict,
    shares_outstanding=None,
    market_cap=None,
):
    values = []
    shares = _safe_float(shares_outstanding)
    market_cap = _safe_float(market_cap, _safe_float(metrics.get("market_cap")))

    if shares and shares > 0:
        eps = _safe_float(metrics.get("earnings_per_share"))
        bvps = _safe_float(metrics.get("book_value_per_share"))
        revenue_per_share = _safe_float(metrics.get("revenue_per_share"))

        if eps and eps > 0 and comparable_multiples.get("pe"):
            values.append(eps * _safe_float(comparable_multiples.get("pe")) * shares)
        if bvps and bvps > 0 and comparable_multiples.get("pb"):
            values.append(bvps * _safe_float(comparable_multiples.get("pb")) * shares)
        if revenue_per_share and revenue_per_share > 0 and comparable_multiples.get("ps"):
            values.append(revenue_per_share * _safe_float(comparable_multiples.get("ps")) * shares)

    if market_cap and market_cap > 0:
        current_pe = _safe_float(metrics.get("pe_ratio"))
        current_pb = _safe_float(metrics.get("price_to_book"))
        current_ps = _safe_float(metrics.get("price_to_sales"))
        target_pe = _safe_float(comparable_multiples.get("pe"))
        target_pb = _safe_float(comparable_multiples.get("pb"))
        target_ps = _safe_float(comparable_multiples.get("ps"))

        if not values and current_pe and current_pe > 0 and target_pe:
            values.append(market_cap * target_pe / current_pe)
        if not values and current_pb and current_pb > 0 and target_pb:
            values.append(market_cap * target_pb / current_pb)
        if not values and current_ps and current_ps > 0 and target_ps:
            values.append(market_cap * target_ps / current_ps)
        if values and shares is None:
            # For fallback scaling, average all available current multiple methods.
            fallback_values = []
            if current_pe and current_pe > 0 and target_pe:
                fallback_values.append(market_cap * target_pe / current_pe)
            if current_pb and current_pb > 0 and target_pb:
                fallback_values.append(market_cap * target_pb / current_pb)
            if current_ps and current_ps > 0 and target_ps:
                fallback_values.append(market_cap * target_ps / current_ps)
            if fallback_values:
                values = fallback_values

    if not values:
        return None
    return sum(values) / len(values)


def adjust_equity_value_for_net_debt(enterprise_value, cash, total_debt):
    enterprise_value = _safe_float(enterprise_value, 0.0)
    cash = _safe_float(cash)
    total_debt = _safe_float(total_debt)
    if cash is None or total_debt is None:
        return enterprise_value
    net_debt = total_debt - cash
    return max(enterprise_value - net_debt, 0.0)


@agent_endpoint("valuation", "估值分析师，使用DCF和所有者收益法评估公司内在价值")
def valuation_agent(state: AgentState):
    """Responsible for valuation analysis"""
    show_workflow_status("估值分析师")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    symbol = data["ticker"]
    metrics = data["financial_metrics"][0]
    current_financial_line_item = data["financial_line_items"][0]
    previous_financial_line_item = data["financial_line_items"][1]
    market_cap = data["market_cap"]

    if not market_cap or market_cap <= 0:
        message_content = {
            "signal": "neutral",
            "confidence": "0%",
            "status": "unavailable",
            "reasoning": {
                "error": "市值数据不可用，跳过估值分析。"
            }
        }
        message = HumanMessage(
            content=json.dumps(message_content),
            name="valuation_agent",
        )
        if show_reasoning:
            show_agent_reasoning(message_content, "估值分析")
            state["metadata"]["agent_reasoning"] = message_content
        show_workflow_status("估值分析师", "completed")
        return {
            "messages": [message],
            "data": {
                **data,
                "valuation_analysis": message_content
            },
            "metadata": state["metadata"],
        }

    reasoning = {}
    earnings_growth = metrics.get("earnings_growth") or 0.0

    # Calculate working capital change
    working_capital_change = (current_financial_line_item.get(
        'working_capital') or 0) - (previous_financial_line_item.get('working_capital') or 0)

    dynamic_wacc = calculate_wacc(symbol=symbol, financial_metrics=metrics)

    # Owner Earnings Valuation (Buffett Method)
    owner_earnings_value = calculate_owner_earnings_value(
        net_income=current_financial_line_item.get('net_income'),
        depreciation=current_financial_line_item.get(
            'depreciation_and_amortization'),
        capex=current_financial_line_item.get('capital_expenditure'),
        working_capital_change=working_capital_change,
        growth_rate=earnings_growth,
        required_return=min(dynamic_wacc + 0.05, 0.25),
        margin_of_safety=0.25
    )

    # DCF Valuation
    dcf_value = calculate_intrinsic_value(
        free_cash_flow=current_financial_line_item.get('free_cash_flow'),
        growth_rate=earnings_growth,
        discount_rate=dynamic_wacc,
        terminal_growth_rate=0.03,
        num_years=5,
    )

    cash = current_financial_line_item.get("cash_and_equivalents")
    total_debt = current_financial_line_item.get("total_debt")
    adjusted_dcf_value = adjust_equity_value_for_net_debt(dcf_value, cash, total_debt)
    adjusted_owner_earnings_value = adjust_equity_value_for_net_debt(
        owner_earnings_value, cash, total_debt
    )

    industry = data.get("industry_classification", "default")
    comparable_profile = INDUSTRY_THRESHOLDS.get(industry, INDUSTRY_THRESHOLDS["default"])
    comparable_multiples = {
        "pe": comparable_profile.get("pe_ratio"),
        "pb": comparable_profile.get("price_to_book"),
        "ps": comparable_profile.get("price_to_sales"),
    }
    shares_outstanding = metrics.get("total_shares") or metrics.get("float_shares")
    comparable_value = calculate_comparable_value(
        metrics=metrics,
        comparable_multiples=comparable_multiples,
        shares_outstanding=shares_outstanding,
        market_cap=market_cap,
    )
    peg_ratio = calculate_peg_ratio(metrics.get("pe_ratio"), earnings_growth)

    # Calculate combined valuation gap from available value-based methods.
    dcf_gap = (adjusted_dcf_value - market_cap) / market_cap
    owner_earnings_gap = (adjusted_owner_earnings_value - market_cap) / market_cap
    value_gaps = [dcf_gap, owner_earnings_gap]
    comparable_gap = None
    if comparable_value is not None:
        comparable_gap = (comparable_value - market_cap) / market_cap
        value_gaps.append(comparable_gap)
    valuation_gap = sum(value_gaps) / len(value_gaps)

    if (current_financial_line_item.get('free_cash_flow') or 0) <= 0:
        reasoning["cash_flow_warning"] = {
            "signal": "bearish",
            "details": "自由现金流为负，DCF估值参考意义有限，已按保守方式处理。",
        }

    if valuation_gap > 0.10:  # Changed from 0.15 to 0.10 (10% undervalued)
        signal = 'bullish'
    elif valuation_gap < -0.20:  # Changed from -0.15 to -0.20 (20% overvalued)
        signal = 'bearish'
    else:
        signal = 'neutral'

    reasoning["dcf_analysis"] = {
        "signal": "bullish" if dcf_gap > 0.10 else "bearish" if dcf_gap < -0.20 else "neutral",
        "details": f"股权价值: ¥{adjusted_dcf_value:,.2f}, 原始DCF: ¥{dcf_value:,.2f}, 市值: ¥{market_cap:,.2f}, 差距: {dcf_gap:.1%}, WACC: {dynamic_wacc:.2%}"
    }

    reasoning["owner_earnings_analysis"] = {
        "signal": "bullish" if owner_earnings_gap > 0.10 else "bearish" if owner_earnings_gap < -0.20 else "neutral",
        "details": f"所有者收益股权价值: ¥{adjusted_owner_earnings_value:,.2f}, 原始估值: ¥{owner_earnings_value:,.2f}, 市值: ¥{market_cap:,.2f}, 差距: {owner_earnings_gap:.1%}"
    }

    reasoning["peg_analysis"] = {
        "signal": "bullish" if peg_ratio is not None and peg_ratio < 1 else "bearish" if peg_ratio is not None and peg_ratio > 2 else "neutral",
        "details": f"PEG: {peg_ratio:.2f}" if peg_ratio is not None else "PEG不可用（市盈率或盈利增速不可用/非正）",
    }

    reasoning["comparable_analysis"] = {
        "signal": "bullish" if comparable_gap is not None and comparable_gap > 0.10 else "bearish" if comparable_gap is not None and comparable_gap < -0.20 else "neutral",
        "details": (
            f"行业: {industry}, 可比倍数估值: ¥{comparable_value:,.2f}, 差距: {comparable_gap:.1%}, 倍数: {comparable_multiples}"
            if comparable_value is not None
            else f"行业: {industry}, 可比估值不可用，缺少每股/当前倍数字段"
        ),
    }

    reasoning["net_debt_adjustment"] = {
        "cash_and_equivalents": cash,
        "total_debt": total_debt,
        "details": (
            "已按净债务从企业价值调整为股权价值"
            if cash is not None and total_debt is not None
            else "现金或有息债务数据不可用，未做净债务调整"
        ),
    }

    message_content = {
        "signal": signal,
        "confidence": f"{abs(valuation_gap):.0%}",
        "reasoning": {
            **reasoning,
            "valuation_assumptions": {
                "wacc": dynamic_wacc,
                "earnings_growth": earnings_growth,
            },
        },
    }

    message = HumanMessage(
        content=json.dumps(message_content),
        name="valuation_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "估值分析")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("估值分析师", "completed")
    # logger.info(
    # f"--- DEBUG: valuation_agent RETURN messages: {[msg.name for msg in [message]]} ---")
    return {
        "messages": [message],
        "data": {
            **data,
            "valuation_analysis": message_content,
            "valuation_report": message_content,
        },
        "metadata": state["metadata"],
    }


def calculate_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5


) -> float:
    """
    使用改进的所有者收益法计算公司价值。

    Args:
        net_income: 净利润
        depreciation: 折旧和摊销
        capex: 资本支出
        working_capital_change: 营运资金变化
        growth_rate: 预期增长率
        required_return: 要求回报率
        margin_of_safety: 安全边际
        num_years: 预测年数

    Returns:
        float: 计算得到的公司价值
    """
    try:
        # 数据有效性检查
        if not all(isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]):
            return 0

        # 计算初始所有者收益
        owner_earnings = (
            net_income +
            depreciation -
            capex -
            working_capital_change
        )

        if owner_earnings <= 0:
            return 0

        # 调整增长率，允许负增长但设置下限
        growth_rate = min(max(growth_rate, -0.10), 0.25)

        # 计算预测期收益现值
        future_values = []
        for year in range(1, num_years + 1):
            # 使用递减增长率模型
            year_growth = growth_rate * (1 - year / (2 * num_years))
            future_value = owner_earnings * (1 + year_growth) ** year
            discounted_value = future_value / (1 + required_return) ** year
            future_values.append(discounted_value)

        # 计算永续价值
        terminal_growth = min(max(growth_rate * 0.4, 0.0), 0.03)
        terminal_value = (
            future_values[-1] * (1 + terminal_growth)) / (required_return - terminal_growth)
        terminal_value_discounted = terminal_value / \
            (1 + required_return) ** num_years

        # 计算总价值并应用安全边际
        intrinsic_value = sum(future_values) + terminal_value_discounted
        value_with_safety_margin = intrinsic_value * (1 - margin_of_safety)

        return max(value_with_safety_margin, 0)  # 确保不返回负值

    except Exception as e:
        print(f"所有者收益计算错误: {e}")
        return 0


def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """
    使用改进的DCF方法计算内在价值，考虑增长率和风险因素。

    Args:
        free_cash_flow: 自由现金流
        growth_rate: 预期增长率
        discount_rate: 基础折现率
        terminal_growth_rate: 永续增长率
        num_years: 预测年数

    Returns:
        float: 计算得到的内在价值
    """
    try:
        if not isinstance(free_cash_flow, (int, float)) or free_cash_flow <= 0:
            return 0

        # 调整增长率，允许负增长但设置下限
        growth_rate = min(max(growth_rate, -0.10), 0.25)

        # 调整永续增长率，不能超过经济平均增长
        terminal_growth_rate = min(max(growth_rate * 0.4, 0.0), 0.03)

        # 计算预测期现金流现值
        present_values = []
        for year in range(1, num_years + 1):
            future_cf = free_cash_flow * (1 + growth_rate) ** year
            present_value = future_cf / (1 + discount_rate) ** year
            present_values.append(present_value)

        # 计算永续价值
        terminal_year_cf = free_cash_flow * (1 + growth_rate) ** num_years
        terminal_value = terminal_year_cf * \
            (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
        terminal_present_value = terminal_value / \
            (1 + discount_rate) ** num_years

        # 总价值
        total_value = sum(present_values) + terminal_present_value

        return max(total_value, 0)  # 确保不返回负值

    except Exception as e:
        print(f"DCF计算错误: {e}")
        return 0


def calculate_working_capital_change(
    current_working_capital: float,
    previous_working_capital: float,
) -> float:
    """
    Calculate the absolute change in working capital between two periods.
    A positive change means more capital is tied up in working capital (cash outflow).
    A negative change means less capital is tied up (cash inflow).

    Args:
        current_working_capital: Current period's working capital
        previous_working_capital: Previous period's working capital

    Returns:
        float: Change in working capital (current - previous)
    """
    return current_working_capital - previous_working_capital
