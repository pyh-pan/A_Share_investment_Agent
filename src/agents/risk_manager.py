import math

from langchain_core.messages import HumanMessage

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.api import calculate_beta, prices_to_df
from src.utils.api_utils import agent_endpoint, log_llm_interaction

import json
import ast

##### Risk Management Agent #####


def _finite_float(value, default=0.0):
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return default


def calculate_expected_shortfall(returns, confidence=0.95) -> float:
    """Historical CVaR / expected shortfall for the lower return tail."""
    clean_returns = returns.dropna() if hasattr(returns, "dropna") else []
    if len(clean_returns) == 0:
        return 0.0

    var_threshold = clean_returns.quantile(1 - confidence)
    tail_losses = clean_returns[clean_returns <= var_threshold]
    if len(tail_losses) == 0:
        return 0.0
    return _finite_float(tail_losses.mean())


def calculate_liquidity_risk(prices_df, max_position_size) -> dict:
    """Estimate liquidity risk from turnover and daily traded amount."""
    if prices_df is None or prices_df.empty:
        return {
            "score": 0,
            "signal": "unknown",
            "avg_daily_amount": 0.0,
            "avg_turnover": 0.0,
            "position_to_amount_ratio": 0.0,
        }

    if "amount" in prices_df:
        amount = prices_df["amount"]
    elif {"close", "volume"}.issubset(prices_df.columns):
        amount = prices_df["close"] * prices_df["volume"]
    else:
        amount = None

    turnover_col = "turnover" if "turnover" in prices_df else "turnover_rate"
    avg_turnover = (
        _finite_float(prices_df[turnover_col].tail(20).mean())
        if turnover_col in prices_df
        else 0.0
    )
    avg_daily_amount = _finite_float(amount.tail(20).mean()) if amount is not None else 0.0
    max_position_size = max(_finite_float(max_position_size), 0.0)
    position_to_amount_ratio = (
        max_position_size / avg_daily_amount if avg_daily_amount > 0 else 0.0
    )

    score = 0
    if avg_daily_amount and avg_daily_amount < 5_000_000:
        score += 1
    if avg_turnover and avg_turnover < 0.5:
        score += 1
    if position_to_amount_ratio > 1.0:
        score += 2
    elif position_to_amount_ratio > 0.3:
        score += 1

    if score >= 3:
        signal = "high"
    elif score >= 1:
        signal = "elevated"
    else:
        signal = "low"

    return {
        "score": int(score),
        "signal": signal,
        "avg_daily_amount": float(avg_daily_amount),
        "avg_turnover": float(avg_turnover),
        "position_to_amount_ratio": float(position_to_amount_ratio),
    }


def calculate_t1_settlement_constraint(portfolio) -> dict:
    """A-share T+1: only previously settled shares are sellable."""
    stock = int(portfolio.get("stock", 0) or 0)
    sellable_stock = int(portfolio.get("sellable_stock", stock) or 0)
    sellable_stock = max(0, min(sellable_stock, stock))
    blocked_stock = max(stock - sellable_stock, 0)
    return {
        "stock": stock,
        "sellable_stock": sellable_stock,
        "blocked_stock": blocked_stock,
        "can_sell": sellable_stock > 0,
    }


def calculate_beta_risk(beta) -> dict:
    beta = _finite_float(beta, 1.0)
    if beta >= 1.5:
        return {"beta": beta, "score": 2, "signal": "high"}
    if beta >= 1.2:
        return {"beta": beta, "score": 1, "signal": "elevated"}
    if beta <= 0.7:
        return {"beta": beta, "score": 0, "signal": "defensive"}
    return {"beta": beta, "score": 0, "signal": "neutral"}


def calculate_stress_test_results(
    current_position_value,
    total_portfolio_value,
    volatility,
    beta=1.0,
) -> dict:
    current_position_value = _finite_float(current_position_value)
    total_portfolio_value = _finite_float(total_portfolio_value)
    volatility = max(_finite_float(volatility), 0.0)
    beta = max(_finite_float(beta, 1.0), 0.1)

    daily_volatility = volatility / math.sqrt(252) if volatility > 0 else 0.0
    stress_test_scenarios = {
        "market_crash": min(-0.20 * beta, -0.10),
        "volatility_shock": -max(0.05, daily_volatility * 3),
        "liquidity_gap": -0.08,
    }

    results = {}
    for scenario, decline in stress_test_scenarios.items():
        potential_loss = current_position_value * decline
        portfolio_impact = (
            potential_loss / total_portfolio_value
            if total_portfolio_value > 0
            else 0.0
        )
        results[scenario] = {
            "potential_loss": float(potential_loss),
            "portfolio_impact": float(portfolio_impact),
            "shock": float(decline),
        }
    return results


@agent_endpoint("risk_management", "风险管理专家，评估投资风险并给出风险调整后的交易建议")
def risk_management_agent(state: AgentState):
    """Responsible for risk management"""
    show_workflow_status("风险管理")
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]

    prices_df = prices_to_df(data["prices"])

    if len(prices_df) < 2:
        message_content = {
            "max_position_size": 0.0,
            "risk_score": 10,
            "trading_action": "hold",
            "risk_metrics": {
                "volatility": 0.0,
                "value_at_risk_95": 0.0,
                "conditional_value_at_risk_95": 0.0,
                "max_drawdown": 0.0,
                "market_risk_score": 10,
                "liquidity_risk": {
                    "score": 0,
                    "signal": "unknown",
                    "avg_daily_amount": 0.0,
                    "avg_turnover": 0.0,
                    "position_to_amount_ratio": 0.0,
                },
                "t1_settlement": calculate_t1_settlement_constraint(portfolio),
                "beta_risk": calculate_beta_risk(1.0),
                "stress_test_results": {},
            },
            "debate_analysis": {},
            "reasoning": "价格数据不可用，默认持有。",
        }
        message = HumanMessage(
            content=json.dumps(message_content),
            name="risk_management_agent",
        )
        if show_reasoning:
            show_agent_reasoning(message_content, "风险管理")
            state["metadata"]["agent_reasoning"] = message_content
        show_workflow_status("风险管理", "completed")
        return {
            "messages": [message],
            "data": {
                **data,
                "risk_analysis": message_content
            },
            "metadata": state["metadata"],
        }

    # Fetch debate room message instead of individual analyst messages
    debate_message = next(
        msg for msg in state["messages"] if msg.name == "debate_room_agent")

    try:
        debate_results = json.loads(debate_message.content)
    except Exception as e:
        debate_results = ast.literal_eval(debate_message.content)

    # 1. Calculate Risk Metrics
    returns = prices_df['close'].pct_change().dropna()
    daily_vol = returns.std()
    # Annualized volatility approximation
    volatility = daily_vol * (252 ** 0.5)

    # 计算波动率的历史分布
    rolling_std = returns.rolling(window=120).std() * (252 ** 0.5)
    volatility_mean = rolling_std.mean()
    volatility_std = rolling_std.std()
    volatility_percentile = (
        (volatility - volatility_mean) / volatility_std
        if volatility_std and not math.isnan(volatility_std)
        else 0.0
    )

    # Simple historical VaR at 95% confidence
    var_95 = returns.quantile(0.05)
    cvar_95 = calculate_expected_shortfall(returns, confidence=0.95)
    # 使用60天窗口计算最大回撤
    max_drawdown = (
        prices_df['close'] / prices_df['close'].rolling(window=60).max() - 1).min()
    max_drawdown = _finite_float(max_drawdown)

    try:
        beta = calculate_beta(data.get("ticker", ""))
    except Exception:
        beta = 1.0
    beta_risk = calculate_beta_risk(beta)

    # 2. Market Risk Assessment
    market_risk_score = 0

    # Volatility scoring based on percentile
    if volatility_percentile > 1.5:     # 高于1.5个标准差
        market_risk_score += 2
    elif volatility_percentile > 1.0:   # 高于1个标准差
        market_risk_score += 1

    # VaR scoring
    # Note: var_95 is typically negative. The more negative, the worse.
    if var_95 < -0.03:
        market_risk_score += 2
    elif var_95 < -0.02:
        market_risk_score += 1

    # Max Drawdown scoring
    if max_drawdown < -0.20:  # Severe drawdown
        market_risk_score += 2
    elif max_drawdown < -0.10:
        market_risk_score += 1

    # 3. Position Size Limits
    # Consider total portfolio value, not just cash
    current_stock_value = portfolio['stock'] * prices_df['close'].iloc[-1]
    total_portfolio_value = portfolio['cash'] + current_stock_value

    # Start with 25% max position of total portfolio
    base_position_size = total_portfolio_value * 0.25

    if market_risk_score >= 4:
        # Reduce position for high risk
        max_position_size = base_position_size * 0.5
    elif market_risk_score >= 2:
        # Slightly reduce for moderate risk
        max_position_size = base_position_size * 0.75
    else:
        # Keep base size for low risk
        max_position_size = base_position_size

    liquidity_risk = calculate_liquidity_risk(prices_df, max_position_size)
    market_risk_score += liquidity_risk["score"]
    market_risk_score += beta_risk["score"]
    t1_settlement = calculate_t1_settlement_constraint(portfolio)

    # 4. Stress Testing
    current_position_value = current_stock_value
    stress_test_results = calculate_stress_test_results(
        current_position_value=current_position_value,
        total_portfolio_value=portfolio['cash'] + current_position_value,
        volatility=volatility,
        beta=beta,
    )

    # 5. Risk-Adjusted Signal Analysis
    # Consider debate room confidence levels
    bull_confidence = debate_results["bull_confidence"]
    bear_confidence = debate_results["bear_confidence"]
    debate_confidence = debate_results["confidence"]

    # Add to risk score if confidence is low or debate was close
    confidence_diff = abs(bull_confidence - bear_confidence)
    if confidence_diff < 0.1:  # Close debate
        market_risk_score += 1
    if debate_confidence < 0.3:  # Low overall confidence
        market_risk_score += 1

    # Cap risk score at 10
    risk_score = min(round(market_risk_score), 10)

    # 6. Generate Trading Action
    # Consider debate room signal along with risk assessment
    debate_signal = debate_results["signal"]

    if risk_score >= 9:
        trading_action = "hold"
    elif risk_score >= 7:
        trading_action = "reduce"
    else:
        if debate_signal == "bullish" and debate_confidence > 0.5:
            trading_action = "buy"
        elif debate_signal == "bearish" and debate_confidence > 0.5:
            trading_action = "sell" if t1_settlement["can_sell"] else "hold"
        else:
            trading_action = "hold"

    message_content = {
        "max_position_size": float(max_position_size),
        "risk_score": risk_score,
        "trading_action": trading_action,
        "risk_metrics": {
            "volatility": float(volatility),
            "value_at_risk_95": float(var_95),
            "conditional_value_at_risk_95": float(cvar_95),
            "max_drawdown": float(max_drawdown),
            "market_risk_score": market_risk_score,
            "liquidity_risk": liquidity_risk,
            "t1_settlement": t1_settlement,
            "beta_risk": beta_risk,
            "stress_test_results": stress_test_results
        },
        "debate_analysis": {
            "bull_confidence": bull_confidence,
            "bear_confidence": bear_confidence,
            "debate_confidence": debate_confidence,
            "debate_signal": debate_signal
        },
        "reasoning": f"风险评分 {risk_score}/10: 市场风险={market_risk_score}, 波动率={volatility:.2%}, VaR={var_95:.2%}, CVaR={cvar_95:.2%}, 最大回撤={max_drawdown:.2%}, Beta风险={beta_risk['signal']}, 流动性={liquidity_risk['signal']}, T+1可卖={t1_settlement['sellable_stock']}, 辩论信号={debate_signal}"
    }

    # Create the risk management message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="risk_management_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "风险管理")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("风险管理", "completed")
    return {
        "messages": [message],
        "data": {
            **data,
            "risk_analysis": message_content,
            "risk_report": message_content,
        },
        "metadata": state["metadata"],
    }
