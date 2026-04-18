from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import re
from src.utils.logging_config import setup_logger

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.openrouter_config import get_chat_completion, get_chat_completion_cached
from src.utils.api_utils import agent_endpoint, log_llm_interaction

# 初始化 logger
logger = setup_logger('portfolio_management_agent')

##### Portfolio Management Agent #####

# ──────────────────────────────────────────────
# Signal Weighting Engine — deterministic scoring
# ──────────────────────────────────────────────

SIGNAL_WEIGHTS = {
    "valuation": 0.30,
    "fundamentals": 0.25,
    "technical": 0.20,
    "macro": 0.15,
    "sentiment": 0.10,
}


class SignalWeightingEngine:
    """Deterministic signal combination. LLM NEVER touches the weights."""

    SIGNAL_SCORES = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}

    @staticmethod
    def parse_confidence(conf) -> float:
        """Normalize confidence to 0-1 float from various formats."""
        if isinstance(conf, (int, float)):
            val = float(conf)
            val = val / 100 if val > 1 else val
            return min(max(val, 0), 1)
        if isinstance(conf, str):
            conf = conf.replace("%", "").strip()
            try:
                val = float(conf)
                val = val / 100 if val > 1 else val
                return min(max(val, 0), 1)
            except ValueError:
                return 0.5
        return 0.5

    @classmethod
    def compute_weighted_score(cls, signals: dict) -> dict:
        """Compute weighted signal score deterministically.

        Args:
            signals: dict mapping agent_name -> {"signal": str, "confidence": float/str}

        Returns:
            {
                "weighted_score": float (-1 to 1),
                "base_decision": "buy"|"sell"|"hold",
                "individual_scores": {...},
                "weights_used": SIGNAL_WEIGHTS,
            }
        """
        weighted_sum = 0.0
        total_weight = 0.0
        individual_scores = {}

        for agent_name, weight in SIGNAL_WEIGHTS.items():
            sig = signals.get(agent_name, {})
            signal_str = sig.get("signal", "neutral")
            confidence = cls.parse_confidence(sig.get("confidence", 0.5))
            numeric = cls.SIGNAL_SCORES.get(signal_str, 0.0)

            score = numeric * confidence
            weighted_sum += score * weight
            total_weight += weight * confidence

            individual_scores[agent_name] = {
                "signal": signal_str,
                "confidence": confidence,
                "weighted_contribution": round(score * weight, 4),
            }

        normalized_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Decision thresholds
        if normalized_score > 0.15:
            base_decision = "buy"
        elif normalized_score < -0.15:
            base_decision = "sell"
        else:
            base_decision = "hold"

        return {
            "weighted_score": round(normalized_score, 4),
            "base_decision": base_decision,
            "individual_scores": individual_scores,
            "weights_used": SIGNAL_WEIGHTS,
        }


# ──────────────────────────────────────────────
# Signal Validator — enforce deterministic base
# ──────────────────────────────────────────────

class SignalValidator:
    """Ensures LLM output aligns with deterministic computations."""

    @staticmethod
    def validate(llm_output: dict, engine_result: dict) -> dict:
        """Validate and correct LLM output against deterministic scoring.

        If LLM's action deviates significantly from the weighted score,
        force the base decision but allow LLM to add a qualitative note.
        """
        corrections = []
        llm_action = llm_output.get("action", "hold")
        base_decision = engine_result["base_decision"]

        # Normalize LLM action for comparison
        normalized_llm_action = normalize_action(llm_action)

        # Check if LLM decision conflicts with weighted score
        conflicting = (
            (base_decision == "buy" and normalized_llm_action == "sell") or
            (base_decision == "sell" and normalized_llm_action == "buy")
        )

        if conflicting:
            corrections.append(
                f"Action forced: {llm_action} → {base_decision} "
                f"(weighted_score={engine_result['weighted_score']:.2f})"
            )
            llm_output["action"] = base_decision

        # Ensure confidence is within reasonable range
        llm_conf = llm_output.get("confidence", 0.5)
        if not isinstance(llm_conf, (int, float)):
            llm_conf = 0.5

        # Confidence should not be >0.7 if weighted_score is near zero
        if abs(engine_result["weighted_score"]) < 0.1 and llm_conf > 0.7:
            llm_output["confidence"] = 0.5
            corrections.append("Confidence reduced: signals are mixed")

        if corrections:
            llm_output["validation_corrections"] = corrections

        return llm_output


# ──────────────────────────────────────────────
# Action normalization & target price extraction
# ──────────────────────────────────────────────

ACTION_MAP = {
    "buy": "buy", "买入": "buy", "BUY": "buy", "购买": "buy",
    "sell": "sell", "卖出": "sell", "SELL": "sell", "出售": "sell",
    "hold": "hold", "持有": "hold", "HOLD": "hold", "保持": "hold",
    "观望": "hold",
}


def normalize_action(action: str) -> str:
    """Normalize action to English standard."""
    if not isinstance(action, str):
        return "hold"
    return ACTION_MAP.get(action.strip().lower(), ACTION_MAP.get(action, "hold"))


def extract_target_price(text: str, current_price: float = 0) -> float:
    """Extract target price from LLM response text using regex patterns."""
    patterns = [
        r'目标价[位格]?[：:]?\s*[¥￥]?\s*(\d+(?:\.\d+)?)',
        r'目标[：:]?\s*[¥￥]?\s*(\d+(?:\.\d+)?)',
        r'预期[价价]?[位格]?[：:]?\s*[¥￥]?\s*(\d+(?:\.\d+)?)',
        r'[¥￥]\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*元',
        r'看到\s*(\d+(?:\.\d+)?)',
        r'上涨到\s*(\d+(?:\.\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                price = float(match.group(1))
                # Sanity check: target should be within 50% of current price
                if current_price > 0 and abs(price - current_price) / current_price < 0.5:
                    return price
                elif current_price <= 0:
                    return price
            except ValueError:
                continue
    return None


# ──────────────────────────────────────────────
# Signal extraction from messages & data
# ──────────────────────────────────────────────

def _extract_signal_from_content(content_str: str) -> dict:
    """Try to parse signal+confidence from a message content string."""
    try:
        data = json.loads(content_str) if isinstance(content_str, str) else content_str
        signal = data.get("signal", "neutral")
        confidence = data.get("confidence", 0.5)
        return {"signal": signal, "confidence": confidence}
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {"signal": "neutral", "confidence": 0.5}


def _extract_agent_signals(state: AgentState, cleaned_messages: list) -> dict:
    """Extract signal+confidence for each weighted agent from state data and messages.

    Primary source: state["data"] (technical_report, fundamentals_report, etc.)
    Fallback: parse from message content for backward compatibility.
    """
    data = state.get("data", {})

    agent_sources = {
        "valuation": ("valuation_report", "valuation_agent"),
        "fundamentals": ("fundamentals_report", "fundamentals_agent"),
        "technical": ("technical_report", "technical_analyst_agent"),
        "macro": ("macro_analysis_result", "macro_analyst_agent"),
        "sentiment": ("sentiment_report", "sentiment_agent"),
    }

    signals = {}
    for weight_key, (data_key, msg_name) in agent_sources.items():
        report = data.get(data_key)
        if report and isinstance(report, dict):
            sig = report.get("signal")
            conf = report.get("confidence")
            if sig is not None:
                signals[weight_key] = {"signal": sig, "confidence": conf if conf is not None else 0.5}
                continue
            if "overall_signal" in report:
                signals[weight_key] = {"signal": report["overall_signal"], "confidence": report.get("confidence", 0.5)}
                continue

        msg = get_latest_message_by_name(cleaned_messages, msg_name)
        if msg and msg.content:
            signals[weight_key] = _extract_signal_from_content(msg.content)
        else:
            signals[weight_key] = {"signal": "neutral", "confidence": 0.5}

    return signals


# Helper function to get the latest message by agent name


def get_latest_message_by_name(messages: list, name: str):
    for msg in reversed(messages):
        if msg.name == name:
            return msg
    logger.warning(
        f"投资组合管理中未找到 '{name}' 的消息")
    # Return a dummy message object or raise an error, depending on desired handling
    # For now, returning a dummy message to avoid crashing, but content will be None.
    return HumanMessage(content=json.dumps({"signal": "error", "details": f"未找到 {name} 的消息"}), name=name)


@agent_endpoint("portfolio_management", "负责投资组合管理和最终交易决策")
def portfolio_management_agent(state: AgentState):
    """Responsible for portfolio management"""
    agent_name = "portfolio_management_agent"
    logger.info(f"\n--- DEBUG: {agent_name} START ---")

    # Clean and unique messages by agent name, taking the latest if duplicates exist
    unique_incoming_messages = {}
    for msg in state["messages"]:
        unique_incoming_messages[msg.name] = msg

    cleaned_messages_for_processing = list(unique_incoming_messages.values())

    show_workflow_status(f"{agent_name}: --- 正在执行投资组合管理 ---")
    show_reasoning_flag = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]

    # Get messages from other agents using the cleaned list
    technical_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "technical_analyst_agent")
    fundamentals_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "fundamentals_agent")
    sentiment_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "sentiment_agent")
    valuation_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "valuation_agent")
    risk_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "risk_management_agent")
    tool_based_macro_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "macro_analyst_agent")

    # Extract content, handling potential None
    technical_content = technical_message.content if technical_message else json.dumps(
        {"signal": "error", "details": "Technical message missing"})
    fundamentals_content = fundamentals_message.content if fundamentals_message else json.dumps(
        {"signal": "error", "details": "Fundamentals message missing"})
    sentiment_content = sentiment_message.content if sentiment_message else json.dumps(
        {"signal": "error", "details": "Sentiment message missing"})
    valuation_content = valuation_message.content if valuation_message else json.dumps(
        {"signal": "error", "details": "Valuation message missing"})
    risk_content = risk_message.content if risk_message else json.dumps(
        {"signal": "error", "details": "Risk message missing"})
    tool_based_macro_content = tool_based_macro_message.content if tool_based_macro_message else json.dumps(
        {"signal": "error", "details": "Tool-based Macro message missing"})

    # Market-wide news summary from macro_news_agent
    market_wide_news_summary_content = state["data"].get(
        "macro_news_analysis_result", "大盘宏观新闻分析不可用或未提供。")
    macro_news_agent_message_obj = get_latest_message_by_name(
        cleaned_messages_for_processing, "macro_news_agent")

    agent_signals = _extract_agent_signals(state, cleaned_messages_for_processing)
    engine_result = SignalWeightingEngine.compute_weighted_score(agent_signals)

    show_agent_reasoning(
        agent_name,
        f"确定性加权评分: {engine_result['weighted_score']}, 基础决策: {engine_result['base_decision']}"
    )

    state["data"]["signal_weighting"] = engine_result


    ind = engine_result["individual_scores"]
    scoring_block = f"""确定性加权评分（代码计算，不可修改）：
- 估值信号: {ind['valuation']['signal']} (权重30%, 贡献: {ind['valuation']['weighted_contribution']})
- 基本面信号: {ind['fundamentals']['signal']} (权重25%, 贡献: {ind['fundamentals']['weighted_contribution']})
- 技术信号: {ind['technical']['signal']} (权重20%, 贡献: {ind['technical']['weighted_contribution']})
- 宏观信号: {ind['macro']['signal']} (权重15%, 贡献: {ind['macro']['weighted_contribution']})
- 情绪信号: {ind['sentiment']['signal']} (权重10%, 贡献: {ind['sentiment']['weighted_contribution']})
- 综合加权得分: {engine_result['weighted_score']}
- 基础决策: {engine_result['base_decision']}"""

    current_price = state["data"].get("market_data", {}).get("current_price", 0)
    if not current_price:
        realtime = state["data"].get("realtime_data", {})
        if isinstance(realtime, dict):
            current_price = realtime.get("最新价", 0) or realtime.get("current_price", 0)

    system_message_content = f"""你是一位投资组合经理，负责做出最终的交易决策。
你的工作是根据团队的分析做出交易决策，同时严格遵守风险管理约束。

风险管理约束：
- 你必须不超过风险管理指定的最大持仓规模
- 你必须遵循风险管理建议的交易操作（买入/卖出/持有）
- 这些是硬性约束，不能被其他信号覆盖

权重分配（已由代码确定，不可修改）：
1. 估值分析（30% 权重）
2. 基本面分析（25% 权重）
3. 技术分析（20% 权重）
4. 宏观分析（15% 权重）—— 包含两个输入：
   a) 常规宏观环境（来自宏观分析师，工具型）
   b) 每日大盘新闻摘要（来自宏观新闻）
   两者都为外部风险和机会提供背景。
5. 情绪分析（10% 权重）

决策流程：
1. 首先检查风险管理约束
2. 参考确定性加权评分作为基础决策依据
3. 在定性层面进行微调（如风险考量、特殊情况、宏观环境变化）
4. 如果你的最终决策与基础决策冲突，必须在reasoning中详细说明理由

{scoring_block}

你可以在定性层面调整（如风险考量、特殊情况），但如果你的最终决策与基础决策冲突，请在reasoning中说明理由。

请在输出 JSON 中提供以下内容：
- "action": "buy" | "sell" | "hold"
- "quantity": <正整数>
- "confidence": <0到1之间的浮点数>
- "target_price": <目标价格，浮点数或null>
- "risk_score": <0到1之间的风险评分>
- "agent_signals": <代理信号列表，包含代理名称、信号（bullish | bearish | neutral）及其置信度>
  重要：你的 'agent_signals' 列表必须包含以下条目：
    - "technical_analysis"
    - "fundamental_analysis"
    - "sentiment_analysis"
    - "valuation_analysis"
    - "risk_management"
    - "selected_stock_macro_analysis"（代表来自宏观分析师的工具型宏观输入）
    - "market_wide_news_summary(沪深300指数)"（代表来自宏观新闻的每日新闻摘要输入）
- "reasoning": <简明的决策解释，包括如何权衡所有信号，包括两个宏观输入>

交易规则：
- 永远不要超过风险管理的持仓限制
- 只有在有可用资金时才买入
- 只有在持有股票时才卖出
- 卖出数量必须 ≤ 当前持仓
- 买入数量必须 ≤ 风险管理的最大持仓规模
- 当前股价: {current_price}"""

    system_message = {
        "role": "system",
        "content": system_message_content
    }

    user_message_content = f"""请根据以下团队分析结果做出交易决策。请使用中文进行推理分析。

            技术分析信号: {technical_content}
            基本面分析信号: {fundamentals_content}
            情绪分析信号: {sentiment_content}
            估值分析信号: {valuation_content}
            风险管理信号: {risk_content}
            宏观环境分析（来自宏观分析师）: {tool_based_macro_content}
            每日大盘新闻摘要（来自宏观新闻）:
            {market_wide_news_summary_content}

            确定性加权评分结果:
            {scoring_block}

            当前投资组合:
            现金: {portfolio['cash']:.2f}
            当前持仓: {portfolio['stock']} 股
            当前股价: {current_price}

            请仅输出JSON格式，reasoning字段请使用中文。确保 'agent_signals' 包含系统提示中要求的所有代理。务必填写 target_price 和 risk_score 字段。"""
    user_message = {
        "role": "user",
        "content": user_message_content
    }

    show_agent_reasoning(
        agent_name, f"准备 LLM 调用，输入包含: 技术分析、基本面、情绪、估值、风险、宏观分析、市场新闻 + 确定性加权评分")

    llm_interaction_messages = [system_message, user_message]
    llm_response_content = get_chat_completion_cached(llm_interaction_messages)

    current_metadata = state["metadata"]
    current_metadata["current_agent_name"] = agent_name

    def get_llm_result_for_logging_wrapper():
        return llm_response_content
    log_llm_interaction(state)(get_llm_result_for_logging_wrapper)()

    if llm_response_content is None:
        show_agent_reasoning(
            agent_name, "LLM 调用失败，使用确定性加权评分作为默认决策")
        llm_response_content = json.dumps({
            "action": engine_result["base_decision"],
            "quantity": 0,
            "confidence": 0.5,
            "target_price": None,
            "risk_score": 0.5,
            "agent_signals": [
                {"agent_name": "technical_analysis",
                    "signal": agent_signals.get("technical", {}).get("signal", "neutral"),
                    "confidence": agent_signals.get("technical", {}).get("confidence", 0.0)},
                {"agent_name": "fundamental_analysis",
                    "signal": agent_signals.get("fundamentals", {}).get("signal", "neutral"),
                    "confidence": agent_signals.get("fundamentals", {}).get("confidence", 0.0)},
                {"agent_name": "sentiment_analysis",
                    "signal": agent_signals.get("sentiment", {}).get("signal", "neutral"),
                    "confidence": agent_signals.get("sentiment", {}).get("confidence", 0.0)},
                {"agent_name": "valuation_analysis",
                    "signal": agent_signals.get("valuation", {}).get("signal", "neutral"),
                    "confidence": agent_signals.get("valuation", {}).get("confidence", 0.0)},
                {"agent_name": "risk_management",
                    "signal": "hold", "confidence": 1.0},
                {"agent_name": "macro_analyst_agent",
                    "signal": agent_signals.get("macro", {}).get("signal", "neutral"),
                    "confidence": agent_signals.get("macro", {}).get("confidence", 0.0)},
                {"agent_name": "macro_news_agent",
                    "signal": "unavailable_or_llm_error", "confidence": 0.0}
            ],
            "reasoning": f"LLM API 错误，基于确定性加权评分决策: {engine_result['base_decision']} (得分={engine_result['weighted_score']})"
        })

    try:
        decision_json = json.loads(llm_response_content)
    except json.JSONDecodeError:
        decision_json = {
            "action": engine_result["base_decision"],
            "quantity": 0,
            "confidence": 0.5,
            "target_price": None,
            "risk_score": 0.5,
            "agent_signals": [],
            "reasoning": f"LLM输出解析失败，使用确定性评分决策: {engine_result['base_decision']}"
        }
        llm_response_content = json.dumps(decision_json, ensure_ascii=False)

    decision_json = SignalValidator.validate(decision_json, engine_result)

    decision_json["action"] = normalize_action(decision_json.get("action", "hold"))

    if "target_price" not in decision_json or decision_json["target_price"] is None:
        reasoning_text = decision_json.get("reasoning", "")
        extracted_price = extract_target_price(reasoning_text, current_price)
        if extracted_price is not None:
            decision_json["target_price"] = extracted_price

    if "risk_score" not in decision_json:
        try:
            risk_data = json.loads(risk_content) if isinstance(risk_content, str) else risk_content
            risk_score_val = risk_data.get("risk_score", 0.5)
            if isinstance(risk_score_val, (int, float)) and risk_score_val > 1:
                risk_score_val = risk_score_val / 10.0
            decision_json["risk_score"] = risk_score_val
        except (json.JSONDecodeError, TypeError, AttributeError):
            decision_json["risk_score"] = 0.5

    llm_response_content = json.dumps(decision_json, ensure_ascii=False)

    final_decision_message = HumanMessage(
        content=llm_response_content,
        name=agent_name,
    )

    if show_reasoning_flag:
        show_agent_reasoning(
            agent_name, f"最终 LLM 决策 JSON (已验证): {llm_response_content}")
        if decision_json.get("validation_corrections"):
            show_agent_reasoning(
                agent_name, f"验证修正: {decision_json['validation_corrections']}")

    agent_decision_details_value = {
        "action": decision_json.get("action"),
        "quantity": decision_json.get("quantity"),
        "confidence": decision_json.get("confidence"),
        "target_price": decision_json.get("target_price"),
        "risk_score": decision_json.get("risk_score"),
        "weighted_score": engine_result["weighted_score"],
        "base_decision": engine_result["base_decision"],
        "validation_corrections": decision_json.get("validation_corrections", []),
        "reasoning_snippet": decision_json.get("reasoning", "")[:150] + "..."
    }

    show_workflow_status(f"{agent_name}: --- 投资组合管理完成 ---")

    final_messages_output = [final_decision_message]

    return {
        "messages": final_messages_output,
        "data": state["data"],
        "metadata": {
            **state["metadata"],
            f"{agent_name}_decision_details": agent_decision_details_value,
            "agent_reasoning": llm_response_content
        }
    }


def format_decision(action: str, quantity: int, confidence: float, agent_signals: list, reasoning: str, market_wide_news_summary: str = "未提供",
                    target_price: float = None, risk_score: float = None) -> dict:
    """Format the trading decision into a standardized output format.
    Think in English but output analysis in Chinese."""

    fundamental_signal = next(
        (s for s in agent_signals if s["agent_name"] == "fundamental_analysis"), None)
    valuation_signal = next(
        (s for s in agent_signals if s["agent_name"] == "valuation_analysis"), None)
    technical_signal = next(
        (s for s in agent_signals if s["agent_name"] == "technical_analysis"), None)
    sentiment_signal = next(
        (s for s in agent_signals if s["agent_name"] == "sentiment_analysis"), None)
    risk_signal = next(
        (s for s in agent_signals if s["agent_name"] == "risk_management"), None)
    # Existing macro signal from macro_analyst_agent (tool-based)
    general_macro_signal = next(
        (s for s in agent_signals if s["agent_name"] == "macro_analyst_agent"), None)
    # New market-wide news summary signal from macro_news_agent
    market_wide_news_signal = next(
        (s for s in agent_signals if s["agent_name"] == "macro_news_agent"), None)

    def signal_to_chinese(signal_data):
        if not signal_data:
            return "无数据"
        if signal_data.get("signal") == "bullish":
            return "看多"
        if signal_data.get("signal") == "bearish":
            return "看空"
        return "中性"

    target_price_str = f"¥{target_price:.2f}" if target_price is not None else "未设定"
    risk_score_str = f"{risk_score:.1%}" if risk_score is not None else "未评估"

    detailed_analysis = f"""
====================================
          投资分析报告
====================================

一、策略分析

1. 估值分析 (权重30%):
   信号: {signal_to_chinese(valuation_signal)}
   置信度: {valuation_signal['confidence']*100 if valuation_signal else 0:.0f}%
   要点:
   - DCF估值: {valuation_signal.get('reasoning', {}).get('dcf_analysis', {}).get('details', '无数据') if valuation_signal else '无数据'}
   - 所有者收益法: {valuation_signal.get('reasoning', {}).get('owner_earnings_analysis', {}).get('details', '无数据') if valuation_signal else '无数据'}

2. 基本面分析 (权重25%):
   信号: {signal_to_chinese(fundamental_signal)}
   置信度: {fundamental_signal['confidence']*100 if fundamental_signal else 0:.0f}%
   要点:
   - 盈利能力: {fundamental_signal.get('reasoning', {}).get('profitability_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
   - 增长情况: {fundamental_signal.get('reasoning', {}).get('growth_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
   - 财务健康: {fundamental_signal.get('reasoning', {}).get('financial_health_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
   - 估值水平: {fundamental_signal.get('reasoning', {}).get('price_ratios_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}

3. 技术分析 (权重20%):
   信号: {signal_to_chinese(technical_signal)}
   置信度: {technical_signal['confidence']*100 if technical_signal else 0:.0f}%
   要点:
   - 趋势跟踪: ADX={technical_signal.get('strategy_signals', {}).get('trend_following', {}).get('metrics', {}).get('adx', 0.0) if technical_signal else 0.0:.2f}
   - 均值回归: RSI(14)={technical_signal.get('strategy_signals', {}).get('mean_reversion', {}).get('metrics', {}).get('rsi_14', 0.0) if technical_signal else 0.0:.2f}
   - 动量指标:
     * 1月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_1m', 0.0) if technical_signal else 0.0:.2%}
     * 3月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_3m', 0.0) if technical_signal else 0.0:.2%}
     * 6月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_6m', 0.0) if technical_signal else 0.0:.2%}
   - 波动性: {technical_signal.get('strategy_signals', {}).get('volatility', {}).get('metrics', {}).get('historical_volatility', 0.0) if technical_signal else 0.0:.2%}

4. 宏观分析 (综合权重15%):
   a) 常规宏观分析 (来自 Macro Analyst Agent):
      信号: {signal_to_chinese(general_macro_signal)}
      置信度: {general_macro_signal['confidence']*100 if general_macro_signal else 0:.0f}%
      宏观环境: {general_macro_signal.get(
          'macro_environment', '无数据') if general_macro_signal else '无数据'}
      对股票影响: {general_macro_signal.get(
          'impact_on_stock', '无数据') if general_macro_signal else '无数据'}
      关键因素: {', '.join(general_macro_signal.get(
          'key_factors', ['无数据']) if general_macro_signal else ['无数据'])}

   b) 大盘宏观新闻分析 (来自 Macro News Agent):
      信号: {signal_to_chinese(market_wide_news_signal)}
      置信度: {market_wide_news_signal['confidence']*100 if market_wide_news_signal else 0:.0f}%
      摘要或结论: {market_wide_news_signal.get(
          'reasoning', market_wide_news_summary) if market_wide_news_signal else market_wide_news_summary}

5. 情绪分析 (权重10%):
   信号: {signal_to_chinese(sentiment_signal)}
   置信度: {sentiment_signal['confidence']*100 if sentiment_signal else 0:.0f}%
   分析: {sentiment_signal.get('reasoning', '无详细分析')
                             if sentiment_signal else '无详细分析'}

二、风险评估
风险评分: {risk_signal.get('risk_score', '无数据') if risk_signal else '无数据'}/10
主要指标:
- 波动率: {risk_signal.get('risk_metrics', {}).get('volatility', 0.0)*100 if risk_signal else 0.0:.1f}%
- 最大回撤: {risk_signal.get('risk_metrics', {}).get('max_drawdown', 0.0)*100 if risk_signal else 0.0:.1f}%
- VaR(95%): {risk_signal.get('risk_metrics', {}).get('value_at_risk_95', 0.0)*100 if risk_signal else 0.0:.1f}%
- 市场风险: {risk_signal.get('risk_metrics', {}).get('market_risk_score', '无数据') if risk_signal else '无数据'}/10

三、投资建议
操作建议: {'买入' if action == 'buy' else '卖出' if action == 'sell' else '持有'}
交易数量: {quantity}股
决策置信度: {confidence*100:.0f}%
目标价格: {target_price_str}
综合风险评分: {risk_score_str}

四、决策依据
{reasoning}

==================================="""

    return {
        "action": action,
        "quantity": quantity,
        "confidence": confidence,
        "target_price": target_price,
        "risk_score": risk_score,
        "agent_signals": agent_signals,
        "分析报告": detailed_analysis
    }
