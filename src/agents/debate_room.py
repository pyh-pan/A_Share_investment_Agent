from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.openrouter_config import get_chat_completion, get_chat_completion_with_validation
from src.utils.api_utils import agent_endpoint, log_llm_interaction
import json
import logging

logger = logging.getLogger('debate_room')


def _parse_llm_json(response):
    if response is None:
        return None
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


@agent_endpoint("debate_room", "辩论裁判，评估多空辩论并做出决策")
def debate_room_agent(state: AgentState):
    show_workflow_status("辩论室")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    logger.info("开始评估多空辩论...")

    debate_state = data.get("debate_state", {})
    combined_history = debate_state.get("combined_history", "")
    bull_history = debate_state.get("bull_history", "")
    bear_history = debate_state.get("bear_history", "")

    technical = data.get("technical_report", {})
    fundamentals = data.get("fundamentals_report", {})
    sentiment = data.get("sentiment_report", {})
    valuation = data.get("valuation_report", {})

    reports_summary = f"""技术:{technical.get('signal', 'N/A')} 基本面:{fundamentals.get('signal', 'N/A')} 情绪:{sentiment.get('signal', 'N/A')} 估值:{valuation.get('signal', 'N/A')}"""

    prompt = f"""你是一位中立的辩论裁判，负责评估多空双方的辩论并做出明确决策。

⚠️ 要求：
1. 总结双方最有说服力的2-3个论点
2. 指出双方的逻辑漏洞或数据引用错误
3. 做出明确决策（避免"双方都有道理"的模糊判断）
4. 评估每方论据的数据真实性

分析师报告（用于验证论据真实性）：
{reports_summary}

完整辩论记录：
{combined_history}

请输出JSON格式：
{{
    "signal": "bullish|bearish|neutral",
    "confidence": 0.0到1.0,
    "bull_confidence": 0.0到1.0,
    "bear_confidence": 0.0到1.0,
    "strongest_bull_args": ["最强看多论点1", "最强看多论点2"],
    "strongest_bear_args": ["最强看空论点1", "最强看空论点2"],
    "bull_weaknesses": ["看多方逻辑漏洞1"],
    "bear_weaknesses": ["看空方逻辑漏洞1"],
    "data_accuracy_check": "论据数据真实性评估",
    "reasoning": "决策理由（2-3句话）"
}}"""

    data_context = {
        "technical": technical,
        "fundamentals": fundamentals,
        "sentiment": sentiment,
        "valuation": valuation,
    }

    result = None
    try:
        logger.info("调用 LLM 评估辩论...")
        messages = [
            {"role": "system", "content": "你是一位中立的金融辩论裁判，请使用中文提供分析。"},
            {"role": "user", "content": prompt}
        ]
        response = get_chat_completion_with_validation(
            messages=messages,
            data_context=data_context,
            validation_mode="force",
        )
        if response:
            result = _parse_llm_json(response)
    except Exception as e:
        logger.error(f"LLM 评估辩论失败: {e}")

    if result is None:
        signals = [
            technical.get("signal", "neutral"),
            fundamentals.get("signal", "neutral"),
            sentiment.get("signal", "neutral"),
            valuation.get("signal", "neutral"),
        ]
        bull_count = signals.count("bullish")
        bear_count = signals.count("bearish")
        result = {
            "signal": "bullish" if bull_count > bear_count else "bearish" if bear_count > bull_count else "neutral",
            "confidence": 0.5,
            "bull_confidence": bull_count / max(len(signals), 1),
            "bear_confidence": bear_count / max(len(signals), 1),
            "strongest_bull_args": [],
            "strongest_bear_args": [],
            "bull_weaknesses": [],
            "bear_weaknesses": [],
            "data_accuracy_check": "fallback mode",
            "reasoning": "基于分析师信号简单投票",
        }

    bull_conf = result.get("bull_confidence", 0.5)
    bear_conf = result.get("bear_confidence", 0.5)
    signal = result.get("signal", "neutral")
    confidence = result.get("confidence", 0.5)

    debate_summary = result.get("strongest_bull_args", []) + result.get("strongest_bear_args", [])

    message_content = {
        "signal": signal,
        "confidence": confidence,
        "bull_confidence": bull_conf,
        "bear_confidence": bear_conf,
        "confidence_diff": abs(bull_conf - bear_conf),
        "mixed_confidence_diff": (bull_conf - bear_conf),
        "debate_summary": debate_summary,
        "strongest_bull_args": result.get("strongest_bull_args", []),
        "strongest_bear_args": result.get("strongest_bear_args", []),
        "reasoning": result.get("reasoning", ""),
        "data_accuracy_check": result.get("data_accuracy_check", ""),
    }

    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False),
        name="debate_room_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "辩论室")
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("辩论室", "completed")
    logger.info("辩论室评估完成")
    return {
        "messages": [message],
        "data": {
            **data,
            "debate_analysis": message_content,
            "debate_report": message_content,
            "debate_state": {**debate_state, "judge_decision": message_content},
        },
        "metadata": state["metadata"],
    }
