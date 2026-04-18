from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.utils.api_utils import agent_endpoint, log_llm_interaction
import json
import logging

logger = logging.getLogger('researcher_bull')


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


@agent_endpoint("researcher_bull", "看多研究员，基于分析师报告和辩论历史生成看多论点并反驳看空观点")
def researcher_bull_agent(state: AgentState):
    show_workflow_status("看多研究员")
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]

    technical = data.get("technical_report", {})
    fundamentals = data.get("fundamentals_report", {})
    sentiment = data.get("sentiment_report", {})
    valuation = data.get("valuation_report", {})

    reports_summary = f"""技术分析: 信号={technical.get('signal', 'N/A')}, 置信度={technical.get('confidence', 'N/A')}
基本面分析: 信号={fundamentals.get('signal', 'N/A')}, 置信度={fundamentals.get('confidence', 'N/A')}
情绪分析: 信号={sentiment.get('signal', 'N/A')}, 置信度={sentiment.get('confidence', 'N/A')}
估值分析: 信号={valuation.get('signal', 'N/A')}, 置信度={valuation.get('confidence', 'N/A')}"""

    debate_state = data.get("debate_state", {})
    debate_history = debate_state.get("combined_history", "")
    opponent_last = debate_state.get("current_response", "")
    round_count = debate_state.get("round_count", 0)

    if round_count == 0:
        prompt = f"""你是一位看涨分析师，负责为股票建立强有力的看多论证。

⚠️ 严格要求：
- 必须引用具体数据支持你的观点（来自以下分析师报告）
- 不得捏造数据或脱离事实
- 如果某些数据不利于看多，需承认但解释为何不影响整体判断

可用分析师报告：
{reports_summary}

请从以下角度构建看多论点：
1. 增长潜力：基于基本面数据
2. 技术信号：基于技术指标
3. 市场情绪：基于情绪分析
4. 估值合理性：基于估值分析

请用中文回答，输出JSON格式：
{{
    "argument": "你的看多论点（2-3句话）",
    "evidence": ["引用的具体数据1", "引用的具体数据2"],
    "confidence": 0.0到1.0之间的数字,
    "rebuttal_targets": []
}}"""
    else:
        prompt = f"""你是一位看涨分析师，现在需要针对看空方的论点进行反驳。

⚠️ 严格要求：
- 必须引用具体数据反驳看空方的每个论点
- 不得捏造数据或脱离事实强行反驳
- 如果看空方某个论点合理，需承认并解释为何整体仍看多

可用分析师报告：
{reports_summary}

辩论历史：
{debate_history}

看空方最新论点：
{opponent_last}

请逐点反驳看空方的论点，并强化看多立场。输出JSON格式：
{{
    "argument": "你的反驳论点（2-3句话）",
    "evidence": ["引用的具体数据1", "引用的具体数据2"],
    "confidence": 0.0到1.0之间的数字,
    "rebuttal_targets": ["看空方论点1→你的反驳", "看空方论点2→你的反驳"]
}}"""

    from src.tools.openrouter_config import get_chat_completion_with_validation
    content = None
    try:
        data_context = {
            "technical": technical,
            "fundamentals": fundamentals,
            "sentiment": sentiment,
            "valuation": valuation,
        }
        response = get_chat_completion_with_validation(
            messages=[{"role": "user", "content": prompt}],
            data_context=data_context,
            validation_mode="force",
        )
        if response:
            content = _parse_llm_json(response)
    except Exception as e:
        logger.warning(f"看多研究员 LLM 调用失败: {e}")

    if content is None:
        content = {
            "argument": "基于综合分析，看多观点成立",
            "evidence": [],
            "confidence": 0.5,
            "rebuttal_targets": [],
        }

    argument_text = f"[Round {round_count + 1}] Bull Analyst: {content.get('argument', '')}"
    new_combined = (debate_history + "\n" + argument_text).strip() if debate_history else argument_text
    new_bull = (debate_state.get("bull_history", "") + "\n" + argument_text).strip()
    new_debate_state = {
        "combined_history": new_combined,
        "bull_history": new_bull,
        "bear_history": debate_state.get("bear_history", ""),
        "current_response": argument_text,
        "round_count": round_count + 1,
    }

    message_content = {
        "perspective": "bullish",
        "confidence": content.get("confidence", 0.5),
        "argument": content.get("argument", ""),
        "evidence": content.get("evidence", []),
        "rebuttal_targets": content.get("rebuttal_targets", []),
        "thesis_points": [content.get("argument", "")],
        "reasoning": content.get("argument", ""),
    }

    message = HumanMessage(
        content=json.dumps(message_content, ensure_ascii=False),
        name="researcher_bull_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "看多研究员")
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("看多研究员", "completed")
    return {
        "messages": [message],
        "data": {
            **data,
            "bull_report": message_content,
            "debate_state": new_debate_state,
        },
        "metadata": state["metadata"],
    }
