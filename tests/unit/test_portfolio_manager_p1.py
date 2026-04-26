from src.agents.portfolio_manager import (
    build_forced_decision,
    enrich_prompt_with_memory,
    is_valid_decision_json,
)


def test_build_forced_decision_uses_base_decision():
    result = build_forced_decision(
        engine_result={"base_decision": "buy", "weighted_score": 0.4},
        agent_signals={"technical": {"signal": "bullish", "confidence": "80%"}},
    )

    assert result["action"] == "buy"
    assert result["reasoning"]


def test_build_forced_decision_contains_required_fields():
    result = build_forced_decision(
        engine_result={"base_decision": "hold", "weighted_score": 0.0},
        agent_signals={},
    )

    assert set(result) >= {
        "action",
        "quantity",
        "confidence",
        "target_price",
        "risk_score",
        "agent_signals",
        "reasoning",
    }


def test_build_forced_decision_normalizes_agent_signals():
    result = build_forced_decision(
        engine_result={"base_decision": "sell", "weighted_score": -0.3},
        agent_signals={"macro": {"signal": "bearish", "confidence": 0.7}},
    )
    names = {item["agent_name"] for item in result["agent_signals"]}

    assert "macro_analyst_agent" in names
    assert "macro_news_agent" in names


def test_enrich_prompt_with_memory_appends_memory_when_available():
    prompt = enrich_prompt_with_memory("base", "memory text")

    assert "base" in prompt
    assert "memory text" in prompt


def test_enrich_prompt_with_memory_ignores_blank_memory():
    assert enrich_prompt_with_memory("base", "  ") == "base"


def test_is_valid_decision_json_rejects_non_dict():
    assert is_valid_decision_json(["not", "dict"]) is False


def test_is_valid_decision_json_rejects_missing_required_fields():
    assert is_valid_decision_json({"action": "buy"}) is False


def test_is_valid_decision_json_accepts_complete_payload():
    assert is_valid_decision_json(
        {
            "action": "买入",
            "quantity": 0,
            "confidence": 0.5,
            "target_price": None,
            "risk_score": 0.5,
            "agent_signals": [],
            "reasoning": "fallback",
        }
    ) is True
