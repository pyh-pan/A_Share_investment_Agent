import json
from uuid import uuid4

from src.agents import macro_analyst
from src.agents.macro_analyst import get_macro_news_analysis, score_macro_environment


def test_score_macro_environment_positive_growth_and_liquidity():
    result = score_macro_environment(
        {
            "gdp_growth": {"value": 5.4, "data_available": True},
            "pmi": {"value": 51.8, "data_available": True},
            "m2_growth": {"value": 8.2, "data_available": True},
            "cpi": {"value": 1.8, "data_available": True},
        }
    )

    assert result["macro_environment"] == "positive"
    assert result["impact_on_stock"] == "positive"
    assert result["score"] > 0
    assert result["data_available"] is True
    assert any("PMI" in factor for factor in result["key_factors"])


def test_score_macro_environment_negative_weak_growth_high_inflation():
    result = score_macro_environment(
        {
            "gdp_growth": {"value": 3.2, "data_available": True},
            "pmi": {"value": 48.4, "data_available": True},
            "m2_growth": {"value": 5.1, "data_available": True},
            "cpi": {"value": 4.2, "data_available": True},
        }
    )

    assert result["macro_environment"] == "negative"
    assert result["impact_on_stock"] == "negative"
    assert result["score"] < 0
    assert result["data_available"] is True
    assert any("CPI" in factor for factor in result["key_factors"])


def test_score_macro_environment_empty_indicators_returns_unavailable_neutral():
    result = score_macro_environment({})

    assert result == {
        "macro_environment": "neutral",
        "impact_on_stock": "neutral",
        "score": 0,
        "key_factors": [],
        "data_available": False,
        "reasoning": "宏观指标不可用，返回中性判断。",
    }


def test_get_macro_news_analysis_falls_back_to_deterministic_score_on_invalid_llm(monkeypatch):
    monkeypatch.setattr(macro_analyst, "get_chat_completion_with_validation", lambda *args, **kwargs: "not json")

    macro_indicators = {
        "gdp_growth": {"value": 5.5, "data_available": True},
        "pmi": {"value": 52.0, "data_available": True},
        "m2_growth": {"value": 8.0, "data_available": True},
        "cpi": {"value": 1.5, "data_available": True},
    }

    result = get_macro_news_analysis(
        macro_indicators,
        [{"title": "行业需求改善", "content": "订单增加", "publish_time": "2026-04-26"}],
        f"UNIT-INVALID-LLM-{uuid4()}",
        "unit-test-industry",
    )

    assert result["macro_environment"] == "positive"
    assert result["impact_on_stock"] == "positive"
    assert result["score"] > 0
    assert result["data_available"] is True
    assert result["deterministic_macro_score"]["macro_environment"] == "positive"
    assert "确定性宏观评分" in result["reasoning"]


def test_get_macro_news_analysis_includes_deterministic_score_in_llm_context(monkeypatch):
    calls = {}

    def fake_completion(messages, data_context, validation_mode):
        calls["messages"] = messages
        calls["data_context"] = data_context
        calls["validation_mode"] = validation_mode
        return json.dumps(
            {
                "macro_environment": "neutral",
                "impact_on_stock": "neutral",
                "key_factors": ["LLM综合判断"],
                "reasoning": "LLM reasoning",
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(macro_analyst, "get_chat_completion_with_validation", fake_completion)

    macro_indicators = {
        "gdp_growth": {"value": 5.2, "data_available": True},
        "pmi": {"value": 51.0, "data_available": True},
        "m2_growth": {"value": 7.4, "data_available": True},
        "cpi": {"value": 2.0, "data_available": True},
    }

    result = get_macro_news_analysis(
        macro_indicators,
        [{"title": "政策支持制造业", "content": "政策延续", "publish_time": "2026-04-26"}],
        f"UNIT-VALID-LLM-{uuid4()}",
        "unit-test-industry",
    )

    assert calls["data_context"]["deterministic_macro_score"]["macro_environment"] == "positive"
    assert "确定性宏观评分" in calls["messages"][1]["content"]
    assert result["deterministic_macro_score"]["macro_environment"] == "positive"
