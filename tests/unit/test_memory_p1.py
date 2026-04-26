from src.tools.memory import build_outcome_reflection, update_decision_outcome


def test_build_outcome_reflection_marks_successful_buy_correct():
    result = build_outcome_reflection("buy", actual_return=0.083, days_held=20)

    assert result["was_correct"] is True
    assert "buy" in result["reflection"]
    assert "+8.3%" in result["reflection"]
    assert "20 days" in result["reflection"]
    assert "confirmed" in result["reflection"]


def test_build_outcome_reflection_marks_bad_hold_incorrect():
    result = build_outcome_reflection("hold", actual_return=-0.12, days_held=15)

    assert result["was_correct"] is False
    assert "hold" in result["reflection"]
    assert "-12.0%" in result["reflection"]
    assert "missed risk control" in result["reflection"]


def test_update_decision_outcome_stores_reflection(monkeypatch):
    calls = []

    class FakeMemory:
        def store_reflection(
            self,
            ticker,
            date,
            situation_summary,
            reflection,
            was_correct,
            actual_return=0,
        ):
            calls.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "situation_summary": situation_summary,
                    "reflection": reflection,
                    "was_correct": was_correct,
                    "actual_return": actual_return,
                }
            )
            return True

    monkeypatch.setattr("src.tools.memory.get_memory", lambda: FakeMemory())

    assert (
        update_decision_outcome(
            ticker="301155",
            date="2026-04-26",
            situation_summary="技术:bullish 基本面:neutral",
            decision="sell",
            actual_return=-0.06,
            days_held=10,
        )
        is True
    )

    assert calls == [
        {
            "ticker": "301155",
            "date": "2026-04-26",
            "situation_summary": "技术:bullish 基本面:neutral",
            "reflection": calls[0]["reflection"],
            "was_correct": True,
            "actual_return": -0.06,
        }
    ]
    assert "sell" in calls[0]["reflection"]
    assert "-6.0%" in calls[0]["reflection"]


def test_update_decision_outcome_never_raises_when_memory_unavailable(monkeypatch):
    def unavailable_memory():
        raise RuntimeError("chromadb unavailable")

    monkeypatch.setattr("src.tools.memory.get_memory", unavailable_memory)

    assert (
        update_decision_outcome(
            ticker="301155",
            date="2026-04-26",
            situation_summary="state",
            decision="buy",
            actual_return=0.03,
            days_held=5,
        )
        is False
    )
