from src.agents.sentiment import _temporal_decay_weights


def test_temporal_decay_weights_normalized():
    news = [
        {"publish_time": "2026-04-18 10:00:00"},
        {"publish_time": "2026-04-17 10:00:00"},
        {"publish_time": "2026-04-14 10:00:00"},
    ]
    weights = _temporal_decay_weights(news, half_life_days=3)
    assert abs(sum(weights) - 1.0) < 1e-8
    assert weights[0] >= weights[1] >= weights[2]
