# tests/test_self_validator.py
from datetime import datetime
from Autonomous_Reasoning_System.cognition.self_validator import SelfValidator


def test_self_validator_basic():
    # SelfValidator doesn't use external dependencies in __init__, so we don't need to patch anything
    # unless it's importing something we haven't seen (checked code, it seems clean).
    # The previous failure was because I patched 'get_memory_storage' which doesn't exist in that module.

    sv = SelfValidator()

    # Positive case
    res1 = sv.evaluate(
        input_text="Summarize today's tasks",
        output_text="Task summary completed.",
        meta={"intent": "reflect", "confidence": 0.9}
    )
    assert res1["success"] and res1["feeling"] == "positive"

    # Neutral case
    res2 = sv.evaluate(
        input_text="Maybe check logs?",
        output_text="Logs reviewed but unsure.",
        meta={"intent": "analyze", "confidence": 0.6}
    )
    assert res2["feeling"] == "neutral"

    # Negative case (error)
    res3 = sv.evaluate(
        input_text="Run report",
        output_text="Sorry, I failed to load data.",
        meta={"intent": "execute", "confidence": 0.4, "error": "File not found"}
    )
    assert not res3["success"] and res3["feeling"] == "negative"

    # Trend summary
    summary = sv.summary()
    assert "avg_conf" in summary
    assert "trend" in summary
    print("✅ SelfValidator test passed:", summary)


if __name__ == "__main__":
    test_self_validator_basic()
