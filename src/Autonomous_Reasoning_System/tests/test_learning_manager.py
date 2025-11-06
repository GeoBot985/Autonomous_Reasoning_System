# tests/test_learning_manager.py
from ..cognition.learning_manager import LearningManager

def test_learning_manager_pipeline():
    lm = LearningManager()

    # Ingest sample experiences
    sample_data = [
        {"success": True, "feeling": "positive", "reason": "Goal achieved", "intent": "reflect", "confidence": 0.9},
        {"success": True, "feeling": "neutral", "reason": "Okay result", "intent": "analyze", "confidence": 0.6},
        {"success": False, "feeling": "negative", "reason": "Error encountered", "intent": "execute", "confidence": 0.3},
    ]
    for r in sample_data:
        lm.ingest(r)

    summary = lm.summarise_recent(window_minutes=120)
    print("✅ Summary:", summary["summary"])

    drift = lm.perform_drift_correction()
    print("✅ Drift correction:", drift)


if __name__ == "__main__":
    test_learning_manager_pipeline()
