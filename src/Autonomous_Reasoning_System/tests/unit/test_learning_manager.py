# tests/test_learning_manager.py
import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.cognition.learning_manager import LearningManager

@patch("Autonomous_Reasoning_System.cognition.learning_manager.get_memory_storage")
def test_learning_manager_pipeline(mock_get_memory_storage):
    # Setup mock memory
    mock_memory = MagicMock()
    mock_get_memory_storage.return_value = mock_memory

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

    assert "summary" in summary
    assert "positive" in summary["summary"] or "negative" in summary["summary"] or "neutral" in summary["summary"]

    drift = lm.perform_drift_correction()
    print("✅ Drift correction:", drift)

    assert isinstance(drift, str)
