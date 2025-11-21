# tests/test_learning_manager.py
import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.cognition.learning_manager import LearningManager

def test_learning_manager_pipeline():
    # Setup mock memory
    mock_memory = MagicMock()

    lm = LearningManager(memory_storage=mock_memory)

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
    # Should call memory.add_memory
    mock_memory.add_memory.assert_called()

    # Mock get_all_memories to test drift correction
    import pandas as pd
    mock_memory.get_all_memories.return_value = pd.DataFrame()

    drift = lm.perform_drift_correction()
    print("✅ Drift correction:", drift)

    assert isinstance(drift, str)
