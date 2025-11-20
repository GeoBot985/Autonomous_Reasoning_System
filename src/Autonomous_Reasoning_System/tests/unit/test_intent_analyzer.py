import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.cognition.intent_analyzer import IntentAnalyzer

@patch("Autonomous_Reasoning_System.cognition.intent_analyzer.call_llm")
def test_intent_analyzer(mock_call_llm):
    # Mock response from LLM.
    mock_call_llm.return_value = '{"intent": "execute", "entities": {}, "reason": "Action oriented", "confidence": 0.95}'

    analyzer = IntentAnalyzer()
    result = analyzer.analyze("Remind me to test the camera")

    assert result['intent'] == 'execute'

    # Test fallback or other cases
    mock_call_llm.return_value = '{"intent": "reflect", "entities": {}, "reason": "Reflection", "confidence": 0.8}'
    result = analyzer.analyze("Reflect on progress")
    assert result['intent'] == 'reflect'

    # Test invalid JSON
    mock_call_llm.return_value = "Not JSON"
    result = analyzer.analyze("Bad input")
    assert result['intent'] == 'unknown'
    assert "Fallback" in result['reason']
