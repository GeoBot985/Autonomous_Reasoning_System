import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.cognition.router import Router

# When we patch MemoryInterface, we need to make sure we are patching where it is imported in router.py
@patch("Autonomous_Reasoning_System.cognition.router.call_llm")
@patch("Autonomous_Reasoning_System.cognition.router.MemoryInterface")
def test_router_routing(mock_MemoryInterface, mock_call_llm):
    # Mock MemoryInterface
    mock_memory_instance = MagicMock()
    mock_MemoryInterface.return_value = mock_memory_instance
    mock_memory_instance.search_similar.return_value = [] # No memories found by default
    mock_memory_instance.retrieve.return_value = [] # New API name

    dispatcher = MagicMock()
    router = Router(dispatcher)

    # Test Case 1: Deterministic Planner
    # Matches "plan" in regex
    res1 = router.resolve("Create a plan to test camera")
    assert res1['intent'] == "plan"
    # Pipeline check might depend on implementation details (tool names)
    assert "plan_builder" in res1['pipeline'] or "PlanBuilder" in res1['pipeline']

    # Test Case 2: Deterministic Reflection
    # "Reflect on my progress" -> contains "reflect" -> "reflect"
    res2 = router.resolve("Reflect on my progress")
    assert res2['intent'] == "reflect"
    assert "reflector" in res2['pipeline']

    # Test Case 3: LLM Based
    mock_call_llm.return_value = '{"intent": "chat", "pipeline": ["context_adapter"], "reason": "Chatting"}'
    res3 = router.resolve("I like turtles.")

    assert res3['intent'] == "chat"
    assert "context_adapter" in res3['pipeline']

    # Test Case 4: JSON Failure fallback
    mock_call_llm.return_value = "This is not JSON"
    res4 = router.resolve("Something weird")

    # Fallback might be query/chat depending on implementation
    assert res4['intent'] in ["query", "chat", "unknown"]
