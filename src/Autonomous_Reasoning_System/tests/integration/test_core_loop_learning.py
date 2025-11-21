import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.control.core_loop import CoreLoop

@patch("Autonomous_Reasoning_System.control.core_loop.MemoryInterface")
@patch("Autonomous_Reasoning_System.control.core_loop.start_heartbeat_with_plans")
def test_core_loop_learning_cycle(mock_heartbeat, MockMemoryInterface):
    # Setup mock memory
    mock_memory_instance = MagicMock()
    MockMemoryInterface.return_value = mock_memory_instance

    # Initialize CoreLoop with mocked dependencies
    tyrone = CoreLoop()

    # Mock other components if necessary to avoid side effects or IO
    tyrone.router = MagicMock()
    tyrone.router.route.return_value = {
        "intent": "reflect",
        "pipeline": ["IntentAnalyzer", "ReflectionInterpreter"],
        "reason": "Test reason"
    }

    tyrone.intent_analyzer = MagicMock()
    tyrone.intent_analyzer.analyze.return_value = {"intent": "reflect"}

    tyrone.reflector = MagicMock()
    tyrone.reflector.interpret.return_value = {
        "summary": "Test reflection",
        "insight": "Test insight"
    }

    # Mock Router.resolve directly since run_once calls resolve, not route (except later in PlanExecutor)
    tyrone.router.resolve = MagicMock(return_value={
        "intent": "reflect",
        "family": "reflection",
        "pipeline": ["perform_reflection"],
        "entities": {},
        "analysis_data": {}
    })

    # Run the method
    result = tyrone.run_once("Reflect on how confident you feel about recent progress.")

    # Assertions
    assert "decision" in result
    assert result["decision"]["intent"] == "reflect"
    assert result["reflection_data"]["insight"] == "Test insight"

    # Verify memory store was called
    mock_memory_instance.store.assert_called()
