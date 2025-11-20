import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.control.core_loop import CoreLoop

@patch("Autonomous_Reasoning_System.control.core_loop.Router")
@patch("Autonomous_Reasoning_System.control.core_loop.IntentAnalyzer")
@patch("Autonomous_Reasoning_System.control.core_loop.MemoryInterface")
@patch("Autonomous_Reasoning_System.control.core_loop.PlanBuilder")
@patch("Autonomous_Reasoning_System.control.core_loop.SelfValidator")
@patch("Autonomous_Reasoning_System.control.core_loop.LearningManager")
@patch("Autonomous_Reasoning_System.control.core_loop.ReflectionInterpreter")
@patch("Autonomous_Reasoning_System.control.core_loop.ConfidenceManager")
@patch("Autonomous_Reasoning_System.control.core_loop.start_heartbeat_with_plans")
def test_core_loop_integration(
    mock_start_heartbeat,
    mock_ConfidenceManager,
    mock_ReflectionInterpreter,
    mock_LearningManager,
    mock_SelfValidator,
    mock_PlanBuilder,
    mock_MemoryInterface,
    mock_IntentAnalyzer,
    mock_Router
):
    # Setup mocks
    mock_router_inst = MagicMock()
    mock_Router.return_value = mock_router_inst

    mock_memory_inst = MagicMock()
    mock_MemoryInterface.return_value = mock_memory_inst

    mock_validator_inst = MagicMock()
    mock_SelfValidator.return_value = mock_validator_inst
    mock_validator_inst.evaluate.return_value = {"success": True, "feeling": "positive", "confidence": 0.9}

    mock_reflector_inst = MagicMock()
    mock_ReflectionInterpreter.return_value = mock_reflector_inst

    loop = CoreLoop()

    # Test case 1: Reflection
    # The router returns "pipeline", which is a list of strings.
    # The loop iterates over these strings.
    mock_router_inst.route.return_value = {"intent": "reflect", "pipeline": ["ReflectionInterpreter"], "reason": "User asked"}
    mock_memory_inst.recall.return_value = "Past memories..."

    mock_reflector_inst.interpret.return_value = {"insight": "Reflection Output"}

    result = loop.run_once("Reflect on work")

    assert result["reflection_data"] == {"insight": "Reflection Output"}

    # Test case 2: Execution (ContextAdapter in this case as "pipeline" contains "ContextAdapter"?)
    # Actually, let's check what `plan_execute` pipeline usually maps to.
    # If it maps to "ContextAdapter", we need to mock that.

    mock_router_inst.route.return_value = {"intent": "execute", "pipeline": ["ContextAdapter"], "reason": "User asked"}

    with patch("Autonomous_Reasoning_System.llm.context_adapter.ContextAdapter") as mock_ContextAdapter:
        mock_adapter_inst = MagicMock()
        mock_ContextAdapter.return_value = mock_adapter_inst
        mock_adapter_inst.run.return_value = "Execution Result"

        result = loop.run_once("Do something")

        assert "Execution Result" in result["summary"]
