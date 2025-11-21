import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.control.core_loop import CoreLoop
from Autonomous_Reasoning_System.planning.plan_builder import Plan, Step

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

    # Setup PlanBuilder mock to return a valid plan so CoreLoop doesn't crash on plan.id
    mock_plan_builder_inst = MagicMock()
    mock_PlanBuilder.return_value = mock_plan_builder_inst

    dummy_plan = Plan(id="p1", goal_id="g1", title="test", steps=[])
    mock_plan_builder_inst.new_goal.return_value = MagicMock(id="g1")
    mock_plan_builder_inst.build_plan.return_value = dummy_plan

    loop = CoreLoop()

    # Test case 1: Reflection
    # The router returns "pipeline", which is a list of strings.
    mock_router_inst.resolve.return_value = {
        "intent": "reflect",
        "family": "reflection",
        "pipeline": ["perform_reflection"],
        "reason": "User asked"
    }
    mock_memory_inst.recall.return_value = "Past memories..."

    mock_reflector_inst.interpret.return_value = {"insight": "Reflection Output"}

    result = loop.run_once("Reflect on work")

    # CoreLoop now uses "reflection" key
    assert result["reflection"] == {"insight": "Reflection Output"}

    # Test case 2: Execution
    mock_router_inst.resolve.return_value = {
        "intent": "execute",
        "family": "tool_execution",
        "pipeline": ["ContextAdapter"],
        "reason": "User asked"
    }

    # Need to mock PlanExecutor behavior since CoreLoop uses it
    # loop.plan_executor is a real instance but with mocked PlanBuilder/Router/Dispatcher
    # We need to make sure loop.plan_executor.execute_plan returns something.

    # Wait, CoreLoop constructs PlanExecutor internally.
    # We can mock the instance on the loop object.
    loop.plan_executor = MagicMock()
    loop.plan_executor.execute_plan.return_value = {
        "status": "complete",
        "summary": {"summary_text": "Done"},
        "final_output": "Execution Result"
    }

    # Need to ensure PlanBuilder returns a plan with steps if we want final_output from steps?
    # No, my fix allows final_output in execute_plan return dict.
    # Let's verify run_once uses it.

    dummy_plan.steps = [Step(id="s1", description="d", result="Execution Result", status="complete")]

    result = loop.run_once("Do something")

    assert "Execution Result" in result["summary"]
