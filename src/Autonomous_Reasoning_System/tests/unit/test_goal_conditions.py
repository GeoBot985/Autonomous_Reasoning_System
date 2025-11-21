import pytest
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder

def test_goal_conditions(mock_plan_builder):
    # Use mock_plan_builder fixture which injects memory storage
    pb = mock_plan_builder
    goal = pb.new_goal("Build OCR module")

    print("Goal:", goal.text)
    print("Success:", goal.success_criteria)
    print("Failure:", goal.failure_criteria)

    assert goal.text == "Build OCR module"
    assert hasattr(goal, "success_criteria")
    assert hasattr(goal, "failure_criteria")
