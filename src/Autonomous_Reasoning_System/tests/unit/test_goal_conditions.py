import pytest
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder

def test_goal_conditions():
    pb = PlanBuilder()
    goal = pb.new_goal("Build OCR module")

    print("Goal:", goal.text)
    print("Success:", goal.success_criteria)
    print("Failure:", goal.failure_criteria)

    assert goal.text == "Build OCR module"
    assert hasattr(goal, "success_criteria")
    assert hasattr(goal, "failure_criteria")
