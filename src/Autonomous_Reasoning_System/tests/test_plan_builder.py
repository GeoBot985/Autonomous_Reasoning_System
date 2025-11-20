import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder, Goal, Plan, Step

@patch("Autonomous_Reasoning_System.planning.plan_builder.get_memory_storage")
def test_plan_builder_scaffold(mock_get_memory_storage):
    # Mock memory
    mock_memory = MagicMock()
    mock_get_memory_storage.return_value = mock_memory

    pb = PlanBuilder()

    # Test goal creation
    goal = pb.new_goal("Build OCR module")
    assert goal.text == "Build OCR module"
    assert goal.id is not None

    # Test plan building
    plan = pb.build_plan(goal, [
        "Load image",
        "Run OCR",
        "Store extracted text"
    ])

    assert len(plan.steps) == 3
    assert plan.steps[0].description == "Load image"

    print("✅ Goal created:", goal.text)
    print("✅ Plan steps:")
    for s in plan.steps:
        print("  -", s.description)

    # Simulate progress
    step = plan.next_step()
    assert step.description == "Load image"

    pb.update_step(plan.id, step.id, "complete", "image loaded")
    assert plan.steps[0].status == "complete"

    # Verify memory logging
    mock_memory.add_memory.assert_called()

    print("Progress →", [(s.description, s.status) for s in plan.steps])
