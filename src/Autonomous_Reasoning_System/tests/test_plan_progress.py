import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder

@patch("Autonomous_Reasoning_System.planning.plan_builder.get_memory_storage")
def test_plan_progress_summary(mock_get_memory_storage):
    # Mock memory
    mock_memory = MagicMock()
    mock_get_memory_storage.return_value = mock_memory

    pb = PlanBuilder()

    # Mock decompose for predictable steps
    with patch.object(pb, 'decompose_goal', return_value=["Step 1", "Step 2", "Step 3"]):
        goal, plan = pb.new_goal_with_plan("Build OCR module")

        # Simulate some progress
        first = plan.steps[0]
        pb.update_step(plan.id, first.id, "complete", "Loaded sample image")

        summary = pb.get_plan_summary(plan.id)
        print("✅ Progress Summary:")
        print(summary["summary_text"])

        assert summary["completed_steps"] == 1
        assert summary["total_steps"] == 3
        assert "Build OCR module" in summary["title"]
        assert "Step 2" in summary["current_step"]
