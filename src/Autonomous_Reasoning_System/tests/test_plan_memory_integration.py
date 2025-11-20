import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder

@patch("Autonomous_Reasoning_System.planning.plan_builder.get_memory_storage")
def test_plan_memory_integration(mock_get_memory_storage):
    # Mock memory
    mock_memory = MagicMock()
    mock_get_memory_storage.return_value = mock_memory

    pb = PlanBuilder()

    # Mock PlanReasoner to avoid LLM calls or complex logic if needed,
    # though new_goal_with_plan calls decompose_goal which uses it.
    # For this test we can let it run or mock reasoner if it's slow/fragile.
    # We'll let it run but rely on fallback if LLM unavailable, or mock if it fails.

    # Actually, let's mock decompose_goal to ensure deterministic steps
    with patch.object(pb, 'decompose_goal', return_value=["Step 1", "Step 2"]):
        goal, plan = pb.new_goal_with_plan("Build OCR module")

        assert len(plan.steps) == 2

        # Simulate some progress
        step1 = plan.steps[0]
        pb.update_step(plan.id, step1.id, "complete", "Loaded sample image")

        # Verify memory interaction
        assert mock_memory.add_memory.call_count >= 1

        step2 = plan.steps[1]
        pb.update_step(plan.id, step2.id, "complete", "OCR run successful")

        # Should be called again
        assert mock_memory.add_memory.call_count >= 2

    print("🧠 Memory logging test complete. Check console for embedded + stored entries.")
