import pytest
import tempfile
import shutil
import os
import pandas as pd
from unittest.mock import MagicMock
from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface
from Autonomous_Reasoning_System.memory.persistence import PersistenceService
from Autonomous_Reasoning_System.control.goal_manager import GoalManager
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder
from Autonomous_Reasoning_System.planning.plan_executor import PlanExecutor

@pytest.fixture
def temp_persistence():
    temp_dir = tempfile.mkdtemp()
    service = PersistenceService(data_dir=temp_dir)
    yield service
    shutil.rmtree(temp_dir)

def test_goals_lifecycle(temp_persistence):
    # Setup
    # We need to mock PlanBuilder and PlanExecutor partially or rely on their defaults
    # Since PlanBuilder requires LLM, we should mock it or use a dummy one if possible
    # But CoreLoop uses real components.

    # Let's rely on MemoryInterface + GoalManager logic primarily

    # 1. Initialize Memory Interface with temp persistence
    # We need to monkeypatch get_persistence_service to return our temp instance

    # Hack to force singleton reload or reset
    PersistenceService._instance = temp_persistence

    memory = MemoryInterface()

    # Mock PlanBuilder/Executor for GoalManager
    mock_plan_builder = MagicMock(spec=PlanBuilder)
    mock_dispatcher = MagicMock()
    mock_router = MagicMock()

    # Setup mock plan return
    class MockPlan:
        def __init__(self):
            self.steps = [MagicMock(id="s1", description="Step 1"), MagicMock(id="s2", description="Step 2")]
            self.id = "p1"

    mock_plan_builder.new_goal_with_plan.return_value = ("Mock Goal", MockPlan())

    # Setup mock executor return
    mock_router.resolve.return_value = {"pipeline": ["mock_tool"]}
    mock_dispatcher.run_pipeline.return_value = {"status": "success", "data": {"summary": "Step done"}}

    goal_manager = GoalManager(memory, mock_plan_builder, mock_dispatcher, mock_router)

    # 2. Create a Goal
    goal_id = goal_manager.create_goal("Build a spaceship")
    assert goal_id is not None

    # Verify it's in active goals
    active = memory.get_active_goals()
    assert len(active) == 1
    assert active.iloc[0]['text'] == "Build a spaceship"
    assert active.iloc[0]['status'] == "pending"

    # 3. Check Goals (First Pass - Should Plan)
    summary = goal_manager.check_goals()
    print(f"Check 1 Summary: {summary}")
    assert "Planned 2 steps" in summary

    # Verify status updated to active and steps added
    goal = memory.get_goal(goal_id)
    assert goal['status'] == "active"
    assert "Step 1" in goal['steps']

    # 4. Check Goals (Second Pass - Execute Step 1)
    summary = goal_manager.check_goals()
    print(f"Check 2 Summary: {summary}")
    assert "Executed step 'Step 1'" in summary

    # Verify step 1 completed
    goal = memory.get_goal(goal_id)
    # DuckDB 'steps' column is VARCHAR, so it's a string.
    import json
    if isinstance(goal['steps'], str):
        steps = json.loads(goal['steps'])
    else:
        steps = goal['steps']

    assert steps[0]['status'] == "completed"
    assert steps[1]['status'] == "pending"

    # 5. Check Goals (Third Pass - Execute Step 2)
    summary = goal_manager.check_goals()
    print(f"Check 3 Summary: {summary}")
    assert "Executed step 'Step 2'" in summary

    # 6. Check Goals (Fourth Pass - Completion)
    summary = goal_manager.check_goals()
    print(f"Check 4 Summary: {summary}")
    assert "marked as completed" in summary

    # Verify goal completed
    goal = memory.get_goal(goal_id)
    assert goal['status'] == "completed"

    # 7. Verify Persistence
    memory.save()

    # Create new memory interface to verify loading
    # Re-initialize MemoryInterface
    memory2 = MemoryInterface()
    loaded_goals = memory2.get_active_goals()
    assert len(loaded_goals) == 0 # Should be empty as goal is completed

    all_goals = memory2.storage.get_all_goals()
    assert len(all_goals) == 1
    assert all_goals.iloc[0]['status'] == "completed"

if __name__ == "__main__":
    pytest.main([__file__])
