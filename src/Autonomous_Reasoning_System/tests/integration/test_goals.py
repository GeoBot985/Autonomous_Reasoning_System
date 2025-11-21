import pytest
import tempfile
import shutil
import os
import json
import logging
from unittest.mock import MagicMock, patch, ANY, PropertyMock
from datetime import datetime
from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface
from Autonomous_Reasoning_System.memory.storage import MemoryStorage
from Autonomous_Reasoning_System.memory.persistence import PersistenceService
from Autonomous_Reasoning_System.control.goal_manager import GoalManager
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder
from Autonomous_Reasoning_System.planning.plan_executor import PlanExecutor
from Autonomous_Reasoning_System.planning.plan_builder import Plan, Step, Goal

# Configure logging to avoid noise
logging.basicConfig(level=logging.CRITICAL)

@pytest.fixture
def temp_persistence():
    temp_dir = tempfile.mkdtemp()
    # Ensure we start fresh
    PersistenceService._instance = None
    service = PersistenceService(data_dir=temp_dir)
    yield service
    shutil.rmtree(temp_dir)
    PersistenceService._instance = None

def test_goals_lifecycle(temp_persistence):
    """
    Integration test for GoalManager lifecycle.
    Uses real MemoryInterface (DuckDB) but mocks PlanBuilder/Executor/Router to avoid LLM/System calls.
    """

    # 1. Setup Memory Interface with isolated DB
    temp_db_path = os.path.join(temp_persistence.data_dir, "test_memory.duckdb")
    storage = MemoryStorage(db_path=temp_db_path)
    memory = MemoryInterface(memory_storage=storage)

    # 2. Mock Dependencies
    mock_plan_builder = MagicMock(spec=PlanBuilder)
    mock_plan_builder.active_plans = {}
    mock_plan_builder.active_goals = {}

    mock_dispatcher = MagicMock()
    mock_router = MagicMock()
    mock_plan_executor = MagicMock(spec=PlanExecutor)

    # 3. Initialize GoalManager
    goal_manager = GoalManager(memory, mock_plan_builder, mock_dispatcher, mock_router, mock_plan_executor)

    # --- Test Goal Creation ---
    goal_id = goal_manager.create_goal("Build a spaceship")
    assert goal_id is not None

    # Verify in DB
    active = memory.get_active_goals()
    assert len(active) == 1
    assert active.iloc[0]['text'] == "Build a spaceship"
    assert active.iloc[0]['status'] == "pending"

    # --- Test Goal Planning (Check 1) ---

    # Use REAL objects for Goal and Plan
    real_goal = Goal(id=goal_id, text="Build a spaceship")

    steps_list = [
        Step(id="s1", description="Step 1"),
        Step(id="s2", description="Step 2")
    ]
    real_plan = Plan(id="plan_123", goal_id=goal_id, title="Build a spaceship", steps=steps_list, status="pending")

    # Configure mocks to return these real objects

    # new_goal_with_plan
    def new_goal_side_effect(text):
        mock_plan_builder.active_plans["plan_123"] = real_plan
        mock_plan_builder.active_goals[goal_id] = real_goal
        return real_goal, real_plan
    mock_plan_builder.new_goal_with_plan.side_effect = new_goal_side_effect

    # decompose_goal
    mock_plan_builder.decompose_goal.return_value = ["Step 1", "Step 2"]

    # build_plan
    # Note: build_plan in GoalManager is called with a new transient Goal object created locally if not careful.
    # But we want it to return our real_plan.
    def build_plan_side_effect(goal_obj, steps_desc):
        mock_plan_builder.active_plans["plan_123"] = real_plan
        return real_plan
    mock_plan_builder.build_plan.side_effect = build_plan_side_effect

    # Ensure initial active_plans is empty so check_goals triggers planning
    mock_plan_builder.active_plans = {}

    summary = goal_manager.check_goals()

    assert "Planned 2 steps" in summary

    # Verify goal updated in memory with PLAN ID, not necessarily steps JSON
    goal_record = memory.get_goal(goal_id)
    assert goal_record['status'] == "active"
    assert goal_record['plan_id'] == "plan_123"
    # assert "Step 1" in goal_record['steps'] # REMOVED: We no longer sync steps to legacy JSON.

    # --- Test Execution Step 1 (Check 2) ---
    # Setup mock executor to execute step 1 successfully
    mock_plan_executor.execute_next_step.return_value = {
        "status": "running",
        "plan_id": "plan_123",
        "step_completed": "Step 1"
    }

    # Need to ensure plan status is not complete so it tries to execute
    real_plan.status = "active"

    summary = goal_manager.check_goals()

    assert "Executed step for goal 'Build a spaceship': Step 1" in summary
    mock_plan_executor.execute_next_step.assert_called_with("plan_123")

    # --- Test Execution Step 2 / Completion (Check 3) ---
    # Setup mock executor to finish plan
    mock_plan_executor.execute_next_step.return_value = {
        "status": "complete",
        "plan_id": "plan_123",
        "summary": "Done"
    }

    summary = goal_manager.check_goals()

    # Note: GoalManager now says "Goal ... completed." or "Executed step..." depending on return.
    # If execute_next_step returns "complete", GoalManager appends "Goal ... completed."
    assert "Goal 'Build a spaceship' completed." in summary

    # GoalManager should verify completion.
    goal_record = memory.get_goal(goal_id)
    assert goal_record['status'] == "completed"

    # --- Test Cleanup (Check 4) ---
    # Set plan to complete so if check_goals runs, it sees complete status
    real_plan.status = "complete"

    summary = goal_manager.check_goals()
    # Should match behavior for completed plan found in check loop
    # Since check_goals iterates active_goals (pending/active), if we marked it completed in DB above, it shouldn't appear in loop!
    # Wait, get_active_goals returns status IN ('pending', 'active').
    # We updated it to 'completed' in the previous step.
    # So check_goals should find NOTHING.
    assert "No active goals" in summary

if __name__ == "__main__":
    pytest.main([__file__])
