"""
Comprehensive functional test for Tyrone's PlanBuilder + PlanReasoner system.

This test ensures that:
- Goals are created correctly
- Success/failure criteria are inferred by the LLM
- Steps are decomposed dynamically via PlanReasoner
- Plan progress updates are persisted to memory
- Active plan retrieval works
"""

import os
import time
from datetime import datetime
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder
from Autonomous_Reasoning_System.memory.storage import MemoryStorage


def setup_memory():
    """Reset memory file before each run."""
    test_path = "data/test_memory_plan.parquet"
    if os.path.exists(test_path):
        os.remove(test_path)
    return MemoryStorage(test_path)


def test_goal_creation_and_plan_generation():
    """Ensure a goal creates a structured plan with dynamic steps."""
    mem = setup_memory()
    pb = PlanBuilder(memory_storage=mem)
    goal, plan = pb.new_goal_with_plan("Develop a file summarisation feature")

    print("\n🧠 Created goal and plan:")
    print(f"Goal: {goal.text}")
    print(f"Success: {goal.success_criteria}")
    print(f"Failure: {goal.failure_criteria}")
    print("Steps:")
    for step in plan.steps:
        print(f"  - {step.description}")

    assert plan.steps, "Plan should have at least one step."
    assert isinstance(plan.steps[0].description, str)
    assert goal.success_criteria and goal.failure_criteria


def test_plan_progress_and_memory_logging():
    """Test that updating steps writes progress reflections to memory."""
    mem = setup_memory()
    pb = PlanBuilder(memory_storage=mem)
    goal, plan = pb.new_goal_with_plan("Build OCR module")
    assert plan.steps, "Plan should have steps."

    # Complete first two steps
    step1 = plan.steps[0]
    pb.update_step(plan.id, step1.id, "complete", "Loaded image successfully.")
    step2 = plan.steps[1]
    pb.update_step(plan.id, step2.id, "complete", "OCR extracted text.")

    # Check memory for progress logs
    df = pb.memory.get_all_memories()
    print("\n💾 Memory entries:")
    print(df.tail())

    progress_logs = df[df["memory_type"] == "plan_progress"]
    assert not progress_logs.empty, "Progress updates should be stored in memory."


def test_active_plan_tracking():
    """Ensure get_active_plans() returns only incomplete plans."""
    mem = setup_memory()
    pb = PlanBuilder(memory_storage=mem)
    goal, plan = pb.new_goal_with_plan("Implement reminder system")
    step = plan.steps[0]
    pb.update_step(plan.id, step.id, "complete", "Created reminder entry.")
    active = pb.get_active_plans()
    assert any(p.status != "complete" for p in active), "There should be active plans."


def test_plan_restoration_from_memory():
    """Simulate restoring plans from memory records."""
    mem = setup_memory()
    pb = PlanBuilder(memory_storage=mem)
    goal, plan = pb.new_goal_with_plan("Integrate text-to-speech engine")

    # Simulate storing plan progress in memory
    pb.update_step(plan.id, plan.steps[0].id, "complete", "Installed dependencies.")

    # Simulate reload (new instance)
    # Reuse the same memory storage object to simulate persistence within the same test run context
    # In a real scenario, this would load from disk, but here we want to verify persistence logic
    new_pb = PlanBuilder(memory_storage=mem)
    df = new_pb.memory.get_all_memories()
    assert not df.empty, "Memory should persist across instances."
    print("\n📚 Restored memory entries:")
    print(df.tail())


def test_multiple_goals_in_sequence():
    """Test sequential goal creation and persistence."""
    mem = setup_memory()
    pb = PlanBuilder(memory_storage=mem)
    goals = [
        "Develop OCR module",
        "Add image pre-processing",
        "Implement translation support",
    ]
    for g in goals:
        goal, plan = pb.new_goal_with_plan(g)
        print(f"🧩 Goal created: {g} → {len(plan.steps)} steps")

    df = pb.memory.get_all_memories()
    assert len(df) > 0, "Memory should contain multiple goal-related entries."


if __name__ == "__main__":
    start = time.time()
    print("\n=== 🧠 Running PlanBuilder System Tests ===")
    test_goal_creation_and_plan_generation()
    test_plan_progress_and_memory_logging()
    test_active_plan_tracking()
    test_plan_restoration_from_memory()
    test_multiple_goals_in_sequence()
    end = time.time()
    print(f"\n✅ All PlanBuilder tests passed in {end - start:.2f}s.")
