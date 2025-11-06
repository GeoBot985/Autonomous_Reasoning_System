from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder

def test_plan_progress_summary():
    pb = PlanBuilder()
    goal, plan = pb.new_goal_with_plan("Build OCR module")

    # Simulate some progress
    first = plan.steps[0]
    pb.update_step(plan.id, first.id, "complete", "Loaded sample image")

    summary = pb.get_plan_summary(plan.id)
    print("✅ Progress Summary:")
    print(summary["summary_text"])

if __name__ == "__main__":
    test_plan_progress_summary()
