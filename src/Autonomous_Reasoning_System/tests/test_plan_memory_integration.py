from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder

def test_plan_memory_integration():
    pb = PlanBuilder()
    goal, plan = pb.new_goal_with_plan("Build OCR module")

    # Simulate some progress
    step1 = plan.steps[0]
    pb.update_step(plan.id, step1.id, "complete", "Loaded sample image")
    step2 = plan.steps[1]
    pb.update_step(plan.id, step2.id, "complete", "OCR run successful")

    print("🧠 Memory logging test complete. Check console for embedded + stored entries.")

if __name__ == "__main__":
    test_plan_memory_integration()
