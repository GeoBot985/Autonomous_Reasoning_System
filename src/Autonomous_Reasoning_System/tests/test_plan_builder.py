# tests/test_plan_builder.py
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder

def test_plan_builder_scaffold():
    pb = PlanBuilder()

    goal = pb.new_goal("Build OCR module")
    plan = pb.build_plan(goal, [
        "Load image",
        "Run OCR",
        "Store extracted text"
    ])

    print("✅ Goal created:", goal.text)
    print("✅ Plan steps:")
    for s in plan.steps:
        print("  -", s.description)

    # Simulate progress
    step = plan.next_step()
    pb.update_step(plan.id, step.id, "complete", "image loaded")
    print("Progress →", [(s.description, s.status) for s in plan.steps])

if __name__ == "__main__":
    test_plan_builder_scaffold()
