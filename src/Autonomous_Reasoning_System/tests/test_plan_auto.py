from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder

def test_auto_plan_creation():
    pb = PlanBuilder()
    goal, plan = pb.new_goal_with_plan("Build OCR module")

    print("Goal:", goal.text)
    print("Success:", goal.success_criteria)
    print("Failure:", goal.failure_criteria)
    print("Plan Steps:")
    for step in plan.steps:
        print("  -", step.description)

if __name__ == "__main__":
    test_auto_plan_creation()
