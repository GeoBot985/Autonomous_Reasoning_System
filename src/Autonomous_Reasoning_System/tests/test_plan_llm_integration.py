from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder

def test_llm_integration():
    pb = PlanBuilder()
    goal, plan = pb.new_goal_with_plan("Develop a file summarisation feature")

    print("Goal:", goal.text)
    print("Success:", goal.success_criteria)
    print("Failure:", goal.failure_criteria)
    print("Plan Steps:")
    for s in plan.steps:
        print("  -", s.description)

if __name__ == "__main__":
    test_llm_integration()
