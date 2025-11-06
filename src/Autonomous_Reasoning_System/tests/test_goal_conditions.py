from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder

def test_goal_conditions():
    pb = PlanBuilder()
    goal = pb.new_goal("Build OCR module")

    print("Goal:", goal.text)
    print("Success:", goal.success_criteria)
    print("Failure:", goal.failure_criteria)

if __name__ == "__main__":
    test_goal_conditions()

