from Autonomous_Reasoning_System.control.scheduler import start_heartbeat_with_plans
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder
from Autonomous_Reasoning_System.cognition.learning_manager import LearningManager
from Autonomous_Reasoning_System.memory.confidence_manager import ConfidenceManager
import time

def test_scheduler_plan_awareness():
    learner = LearningManager()
    confidence = ConfidenceManager()
    pb = PlanBuilder()
    pb.new_goal_with_plan("Build OCR module")

    start_heartbeat_with_plans(learner, confidence, pb, interval_seconds=5, test_mode=True)
    print("⏳ Waiting 20s for heartbeats...")
    time.sleep(20)

if __name__ == "__main__":
    test_scheduler_plan_awareness()
