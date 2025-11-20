import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder, Plan, Step, Goal
from Autonomous_Reasoning_System.planning.plan_executor import PlanExecutor
from Autonomous_Reasoning_System.control.dispatcher import Dispatcher
from Autonomous_Reasoning_System.control.router import Router

class TestPlanExecution(unittest.TestCase):

    def setUp(self):
        self.mock_memory = MagicMock()
        self.mock_reflector = MagicMock()
        self.plan_builder = PlanBuilder(reflector=self.mock_reflector, memory_storage=self.mock_memory)

        self.dispatcher = Dispatcher()
        self.router = Router(self.dispatcher)
        self.plan_executor = PlanExecutor(self.plan_builder, self.dispatcher, self.router)

    def test_plan_generation_correctness(self):
        # Mock decompose_goal
        with patch.object(self.plan_builder, 'decompose_goal', return_value=["Step 1", "Step 2"]):
             goal, plan = self.plan_builder.new_goal_with_plan("Test Goal")

             self.assertEqual(goal.text, "Test Goal")
             self.assertEqual(len(plan.steps), 2)
             self.assertEqual(plan.steps[0].description, "Step 1")
             self.assertEqual(plan.steps[1].description, "Step 2")
             self.assertEqual(plan.status, "pending")

    def test_plan_execution_success(self):
        # Setup plan
        goal, plan = self.plan_builder.new_goal_with_plan("Test Execution")
        # Manually set steps to avoid LLM dependency in test
        plan.steps = [
            Step(id="s1", description="Task 1"),
            Step(id="s2", description="Task 2")
        ]

        # Mock Router.route to always succeed
        with patch.object(self.router, 'route') as mock_route:
            mock_route.return_value = {
                "intent": "test_intent",
                "pipeline": ["tool1"],
                "results": [{"status": "success", "data": "result1"}],
                "final_output": "result1"
            }

            result = self.plan_executor.execute_plan(plan.id)

            self.assertEqual(result["status"], "success")
            self.assertEqual(plan.status, "complete")
            self.assertEqual(plan.steps[0].status, "complete")
            self.assertEqual(plan.steps[0].result, "result1")
            self.assertEqual(plan.steps[1].status, "complete")

            # Verify router called for each step
            self.assertEqual(mock_route.call_count, 2)
            mock_route.assert_any_call("Task 1")
            mock_route.assert_any_call("Task 2")

    def test_plan_execution_failure(self):
        # Setup plan
        goal, plan = self.plan_builder.new_goal_with_plan("Test Failure")
        plan.steps = [
            Step(id="s1", description="Task 1"),
            Step(id="s2", description="Task 2")
        ]

        # Mock Router.route to fail on first step
        with patch.object(self.router, 'route') as mock_route:
            mock_route.return_value = {
                "intent": "test_intent",
                "pipeline": ["tool1"],
                "results": [{"status": "error", "errors": ["Something went wrong"]}],
                "final_output": None
            }

            result = self.plan_executor.execute_plan(plan.id)

            self.assertEqual(result["status"], "failed")
            self.assertEqual(plan.status, "failed")
            self.assertEqual(plan.steps[0].status, "failed")
            self.assertEqual(plan.steps[1].status, "pending") # Should not run
            self.assertIn("Something went wrong", str(result["errors"]))

    def test_partial_completion_handling(self):
        # Setup plan where step 1 succeeds and step 2 fails
        goal, plan = self.plan_builder.new_goal_with_plan("Test Partial")
        plan.steps = [
            Step(id="s1", description="Task 1"),
            Step(id="s2", description="Task 2")
        ]

        with patch.object(self.router, 'route') as mock_route:
            # Side effect: first call succeeds, second fails, but now with retries
            # Retry limit is 2, so it tries 3 times total.
            # We need to mock enough failures or success eventually.
            # Let's say it fails consistently.

            success_response = {
                "intent": "test_intent",
                "pipeline": ["tool1"],
                "results": [{"status": "success", "data": "res1"}],
                "final_output": "res1"
            }
            fail_response = {
                "intent": "test_intent",
                "pipeline": ["tool2"],
                "results": [{"status": "error", "errors": ["Fail"]}],
                "final_output": None
            }

            # 1 call for step 1 (success) + 3 calls for step 2 (fail, fail, fail)
            mock_route.side_effect = [success_response, fail_response, fail_response, fail_response]

            result = self.plan_executor.execute_plan(plan.id)

            self.assertEqual(result["status"], "failed")
            self.assertEqual(plan.steps[0].status, "complete")
            self.assertEqual(plan.steps[1].status, "failed")
            self.assertEqual(plan.current_index, 1) # 0-based, so 1 means second step was being processed

            # Verify retries happened
            self.assertEqual(mock_route.call_count, 4)

    def test_retry_logic_success(self):
        # Test that retry works if eventually succeeds
        goal, plan = self.plan_builder.new_goal_with_plan("Test Retry Success")
        plan.steps = [Step(id="s1", description="Task 1")]

        fail_response = {
            "intent": "test_intent",
            "pipeline": ["tool2"],
            "results": [{"status": "error", "errors": ["Fail"]}],
            "final_output": None
        }
        success_response = {
            "intent": "test_intent",
            "pipeline": ["tool1"],
            "results": [{"status": "success", "data": "res1"}],
            "final_output": "res1"
        }

        with patch.object(self.router, 'route') as mock_route:
            # Fail once, then succeed
            mock_route.side_effect = [fail_response, success_response]

            result = self.plan_executor.execute_plan(plan.id)

            self.assertEqual(result["status"], "success")
            self.assertEqual(plan.steps[0].status, "complete")
            self.assertEqual(mock_route.call_count, 2)

if __name__ == '__main__':
    unittest.main()
