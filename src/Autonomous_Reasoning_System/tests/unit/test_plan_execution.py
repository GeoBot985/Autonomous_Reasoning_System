import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder, Plan, Step, Goal
from Autonomous_Reasoning_System.planning.plan_executor import PlanExecutor
from Autonomous_Reasoning_System.control.dispatcher import Dispatcher
from Autonomous_Reasoning_System.control.router import Router

class TestPlanExecution(unittest.TestCase):

    def setUp(self):
        # Mock dependencies
        self.mock_plan_builder = MagicMock(spec=PlanBuilder)
        self.mock_dispatcher = MagicMock(spec=Dispatcher)
        self.mock_router = MagicMock(spec=Router)

        # Create Executor with mocks
        self.plan_executor = PlanExecutor(self.mock_plan_builder, self.mock_dispatcher, self.mock_router)

        # Setup a sample plan
        self.plan_id = "test_plan_1"
        self.step1 = Step(id="s1", description="Step 1")
        self.step2 = Step(id="s2", description="Step 2")

        self.plan = MagicMock(spec=Plan)
        self.plan.id = self.plan_id
        self.plan.title = "Test Plan"
        self.plan.status = "pending"
        self.plan.steps = [self.step1, self.step2]
        self.plan.current_index = 0
        self.plan.workspace = MagicMock()
        self.plan.workspace.snapshot.return_value = {}

        # Setup default behavior: not all done
        self.plan.all_done.return_value = False

        # Setup next_step behavior
        # By default, it returns step1
        self.plan.next_step.return_value = self.step1

        # Setup active_plans in builder
        self.mock_plan_builder.active_plans = {self.plan_id: self.plan}
        self.mock_plan_builder.get_plan_summary.return_value = {"status": "running"}


    def test_execute_next_step_success(self):
        """Test successful execution of a single step."""
        # Setup router to return success
        self.mock_router.route.return_value = {
            "status": "success",
            "results": [{"status": "success"}],
            "final_output": "Output 1"
        }

        # Execute
        result = self.plan_executor.execute_next_step(self.plan_id)

        # Verify
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["step_completed"], "Step 1")

        # Verify router was called with description
        self.mock_router.route.assert_called_with("Step 1")

        # Verify plan builder updated step
        self.mock_plan_builder.update_step.assert_called_with(self.plan_id, "s1", "complete", result="Output 1")


    def test_execute_next_step_failure_retry_then_success(self):
        """Test retry logic where first attempt fails, second succeeds."""
        # First call to _execute_step (internal) needs to fail, second succeed.
        # Since _execute_step calls router.route, we can mock router.route to return different values.

        failure_response = {
            "status": "success", # Router returns success structurally but results might fail
            "results": [{"status": "error", "errors": "Some error"}]
        }
        success_response = {
            "status": "success",
            "results": [{"status": "success"}],
            "final_output": "Output Retry"
        }

        self.mock_router.route.side_effect = [failure_response, success_response]

        # Execute
        result = self.plan_executor.execute_next_step(self.plan_id)

        # Verify
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["step_completed"], "Step 1")

        # Verify router called twice
        self.assertEqual(self.mock_router.route.call_count, 2)

        # Verify update_step called with success eventually
        self.mock_plan_builder.update_step.assert_called_with(self.plan_id, "s1", "complete", result="Output Retry")

    def test_execute_next_step_failure_suspend(self):
        """Test plan suspension after max retries."""
        # Setup router to always fail
        failure_response = {
            "status": "success",
            "results": [{"status": "error", "errors": "Persistent error"}]
        }
        self.mock_router.route.return_value = failure_response

        # Execute
        result = self.plan_executor.execute_next_step(self.plan_id)

        # Verify
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failed_step"], "Step 1")

        # Verify retry count (retry_limit=2 -> 3 attempts total)
        self.assertEqual(self.mock_router.route.call_count, 3)

        # Verify plan status update
        self.mock_plan_builder.update_step.assert_called_with(self.plan_id, "s1", "failed", result=unittest.mock.ANY)
        # Verify plan object status set to failed
        self.assertEqual(self.plan.status, "failed")

    def test_execute_next_step_complete_plan(self):
        """Test that finishing the last step marks the plan as complete."""
        # Setup plan to indicate all done after this step
        self.plan.all_done.return_value = True

        self.mock_router.route.return_value = {
            "status": "success",
            "results": [{"status": "success"}],
            "final_output": "Done"
        }

        self.mock_plan_builder.get_plan_summary.return_value = {"status": "complete"}

        # Execute
        result = self.plan_executor.execute_next_step(self.plan_id)

        # Verify
        self.assertEqual(result["status"], "complete")
        self.assertEqual(self.plan.status, "complete")

        # Verify summary returned
        self.mock_plan_builder.get_plan_summary.assert_called_with(self.plan_id)

    def test_execute_next_step_already_complete(self):
        """Test behavior when plan is already complete."""
        self.plan.status = "complete"
        self.mock_plan_builder.get_plan_summary.return_value = {"status": "complete", "info": "already done"}

        result = self.plan_executor.execute_next_step(self.plan_id)

        self.assertEqual(result["status"], "complete")
        self.mock_router.route.assert_not_called()

    def test_execute_next_step_no_more_steps(self):
        """Test behavior when no pending steps remain."""
        self.plan.next_step.return_value = None
        self.plan.all_done.return_value = True # Assuming if no next step, it's done

        self.mock_plan_builder.get_plan_summary.return_value = {"status": "complete"}

        result = self.plan_executor.execute_next_step(self.plan_id)

        self.assertEqual(result["status"], "complete")
        self.assertEqual(self.plan.status, "complete")
        self.mock_router.route.assert_not_called()

if __name__ == '__main__':
    unittest.main()
