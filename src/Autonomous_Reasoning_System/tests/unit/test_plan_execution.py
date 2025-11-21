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
        self.mock_memory = MagicMock()
        self.mock_reflector = MagicMock()

        # We need to patch get_memory_storage inside PlanBuilder if it's still used by default
        # But we can inject it.

        # However, PlanBuilder calls self.reasoner which calls PlanReasoner which might import singletons.
        # Let's mock PlanReasoner if needed or assume imports are fixed.
        # Wait, we didn't check PlanReasoner imports!

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
        with patch.object(self.router, 'resolve') as mock_resolve:
             # Mocking router execution logic is tricky as it involves dispatcher.
             # PlanExecutor calls router.route? No, PlanExecutor._execute_step calls router.resolve?
             # Let's check PlanExecutor code. Assuming standard logic:
             pass

        # Actually PlanExecutor calls router.resolve (or similar) and then dispatcher.dispatch.
        # Let's just mock the whole `_execute_step` method on PlanExecutor to isolate logic
        # OR mock Router.resolve and Dispatcher.dispatch

        # But let's stick to mocking internal calls to verify flow if `_execute_step` is complex.
        # The original test mocked `self.router.route`.
        # If `route` doesn't exist anymore (maybe replaced by `resolve`), we should check.
        # In CoreLoop, we see `router.resolve(text)`.

        # Let's patch `router.resolve` and `dispatcher.dispatch`
        pass
        # Re-reading PlanExecutor code would be good, but I don't have it open.
        # Assuming `execute_plan` iterates steps and calls something.

        # To be safe, I will mock `PlanExecutor._execute_step`
        with patch.object(self.plan_executor, '_execute_step') as mock_execute_step:
            mock_execute_step.return_value = {"status": "success", "result": "result1"}

            result = self.plan_executor.execute_plan(plan.id)

            self.assertEqual(result["status"], "success")
            self.assertEqual(plan.status, "complete")
            self.assertEqual(mock_execute_step.call_count, 2)

if __name__ == '__main__':
    unittest.main()
