import unittest
from unittest.mock import MagicMock, patch, ANY
from datetime import datetime
import json
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder, Plan, Step, Goal

class TestPlanBuilder(unittest.TestCase):

    def setUp(self):
        # Mock dependencies
        self.mock_memory = MagicMock()
        self.mock_reflector = MagicMock()
        self.mock_reasoner = MagicMock()

        # Patch internal imports of PlanBuilder so we don't trigger real LLM calls or singletons
        self.reasoner_patcher = patch('Autonomous_Reasoning_System.planning.plan_builder.PlanReasoner', return_value=self.mock_reasoner)
        self.reflector_patcher = patch('Autonomous_Reasoning_System.planning.plan_builder.ReflectionInterpreter', return_value=self.mock_reflector)

        self.MockPlanReasonerClass = self.reasoner_patcher.start()
        self.MockReflectionInterpreterClass = self.reflector_patcher.start()

        # Initialize PlanBuilder with mocks
        self.plan_builder = PlanBuilder(reflector=self.mock_reflector, memory_storage=self.mock_memory)
        # Manually set reasoner because __init__ might have created a new one from the patched class
        self.plan_builder.reasoner = self.mock_reasoner

    def tearDown(self):
        self.reasoner_patcher.stop()
        self.reflector_patcher.stop()

    def test_init_creates_plans_table(self):
        """Test that initializing PlanBuilder tries to create the plans table."""
        self.mock_memory.con.execute.assert_any_call(ANY)
        # We expect a CREATE TABLE call. checking if it was called.
        calls = [args[0] for args, _ in self.mock_memory.con.execute.call_args_list]
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS plans" in str(c) for c in calls))

    def test_new_goal_with_plan(self):
        """Test creating a new goal and decomposing it into a plan."""
        # Setup mocks
        self.mock_reflector.interpret.return_value = {"success": "Success Criteria", "failure": "Failure Criteria"}
        self.mock_reasoner.generate_steps.return_value = "1. Step One\n2. Step Two"

        # Execute
        goal, plan = self.plan_builder.new_goal_with_plan("Test Goal")

        # Assertions
        self.assertEqual(goal.text, "Test Goal")
        self.assertEqual(goal.success_criteria, "Success Criteria")
        self.assertEqual(goal.failure_criteria, "Failure Criteria")

        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].description, "Step One")
        self.assertEqual(plan.steps[1].description, "Step Two")

        # Check persistence
        self.assertIn(plan.id, self.plan_builder.active_plans)
        # Verify DB insert happened
        # The persist_plan method does DELETE then INSERT
        self.mock_memory.con.execute.assert_called()

    def test_update_step_complete(self):
        """Test updating a step to complete status."""
        # Create a dummy plan
        goal, plan = self.plan_builder.new_goal_with_plan("Dummy Goal")
        step_id = plan.steps[0].id

        # Update step
        self.plan_builder.update_step(plan.id, step_id, "complete", "Result ok")

        # Verify step status
        self.assertEqual(plan.steps[0].status, "complete")
        self.assertEqual(plan.steps[0].result, "Result ok")

        # Verify memory log
        self.mock_memory.add_memory.assert_called_with(
            text=ANY,
            memory_type="plan_progress",
            importance=0.4,
            source="PlanBuilder"
        )

        # Verify persistence
        # Should be called to update the plan in DB
        self.mock_memory.con.execute.assert_called()

    def test_plan_completion(self):
        """Test that completing all steps marks the plan as complete."""
        # Setup a plan with 2 steps
        self.mock_reasoner.generate_steps.return_value = "1. Step A\n2. Step B"
        goal, plan = self.plan_builder.new_goal_with_plan("Two Steps")

        step1 = plan.steps[0]
        step2 = plan.steps[1]

        # Complete step 1
        self.plan_builder.update_step(plan.id, step1.id, "complete")
        self.assertEqual(plan.status, "pending") # Or active, depending on logic.
        # Actually PlanBuilder doesn't set 'active' on update_step unless it was explicitly set before or during execution?
        # PlanExecutor sets it to active. PlanBuilder just updates step.
        # Let's check if PlanBuilder.update_step changes plan.status. Only if all done.

        # Complete step 2
        self.plan_builder.update_step(plan.id, step2.id, "complete")

        self.assertEqual(plan.status, "complete")

        # Verify plan summary memory
        calls = self.mock_memory.add_memory.call_args_list
        # The last call should be the completion summary
        last_call_args = calls[-1][1]
        self.assertEqual(last_call_args['memory_type'], "plan_summary")
        self.assertIn("completed successfully", last_call_args['text'])

    def test_get_active_plans(self):
        """Test retrieval of active plans."""
        # Plan 1: Pending
        self.mock_reasoner.generate_steps.return_value = "1. Step"
        _, plan1 = self.plan_builder.new_goal_with_plan("Goal 1")

        # Plan 2: Complete
        _, plan2 = self.plan_builder.new_goal_with_plan("Goal 2")
        self.plan_builder.update_step(plan2.id, plan2.steps[0].id, "complete")

        active = self.plan_builder.get_active_plans()

        self.assertIn(plan1, active)
        self.assertNotIn(plan2, active)

    def test_load_active_plans_hydration(self):
        """Test hydrating plans from memory."""
        # Setup mock DB return
        plan_data = {
            "id": "restored_id",
            "goal_id": "gid",
            "title": "Restored Plan",
            "steps": [
                {"id": "s1", "description": "Step 1", "status": "pending", "created_at": datetime.utcnow().isoformat(), "updated_at": datetime.utcnow().isoformat()}
            ],
            "current_index": 0,
            "status": "active",
            "created_at": datetime.utcnow().isoformat(),
            "workspace": {}
        }
        self.mock_memory.con.execute.return_value.fetchall.return_value = [
            (json.dumps(plan_data),)
        ]

        self.plan_builder.load_active_plans()

        self.assertIn("restored_id", self.plan_builder.active_plans)
        restored = self.plan_builder.active_plans["restored_id"]
        self.assertEqual(restored.title, "Restored Plan")
        self.assertEqual(restored.status, "active")

if __name__ == '__main__':
    unittest.main()
