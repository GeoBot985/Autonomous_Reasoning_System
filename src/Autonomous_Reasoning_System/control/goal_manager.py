import logging
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder, Plan, Step
from Autonomous_Reasoning_System.planning.plan_executor import PlanExecutor
from Autonomous_Reasoning_System.memory.goals import Goal

logger = logging.getLogger(__name__)

class GoalManager:
    def __init__(self, memory_interface: MemoryInterface, plan_builder: PlanBuilder, dispatcher, router, plan_executor: Optional[PlanExecutor] = None):
        self.memory = memory_interface
        self.plan_builder = plan_builder
        self.dispatcher = dispatcher
        self.router = router
        self.plan_executor = plan_executor

    def create_goal(self, text: str, priority: int = 1, metadata: dict = None) -> str:
        """Creates a new long-running goal."""
        logger.info(f"Creating new goal: {text}")
        return self.memory.create_goal(text, priority, metadata)

    def check_goals(self):
        """
        Iterates through active goals and decides if action is needed.
        Returns a summary of actions taken.
        """
        active_goals_df = self.memory.get_active_goals()
        if active_goals_df.empty:
            return "No active goals."

        actions_taken = []

        for _, row in active_goals_df.iterrows():
            goal_id = row['id']
            goal_text = row['text']
            goal_status = row['status']
            steps_raw = row['steps']

            logger.info(f"Checking goal {goal_id}: {goal_text} ({goal_status})")

            # 1. Attempt to resolve linked Plan
            linked_plan = self._resolve_plan(goal_id, goal_text, steps_raw)

            if not linked_plan:
                # If no plan exists (and couldn't be migrated), we need to plan it.
                actions_taken.append(self._plan_goal(goal_id, goal_text))
            else:
                # 2. Execute via PlanExecutor
                if self.plan_executor:
                    # Check status
                    if linked_plan.status == "complete":
                         self.memory.update_goal(goal_id, {'status': 'completed', 'updated_at': datetime.utcnow().isoformat()})
                         actions_taken.append(f"Goal '{goal_text}' marked as completed (Plan finished).")

                    elif linked_plan.status == "failed":
                         self.memory.update_goal(goal_id, {'status': 'failed', 'updated_at': datetime.utcnow().isoformat()})
                         actions_taken.append(f"Goal '{goal_text}' marked as failed.")

                    elif linked_plan.status in ["active", "pending", "suspended", "running"]:
                         # Execute next step
                         # execute_next_step handles execution logic
                         res = self.plan_executor.execute_next_step(linked_plan.id)

                         status = res.get("status")

                         if status == "complete":
                              self.memory.update_goal(goal_id, {'status': 'completed', 'updated_at': datetime.utcnow().isoformat()})
                              actions_taken.append(f"Goal '{goal_text}' completed.")
                         elif status == "failed":
                              self.memory.update_goal(goal_id, {'status': 'failed', 'updated_at': datetime.utcnow().isoformat()})
                              actions_taken.append(f"Goal '{goal_text}' failed: {res.get('message')}")
                         elif status == "running" or status == "success": # success is from _execute_step internal return, execute_next_step returns running usually
                              step_desc = res.get("step_completed", "step")
                              actions_taken.append(f"Executed step for goal '{goal_text}': {step_desc}")
                         else:
                              actions_taken.append(f"Goal '{goal_text}' status: {status}")
                else:
                    actions_taken.append(f"Skipping goal '{goal_text}' (No PlanExecutor available).")

        return "\n".join(actions_taken) if actions_taken else "No actions needed on goals."

    def _resolve_plan(self, goal_id: str, goal_text: str, steps_raw: Any) -> Optional[Plan]:
        """
        Finds the active plan for a goal.
        If a 'legacy' steps JSON exists but no Plan object, migrates it to a Plan.
        """
        # Check PlanBuilder for existing active plan
        for plan in self.plan_builder.active_plans.values():
            if plan.goal_id == goal_id:
                return plan

        # If not found, check if we have legacy steps to migrate
        steps = []
        if isinstance(steps_raw, str):
            try:
                steps = json.loads(steps_raw)
            except:
                steps = []
        elif isinstance(steps_raw, list):
            steps = steps_raw

        if steps:
            logger.info(f"Migrating legacy steps for goal {goal_id} to Plan object.")
            # Create a new Plan object reflecting these steps
            # We need to map legacy step dicts to Step objects
            plan_steps = []
            for s in steps:
                step_obj = Step(
                    id=s.get('id', str(datetime.utcnow().timestamp())), # Use timestamp if no ID, but legacy usually had IDs?
                    description=s.get('description', 'Unknown step'),
                    status=s.get('status', 'pending'),
                    result=s.get('result')
                )
                plan_steps.append(step_obj)

            # Create Plan
            # Note: PlanBuilder usually creates IDs. We create one here.
            # We need to register it with PlanBuilder so it's managed.

            # Create a dummy Goal object for PlanBuilder if needed
            if goal_id not in self.plan_builder.active_goals:
                self.plan_builder.active_goals[goal_id] = Goal(id=goal_id, text=goal_text)

            goal_obj = self.plan_builder.active_goals[goal_id]

            # Manually build plan to inject specific steps
            from uuid import uuid4
            plan = Plan(
                id=str(uuid4()),
                goal_id=goal_id,
                title=goal_text,
                steps=plan_steps,
                status="active"
            )

            # Determine current index based on status
            completed_count = sum(1 for s in plan_steps if s.status in ['complete', 'failed', 'skipped'])
            plan.current_index = completed_count

            if all(s.status in ['complete', 'failed', 'skipped'] for s in plan_steps):
                plan.status = "complete"

            # Register
            self.plan_builder.active_plans[plan.id] = plan
            self.plan_builder._persist_plan(plan)

            # Update goal with plan_id
            self.memory.update_goal(goal_id, {'plan_id': plan.id})

            return plan

        return None

    def get_goals_list(self, status: str = None) -> list:
        """
        Return active goals as a list of plain dicts for fast, clean consumption.
        Optionally filter by status.
        """
        df = self.memory.get_active_goals()
        if status:
            df = df[df["status"] == status]
        return df.to_dict(orient="records") if not df.empty else []

    def _plan_goal(self, goal_id: str, goal_text: str):
        """Builds a plan for a goal."""
        logger.info(f"Building plan for goal: {goal_text}")

        try:
            # Ensure Goal object exists in PlanBuilder
            if goal_id not in self.plan_builder.active_goals:
                self.plan_builder.active_goals[goal_id] = Goal(id=goal_id, text=goal_text)

            goal_obj = self.plan_builder.active_goals[goal_id]

            # Decompose
            steps_desc = self.plan_builder.decompose_goal(goal_text)

            # Build Plan
            plan = self.plan_builder.build_plan(goal_obj, steps_desc)

            # Update Goal in DB with plan_id
            self.memory.update_goal(goal_id, {
                'status': 'active',
                'plan_id': plan.id,
                'updated_at': datetime.utcnow().isoformat()
            })
            return f"Planned {len(steps_desc)} steps for goal '{goal_text}'."
        except Exception as e:
            logger.error(f"Failed to plan goal {goal_id}: {e}")
            return f"Failed to plan goal '{goal_text}'."
