import logging
import json
from datetime import datetime
from typing import Optional
from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder
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

            # Convert steps from JSON string if needed, though MemoryInterface might handle object creation better
            # Here we are reading raw DF, so we need to be careful
            steps_raw = row['steps']
            if isinstance(steps_raw, str):
                try:
                    steps = json.loads(steps_raw)
                except:
                    steps = []
            else:
                steps = steps_raw or []

            logger.info(f"Checking goal {goal_id}: {goal_text} ({goal_status})")

            # Try to find linked Plan ID
            # Assuming metadata or extra columns might hold it, or we stored it in steps previously?
            # The user's report says GoalManager is split brain.
            # Ideally we should rely on PlanBuilder's active plans if possible,
            # but GoalManager seems to be driving the check.

            # Let's check if we stored plan_id in the goal metadata.
            # It's not standard yet, so we look at how _plan_goal stores it.
            # Previously it stored 'steps' json. Now we will store plan_id in metadata column if possible?
            # Or we can query PlanBuilder.active_plans to see if one matches this goal_id.

            linked_plan = None
            if self.plan_builder:
                # Find plan for this goal
                for plan in self.plan_builder.active_plans.values():
                    if plan.goal_id == goal_id:
                        linked_plan = plan
                        break

            if not steps and not linked_plan:
                actions_taken.append(self._plan_goal(goal_id, goal_text))
            else:
                # If we have a linked plan, delegate to PlanExecutor
                if linked_plan and self.plan_executor:
                    # Check if plan is already done
                    if linked_plan.status == "complete":
                         self.memory.update_goal(goal_id, {'status': 'completed', 'updated_at': datetime.utcnow().isoformat()})
                         actions_taken.append(f"Goal '{goal_text}' marked as completed (Plan finished).")
                    elif linked_plan.status in ["active", "pending", "suspended"]:
                         # Execute next step
                         res = self.plan_executor.execute_next_step(linked_plan.id)
                         actions_taken.append(f"Executed step for goal '{goal_text}': {res.get('status')}")

                         # Sync status back to goal if needed (optional, but keeps UI consistent)
                         if res.get("status") == "complete":
                              self.memory.update_goal(goal_id, {'status': 'completed', 'updated_at': datetime.utcnow().isoformat()})

                else:
                    # Legacy fallback or split brain fix:
                    # If we have steps in JSON but no Plan object in memory (maybe after restart without persistence loading?),
                    # we should probably hydrate the plan or fail gracefully.
                    # But since PlanBuilder hydrates plans on startup, we should have found it if it exists.
                    # If not found, maybe we should create a plan from the steps?

                    if steps:
                        # Check if there are pending steps
                        pending_steps = [s for s in steps if s.get('status') == 'pending']
                        if pending_steps:
                            actions_taken.append(self._execute_next_step(goal_id, goal_text, steps, pending_steps[0]))
                        else:
                            if all(s.get('status') in ['completed', 'failed', 'skipped'] for s in steps):
                                self.memory.update_goal(goal_id, {'status': 'completed', 'updated_at': datetime.utcnow().isoformat()})
                                actions_taken.append(f"Goal '{goal_text}' marked as completed.")

        return "\n".join(actions_taken) if actions_taken else "No actions needed on goals."

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
            plan_goal, plan = self.plan_builder.new_goal_with_plan(goal_text)
            # plan_goal has a new ID usually, but we want to link it to the existing goal_id?
            # The goal already exists in Memory (passed as goal_id).
            # new_goal_with_plan creates a NEW goal object in PlanBuilder.
            # We should probably just build a plan for the existing goal.

            # Correct approach: Build plan for THIS goal.
            # But plan_builder.build_plan requires a Goal object.
            # We can construct a temporary Goal object or fetch it?
            # PlanBuilder.new_goal_with_plan does both.

            # Let's use build_plan manually.
            # First decompose.
            steps_desc = self.plan_builder.decompose_goal(goal_text)

            # Create a Goal object wrapper for PlanBuilder (it expects it)
            # We assume goal_id corresponds to the ID in PlanBuilder's concept or we map it.
            # PlanBuilder.new_goal creates a goal and adds to active_goals.

            # We can manually inject it into PlanBuilder
            # Or we can update the Goal object in PlanBuilder if it exists.

            # To avoid ID mismatch, let's re-use goal_id if possible.
            # But PlanBuilder.new_goal generates a UUID.

            # Hack: We just use the plan. The Goal object in PlanBuilder is transient for structure.
            # The persistent goal is in MemoryInterface (DuckDB).

            # Let's create a dummy goal object for PlanBuilder
            goal_obj = Goal(id=goal_id, text=goal_text)
            self.plan_builder.active_goals[goal_id] = goal_obj

            plan = self.plan_builder.build_plan(goal_obj, steps_desc)

            # Extract steps for legacy view (optional)
            steps_data = []
            for step in plan.steps:
                steps_data.append({
                    'id': step.id,
                    'description': step.description,
                    'status': 'pending',
                    'result': None
                })

            self.memory.update_goal(goal_id, {
                'steps': json.dumps(steps_data),
                'status': 'active',
                'plan_id': plan.id, # Link plan ID
                'updated_at': datetime.utcnow().isoformat()
            })
            return f"Planned {len(steps_data)} steps for goal '{goal_text}'."
        except Exception as e:
            logger.error(f"Failed to plan goal {goal_id}: {e}")
            return f"Failed to plan goal '{goal_text}'."

    def _execute_next_step(self, goal_id: str, goal_text: str, all_steps: list, step_to_run: dict):
        """
        Executes the next step of a goal.
        Legacy method: Should only be used if no PlanExecutor/Plan is available.
        """
        logger.info(f"Executing step for goal '{goal_text}': {step_to_run['description']}")

        # FIX: Crash prevention. GoalManager should NOT call dispatcher.run_pipeline directly.
        # Also, we should use PlanExecutor if available even here?
        # But PlanExecutor requires a Plan object.
        # If we are here, it means we rely on 'steps' JSON list.

        try:
            # Update status to in_progress
            step_index = next((i for i, s in enumerate(all_steps) if s['id'] == step_to_run['id']), -1)
            if step_index == -1:
                return "Step error."

            all_steps[step_index]['status'] = 'in_progress'
            self.memory.update_goal(goal_id, {'steps': json.dumps(all_steps)})

            # Execute using Router to resolve pipeline, then Dispatcher to execute tools?
            # BUT dispatcher.run_pipeline doesn't exist.
            # We must use self.router.execute_pipeline

            route_decision = self.router.resolve(step_to_run['description'])
            pipeline = route_decision["pipeline"]

            # FIX: Use router.execute_pipeline instead of dispatcher.run_pipeline
            exec_result = self.router.execute_pipeline(pipeline, step_to_run['description'])
            # exec_result: { "results": [...], "final_output": ... }

            output = exec_result.get("final_output", "Done")

            # Update step status
            all_steps[step_index]['status'] = 'completed'
            all_steps[step_index]['result'] = output
            self.memory.update_goal(goal_id, {
                'steps': json.dumps(all_steps),
                'updated_at': datetime.utcnow().isoformat()
            })

            # Record episode
            self.memory.remember(
                f"Goal Step [{goal_text}]: {step_to_run['description']} -> {output}",
                metadata={"type": "goal_step", "goal_id": goal_id}
            )

            return f"Executed step '{step_to_run['description']}' for goal '{goal_text}'."

        except Exception as e:
            logger.error(f"Error executing step: {e}")
            all_steps[step_index]['status'] = 'failed'
            all_steps[step_index]['result'] = str(e)
            self.memory.update_goal(goal_id, {
                'steps': json.dumps(all_steps),
                'updated_at': datetime.utcnow().isoformat()
            })
            return f"Failed step '{step_to_run['description']}' for goal '{goal_text}'."
