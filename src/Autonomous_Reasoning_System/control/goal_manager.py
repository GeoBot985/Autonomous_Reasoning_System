import logging
import json
from datetime import datetime
from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder
from Autonomous_Reasoning_System.planning.plan_executor import PlanExecutor
from Autonomous_Reasoning_System.memory.goals import Goal

logger = logging.getLogger(__name__)

class GoalManager:
    def __init__(self, memory_interface: MemoryInterface, plan_builder: PlanBuilder, dispatcher, router):
        self.memory = memory_interface
        self.plan_builder = plan_builder
        self.dispatcher = dispatcher
        self.router = router

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

            # Logic to decide if we should act on this goal
            # For now, we act if:
            # 1. It has no steps (needs planning)
            # 2. It has steps but status is not completed/failed

            if not steps:
                actions_taken.append(self._plan_goal(goal_id, goal_text))
            else:
                # Check if there are pending steps
                pending_steps = [s for s in steps if s.get('status') == 'pending']
                if pending_steps:
                    actions_taken.append(self._execute_next_step(goal_id, goal_text, steps, pending_steps[0]))
                else:
                    # All steps done? Mark completed?
                    # For now, if all steps are done/failed, we mark goal completed
                    if all(s.get('status') in ['completed', 'failed', 'skipped'] for s in steps):
                         self.memory.update_goal(goal_id, {'status': 'completed', 'updated_at': datetime.utcnow().isoformat()})
                         actions_taken.append(f"Goal '{goal_text}' marked as completed.")

        return "\n".join(actions_taken) if actions_taken else "No actions needed on goals."

    def _plan_goal(self, goal_id: str, goal_text: str):
        """Builds a plan for a goal."""
        logger.info(f"Building plan for goal: {goal_text}")

        try:
            plan_goal, plan = self.plan_builder.new_goal_with_plan(goal_text)

            # Extract steps
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
                'updated_at': datetime.utcnow().isoformat()
            })
            return f"Planned {len(steps_data)} steps for goal '{goal_text}'."
        except Exception as e:
            logger.error(f"Failed to plan goal {goal_id}: {e}")
            return f"Failed to plan goal '{goal_text}'."

    def _execute_next_step(self, goal_id: str, goal_text: str, all_steps: list, step_to_run: dict):
        """Executes the next step of a goal."""
        logger.info(f"Executing step for goal '{goal_text}': {step_to_run['description']}")

        # We use PlanExecutor to execute a single step?
        # PlanExecutor works on Plan objects.
        # We can construct a temporary Plan object or use Dispatcher directly?
        # PlanExecutor.execute_step might be private or tied to Plan structure.

        # Let's look at PlanExecutor.execute_plan. It iterates steps.
        # We might want to use the Dispatcher/Router logic directly for a single step.

        # "The PlanExecutor will route this step, effectively executing the pipeline determined by Router"
        # So we can treat the step description as input to Router/Dispatcher.

        try:
            # Update status to in_progress
            step_index = next((i for i, s in enumerate(all_steps) if s['id'] == step_to_run['id']), -1)
            if step_index == -1:
                return "Step error."

            all_steps[step_index]['status'] = 'in_progress'
            self.memory.update_goal(goal_id, {'steps': json.dumps(all_steps)})

            # Execute

            # We can simulate a single step execution similar to CoreLoop run_once logic for simple tasks
            route_decision = self.router.resolve(step_to_run['description'])
            pipeline = route_decision["pipeline"]

            result = self.dispatcher.run_pipeline(pipeline, step_to_run['description'])

            output = result.get("data", {}).get("summary", "Done")
            if not output:
                 output = str(result.get("data", ""))

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
