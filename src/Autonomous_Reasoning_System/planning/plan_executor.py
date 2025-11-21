# planning/plan_executor.py
"""
PlanExecutor
------------
Executes plans created by PlanBuilder.
It iterates through plan steps, resolving them to tool executions via the Router/Dispatcher.
Handles state tracking, errors, and retries.
"""
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any

from Autonomous_Reasoning_System.control.dispatcher import Dispatcher
from Autonomous_Reasoning_System.control.router import Router
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder, Plan, Step

logger = logging.getLogger(__name__)

class PlanExecutor:
    """
    Executes a multi-step plan.
    """
    def __init__(self, plan_builder: PlanBuilder, dispatcher: Dispatcher, router: Optional[Router] = None, memory_interface=None):
        self.plan_builder = plan_builder
        self.dispatcher = dispatcher
        self.router = router or Router(dispatcher)
        # Prefer explicit memory injection; fall back to PlanBuilder's storage if present
        self.memory = memory_interface or getattr(plan_builder, "memory", None)
        self.retry_limit = 2

    def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Executes the plan with the given ID until completion or failure.
        Wraps execute_next_step in a loop.
        """
        plan = self.plan_builder.active_plans.get(plan_id)
        if not plan:
            return {"status": "error", "message": f"Plan {plan_id} not found."}

        logger.info(f"Starting execution of plan: {plan.title} ({plan_id})")

        last_result = {}

        # Loop until plan is no longer active (complete, suspended, failed)
        while True:
            result = self.execute_next_step(plan_id)
            status = result.get("status")
            last_result = result

            if status in ["complete", "suspended", "failed", "error"]:
                break

            if status != "running":
                # Should not happen if next_step returns running, but safety break
                break

        # Return the final result along with summary
        return_val = {
            "status": last_result.get("status", "unknown"),
            "plan_id": plan_id,
            "summary": self.plan_builder.get_plan_summary(plan_id),
            "final_output": last_result.get("final_output") or last_result.get("message")
        }

        # If completed successfully, try to get the result of the very last step from the plan object
        if plan.status == "complete" and len(plan.steps) > 0:
             last_step = plan.steps[-1]
             return_val["final_output"] = last_step.result

        return return_val

    def execute_next_step(self, plan_id: str) -> Dict[str, Any]:
        """
        Executes a single pending step of the plan.
        Returns the status of the plan/step execution.
        """
        plan = self.plan_builder.active_plans.get(plan_id)
        if not plan:
            return {"status": "error", "message": f"Plan {plan_id} not found."}

        if plan.status == "complete":
             # Try to get last result
             last_output = None
             if len(plan.steps) > 0:
                 last_output = plan.steps[-1].result

             return {
                "status": "complete",
                "plan_id": plan_id,
                "summary": self.plan_builder.get_plan_summary(plan_id),
                "final_output": last_output
            }

        step = plan.next_step()
        if not step:
            # No more steps, mark complete if not already
            if not plan.all_done():
                 pass
            else:
                 plan.status = "complete"
                 self.plan_builder._persist_plan(plan) # Ensure status is saved

            # Try to get last result
            last_output = None
            if len(plan.steps) > 0:
                 last_output = plan.steps[-1].result

            return {
                "status": "complete",
                "plan_id": plan_id,
                "summary": self.plan_builder.get_plan_summary(plan_id),
                "final_output": last_output
            }

        # Update plan status to active if pending
        if plan.status == "pending":
            plan.status = "active"

        logger.info(f"Executing step {plan.current_index + 1}: {step.description}")
        self.plan_builder.update_step(plan.id, step.id, "running")

        # Retry loop for this single step
        result = {"status": "error", "errors": ["Did not run"]}
        attempts = 0
        max_attempts = self.retry_limit + 1

        while attempts < max_attempts:
            attempts += 1
            result = self._execute_step(step, plan)

            if result["status"] == "success":
                break

            logger.warning(f"Step '{step.description}' failed attempt {attempts}/{max_attempts}. Errors: {result.get('errors')}")
            if attempts < max_attempts:
                time.sleep(0.5) # Backoff slightly

        if result["status"] == "success":
            output = str(result.get("final_output"))
            self.plan_builder.update_step(plan.id, step.id, "complete", result=output)
            plan.current_index += 1

            # check if that was the last step
            if plan.all_done():
                plan.status = "complete"
                self.plan_builder._persist_plan(plan)
                logger.info(f"Plan completed: {plan.title}")
                return {
                    "status": "complete",
                    "plan_id": plan_id,
                    "summary": self.plan_builder.get_plan_summary(plan_id),
                    "final_output": output
                }
            else:
                return {
                    "status": "running",
                    "plan_id": plan_id,
                    "step_completed": step.description,
                    "final_output": output
                }
        else:
            # Mark plan and goal as failed after exhausting retries
            error_msg = f"Failed after retries: {result.get('errors')}"
            self.plan_builder.update_step(plan.id, step.id, "failed", result=error_msg)
            plan.status = "failed"
            self.plan_builder._persist_plan(plan)

            if self.memory and plan.goal_id:
                try:
                    self.memory.update_goal(plan.goal_id, {
                        "status": "failed",
                        "updated_at": datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Failed to update goal {plan.goal_id} status: {e}")

            logger.error(f"Step failed after {attempts} attempts: {step.description}. Errors: {result.get('errors')}. Marking plan/goal as failed.")

            return {
                "status": "failed",
                "plan_id": plan_id,
                "failed_step": step.description,
                "errors": result.get("errors"),
                "message": f"I got stuck on step '{step.description}'. Error: {result.get('errors')}. Plan is marked as failed."
            }

    def _execute_step(self, step: Step, plan: Plan) -> Dict[str, Any]:
        """
        Executes a single step.
        Attempts to use the Router to determine the best tool(s) for the step description.
        """
        # Construct context from workspace
        context = plan.workspace.snapshot()

        try:
            # We rely on the Router to interpret the step description.
            # The router resolves the intent of the step description.
            route_result = self.router.route(step.description)

            failed_results = [r for r in route_result.get("results", []) if r["status"] != "success"]

            if failed_results:
                return {
                    "status": "error",
                    "errors": [r["errors"] for r in failed_results],
                    "route_result": route_result
                }

            # If success, store the output in workspace for future steps
            final_output = route_result.get("final_output")
            if final_output:
                plan.workspace.set("last_output", final_output)
                plan.workspace.set(f"step_{step.id}_output", final_output)

            return {
                "status": "success",
                "final_output": final_output
            }

        except Exception as e:
            logger.exception(f"Exception during step execution: {e}")
            return {"status": "error", "errors": [str(e)]}
