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
from typing import Optional, Dict, Any

from Autonomous_Reasoning_System.control.dispatcher import Dispatcher
from Autonomous_Reasoning_System.control.router import Router
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder, Plan, Step

logger = logging.getLogger(__name__)

class PlanExecutor:
    """
    Executes a multi-step plan.
    """
    def __init__(self, plan_builder: PlanBuilder, dispatcher: Dispatcher, router: Optional[Router] = None):
        self.plan_builder = plan_builder
        self.dispatcher = dispatcher
        self.router = router or Router(dispatcher)
        self.retry_limit = 2

    def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Executes the plan with the given ID.
        """
        plan = self.plan_builder.active_plans.get(plan_id)
        if not plan:
            return {"status": "error", "message": f"Plan {plan_id} not found."}

        logger.info(f"Starting execution of plan: {plan.title} ({plan_id})")
        plan.status = "active"

        # Initialize workspace context if needed (optional)
        # We can use plan.workspace to store intermediate results.

        while not plan.all_done():
            step = plan.next_step()
            if not step:
                break

            logger.info(f"Executing step {plan.current_index + 1}: {step.description}")
            self.plan_builder.update_step(plan.id, step.id, "running")

            # Retry loop
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
                self.plan_builder.update_step(plan.id, step.id, "complete", result=str(result.get("final_output")))
                plan.current_index += 1
            else:
                # Instead of failing the whole plan, we suspend it and ask for help.
                self.plan_builder.update_step(plan.id, step.id, "suspended", result=f"Suspended after failure: {result.get('errors')}")
                logger.error(f"Step failed after {attempts} attempts: {step.description}. Errors: {result.get('errors')}. Suspending plan.")
                plan.status = "suspended"

                return {
                    "status": "suspended",
                    "plan_id": plan_id,
                    "failed_step": step.description,
                    "errors": result.get("errors"),
                    "message": f"I got stuck on step '{step.description}'. Error: {result.get('errors')}. Please provide guidance or modify the plan."
                }

        plan.status = "complete"
        logger.info(f"Plan completed: {plan.title}")
        return {
            "status": "success",
            "plan_id": plan_id,
            "summary": self.plan_builder.get_plan_summary(plan_id)
        }

    def _execute_step(self, step: Step, plan: Plan) -> Dict[str, Any]:
        """
        Executes a single step.
        Attempts to use the Router to determine the best tool(s) for the step description.
        """
        # Use the step description as the intent for the router
        # We can also inject context from previous steps via plan.workspace

        # Construct context from workspace
        context = plan.workspace.snapshot()

        # We might want to prepend context to the query or pass it separately
        # For now, the Router.route accepts text.
        # Ideally, we should pass context to the router/dispatcher.

        # Let's see if we can hint the router with context.
        # The router's route method doesn't explicitly take external context
        # other than what it builds internally or what is passed to tools.
        # But dispatcher tools take arguments.

        # We rely on the Router to interpret the step description.
        # If the step is "Load image", Router should map to OCR/Image tools.

        try:
            route_result = self.router.route(step.description)

            # Check if the route result indicates success
            # The router returns: { "intent": ..., "pipeline": ..., "results": ..., "final_output": ... }
            # We check the results list.

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
                # Store with a key derived from step description or generic "last_output"
                plan.workspace.set("last_output", final_output)
                plan.workspace.set(f"step_{step.id}_output", final_output)

            return {
                "status": "success",
                "final_output": final_output
            }

        except Exception as e:
            logger.exception(f"Exception during step execution: {e}")
            return {"status": "error", "errors": [str(e)]}
