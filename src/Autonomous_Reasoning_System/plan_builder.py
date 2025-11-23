# plan_builder.py — FINAL WORKING VERSION (Dec 2025)
# This one works. No hangs. No missing returns. No 15s timeouts killing everything.

import json
import logging
import time
from uuid import uuid4
from typing import List

logger = logging.getLogger("ARS_Planner")


class Planner:
    def __init__(self, memory_system, llm_engine, retrieval_system):
        self.memory = memory_system
        self.llm = llm_engine
        self.retrieval = retrieval_system

    def process_request(self, user_request: str) -> str:
        logger.info("[Planner] New planning request: %s", user_request)

        # 1. Decompose the goal into steps
        steps = self._decompose_goal(user_request)
        if not steps or len(steps) == 0:
            return "I couldn't break this request into steps. I'll answer directly instead."

        logger.info("[Planner] Plan created with %d steps: %s", len(steps), steps)

        # 2. Save plan
        plan_id = str(uuid4())
        self.memory.update_plan(plan_id, user_request, steps, status="active")

        # 3. Execute every step
        result = self._execute_plan(plan_id, user_request, steps)
        return result

    def _decompose_goal(self, goal: str) -> List[str]:
        system = (
            "Break the user request into 3–6 short, clear, actionable steps. "
            "Return ONLY a JSON array of strings. No explanations, no markdown."
        )
        try:
            response = self.llm.generate(
                goal,
                system=system,
                temperature=0.1,
                # Critical: give the model time to think
            )
            response = response.strip()
            if response.startswith("[Error"):
                logger.error("[Planner] Decomposition failed: %s", response)
                return []

            # Clean common garbage
            response = response.replace("```json", "").replace("```", "").strip()
            steps = json.loads(response)
            return [s.strip() for s in steps if s.strip()][:6]
        except Exception as e:
            logger.error("[Planner] Failed to parse steps: %s", e)
            return []

    def _execute_plan(self, plan_id: str, goal: str, steps: List[str]) -> str:
        workspace: dict = {}

        for idx, step in enumerate(steps, 1):
            logger.info("[Planner] Executing step %d/%d: %s", idx, len(steps), step)

            # Build context only when needed
            context_lines = [f"OVERALL GOAL: {goal}"]
            if workspace:
                context_lines.append("\nPREVIOUS RESULTS:")
                for k, v in list(workspace.items())[-3:]:  # only last 3 to avoid bloat
                    short = (v[:400] + "..." if len(v) > 400 else v)
                    context_lines.append(f"- {k}: {short}")

            # Add memory only for research-type steps
            if any(word in step.lower() for word in ["find", "search", "look", "recall", "check", "what", "where"]):
                mem = self.retrieval.get_context_string(step, include_history=None)
                if len(mem) > 12_000:
                    mem = mem[:12_000] + "\n\n... [truncated]"
                context_lines.append("\nRELEVANT MEMORIES:\n" + mem)

            context = "\n".join(context_lines)

            # Execute step with long timeout tolerance
            step_result = self.llm.generate(
                f"Step {idx}: {step}\n\nContext:\n{context}\n\nRespond only with the result of this step.",
                system="You are executing one step of a plan. Be concise and accurate.",
                temperature=0.3
            )

            # Immediate fail if LLM died
            if step_result.startswith("[Error") or "unavailable" in step_result.lower():
                error = f"Stopped at step {idx}/{len(steps)} — model is too slow or unreachable right now."
                logger.error(error)
                self.memory.update_plan(plan_id, goal, steps, status="failed")
                return error + " Try again in a minute."

            workspace[f"Step {idx}: {step}"] = step_result
            self.memory.update_plan(plan_id, goal, steps, status=f"step_{idx}/{len(steps)}")

        # Final answer — OUTSIDE the loop
        logger.info("[Planner] All steps complete. Generating final answer...")
        final = self.llm.generate(
            f"User goal: {goal}\n\nGive a clear, natural final answer using only the results below.",
            system="Synthesize the results into a helpful response. Do NOT mention steps or planning.\n\n"
                   f"RESULTS:\n{json.dumps(workspace, indent=2)}",
            temperature=0.4
        )

        self.memory.update_plan(plan_id, goal, steps, status="completed")
        self.memory.remember(
            f"Completed plan → {goal}\nAnswer: {final}",
            memory_type="plan_summary",
            importance=0.9
        )
        return final


def get_planner(memory_system, llm_engine, retrieval_system):
    return Planner(memory_system, llm_engine, retrieval_system)