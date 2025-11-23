import json
import logging
import re
from uuid import uuid4
from typing import List

# Setup simple logging
logger = logging.getLogger("ARS_Planner")

class Planner:
    """
    The Engineer.
    Handles multi-step reasoning, goal decomposition, and execution.
    Dependencies: None (Uses injected Memory/LLM).
    """

    def __init__(self, memory_system, llm_engine):
        self.memory = memory_system
        self.llm = llm_engine

    def process_request(self, user_request: str) -> str:
        logger.info(f"🏗️ Planning request: {user_request}")
        
        # 1. Decompose
        steps = self._decompose_goal(user_request)
        if not steps:
            return "⚠️ I failed to generate a plan for that request."

        # 2. Record Plan
        plan_id = str(uuid4())
        self.memory.update_plan(plan_id, user_request, steps, status="active")
        
        logger.info(f"📋 Plan created ({len(steps)} steps). Executing...")
        
        # 3. Execute
        final_result = self._execute_plan(plan_id, user_request, steps)
        
        return final_result

    def _decompose_goal(self, goal_text: str) -> List[str]:
        """Uses LLM to break a goal into steps."""
        system = (
            "You are an expert planner. Break the user's goal into a logical sequence of 3 to 5 actionable steps. "
            "Return ONLY a JSON list of strings. "
            "Example: [\"Search for X\", \"Summarize Y\", \"Save result\"]"
        )
        
        response = self.llm.generate(goal_text, system=system, temperature=0.2)
        
        try:
            # Try parsing JSON
            steps = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: Regex for "1. Step" or "- Step"
            steps = re.findall(r"(?:\d+\.|\-)\s*(.+)", response)
            
        if not isinstance(steps, list) or not steps:
            return ["Analyze the request", "Provide a comprehensive answer"]
            
        return steps[:6]

    def _execute_plan(self, plan_id: str, goal_text: str, steps: List[str]) -> str:
        """Execute steps sequentially with a shared workspace."""
        workspace = {} 
        
        for i, step_desc in enumerate(steps):
            logger.info(f"▶️ Step {i+1}/{len(steps)}: {step_desc}")
            
            # Context construction
            context_str = f"GOAL: {goal_text}\n"
            if workspace:
                context_str += "PREVIOUS RESULTS:\n"
                for k, v in workspace.items():
                    context_str += f"- {k}: {v}\n"
            
            # Execute
            step_result = self.llm.generate(
                f"Execute this step: {step_desc}",
                system=f"Context:\n{context_str}\n\nPerform the task and return the result."
            )
            
            # Save logic (Heuristic)
            if "save" in step_desc.lower() or "store" in step_desc.lower():
                self.memory.remember(step_result, source="planner_step")
            
            workspace[f"Step {i+1}"] = step_result
            self.memory.update_plan(plan_id, goal_text, steps, status=f"working_on_{i+1}")

        # Final Summarization
        final_summary = self.llm.generate(
            f"Based on the results, provide a final response to the user's goal: '{goal_text}'.",
            system=f"You are completing a multi-step task.\nRESULTS:\n{json.dumps(workspace, indent=2)}"
        )
        
        self.memory.update_plan(plan_id, goal_text, steps, status="completed")
        self.memory.remember(f"Completed plan '{goal_text}'. Result: {final_summary}", memory_type="plan_summary")
        
        return final_summary

def get_planner(memory_system, llm_engine):
    return Planner(memory_system, llm_engine)