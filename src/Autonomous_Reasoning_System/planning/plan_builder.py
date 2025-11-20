# planning/plan_builder.py
"""
PlanBuilder
-----------
Creates and manages goal–plan–step hierarchies for Tyrone.
This module handles structure and progress tracking only,
leaving execution control to the CoreLoop or Scheduler.
"""
from unittest import result
from Autonomous_Reasoning_System.memory.singletons import get_memory_storage
from Autonomous_Reasoning_System.llm.reflection_interpreter import ReflectionInterpreter
from Autonomous_Reasoning_System.llm.plan_reasoner import PlanReasoner

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from typing import List, Optional, Dict, Any
from .workspace import Workspace


# ---------------------------------------------------------------------
# 🧩 Core Data Models
# ---------------------------------------------------------------------

@dataclass
class Step:
    id: str
    description: str
    status: str = "pending"          # pending | running | complete | failed
    result: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Plan:
    id: str
    goal_id: str
    title: str
    steps: List[Step] = field(default_factory=list)
    current_index: int = 0
    status: str = "pending"          # pending | active | complete | failed
    created_at: datetime = field(default_factory=datetime.utcnow)
    workspace: Workspace = field(default_factory=Workspace)

    def next_step(self) -> Optional[Step]:
        """Return the next pending step without advancing index."""
        for step in self.steps:
            if step.status == "pending":
                return step
        return None

    def mark_step(self, step_id: str, status: str, result: Optional[str] = None):
        """Update a step's status and result."""
        for step in self.steps:
            if step.id == step_id:
                step.status = status
                step.result = result
                step.updated_at = datetime.utcnow()
                break

    def all_done(self) -> bool:
        """Return True if all steps are complete."""
        return all(s.status == "complete" for s in self.steps)
    
    def progress_summary(self) -> dict:
        """Return structured progress information for this plan."""
        total = len(self.steps)
        completed = sum(1 for s in self.steps if s.status == "complete")
        pending = total - completed
        current = self.next_step().description if self.next_step() else "None"
        percent = (completed / total) * 100 if total else 0

        return {
            "plan_id": self.id,
            "title": self.title,
            "status": self.status,
            "completed_steps": completed,
            "total_steps": total,
            "pending_steps": pending,
            "current_step": current,
            "percent_complete": round(percent, 1)
        }



@dataclass
class Goal:
    id: str
    text: str
    success_criteria: str = ""
    failure_criteria: str = ""
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.utcnow)
    plan: Optional[Plan] = None


# ---------------------------------------------------------------------
# 🧠 PlanBuilder Core
# ---------------------------------------------------------------------

class PlanBuilder:
    def __init__(self, reflector: ReflectionInterpreter | None = None, memory_storage=None):
        self.active_goals: Dict[str, Goal] = {}
        self.active_plans: Dict[str, Plan] = {}
        self.memory = memory_storage if memory_storage else get_memory_storage()
        self.reflector = reflector or ReflectionInterpreter()
        self.reasoner = PlanReasoner() 

    # ------------------- Goal Management -------------------
    def new_goal(self, goal_text: str) -> Goal:
        """Create a new goal with derived success/failure criteria."""
        goal_id = str(uuid4())
        success, failure = self.derive_success_failure(goal_text)
        goal = Goal(
            id=goal_id,
            text=goal_text,
            success_criteria=success,
            failure_criteria=failure
        )
        self.active_goals[goal.id] = goal
        return goal
    
    def new_goal_with_plan(self, goal_text: str) -> tuple[Goal, Plan]:
        """Create a goal, derive conditions, decompose into a plan, and register it."""
        goal = self.new_goal(goal_text)
        steps = self.decompose_goal(goal_text)
        plan = self.build_plan(goal, steps)
        print(f"🧠 Created plan for goal '{goal_text}' with {len(steps)} steps.")
        return goal, plan



    # ------------------- Plan Construction -------------------

    def build_plan(self, goal: Goal, step_descriptions: List[str]) -> Plan:
        """
        Create a plan for a goal, based on a list of step descriptions.
        """
        steps = [Step(id=str(uuid4()), description=s) for s in step_descriptions]
        plan = Plan(id=str(uuid4()), goal_id=goal.id, title=goal.text, steps=steps)
        goal.plan = plan
        self.active_plans[plan.id] = plan
        return plan

    # ------------------- Progress & Accessors -------------------

    def get_active_plans(self) -> List[Plan]:
        """Return all plans that are not complete."""
        return [p for p in self.active_plans.values() if p.status != "complete"]
    
    def get_plan_summary(self, plan_id: str) -> dict:
        """Generate a human-readable summary of a specific plan’s progress."""
        plan = self.active_plans.get(plan_id)
        if not plan:
            return {"error": f"Plan {plan_id} not found."}

        info = plan.progress_summary()
        summary_text = (
            f"Goal: {plan.title}\n"
            f"Status: {info['status']} | {info['completed_steps']}/{info['total_steps']} "
            f"steps complete ({info['percent_complete']}%).\n"
            f"Current step: {info['current_step']}."
        )
        info["summary_text"] = summary_text
        return info


    def update_step(self, plan_id: str, step_id: str, status: str, result: Optional[str] = None):
        """Mark a step as complete or failed, and store a progress note in memory."""
        plan = self.active_plans.get(plan_id)
        if not plan:
            return

        plan.mark_step(step_id, status, result)

        # --- Build progress summary and store it ---
        summary = plan.progress_summary()
        note = (
            f"📋 Plan update for goal '{plan.title}': "
            f"{summary['completed_steps']}/{summary['total_steps']} steps complete "
            f"({summary['percent_complete']}%). Current step: {summary['current_step']}. "
            f"Last action result: {result or 'N/A'}."
        )
        self.memory.add_memory(
            text=note,
            memory_type="plan_progress",
            importance=0.4,
            source="PlanBuilder"
        )

        # --- Mark plan complete if all done ---
        if plan.all_done():
            plan.status = "complete"
            done_note = f"✅ Plan '{plan.title}' completed successfully."
            self.memory.add_memory(
                text=done_note,
                memory_type="plan_summary",
                importance=0.7,
                source="PlanBuilder"
            )


    def archive_completed(self):
        """Remove completed plans from active registry."""
        self.active_plans = {k: v for k, v in self.active_plans.items() if v.status != "complete"}

        # ------------------- Goal Reasoning -------------------

    def derive_success_failure(self, goal_text: str) -> tuple[str, str]:
        try:
            prompt = (
                f"Given this goal: '{goal_text}', "
                "describe in one short sentence what success means, and one sentence what failure means."
            )
            result = self.reflector.interpret(prompt)

            if isinstance(result, dict):
                if "success" in result and "failure" in result:
                    return result["success"], result["failure"]
                if "summary" in result:
                    text = result["summary"]
                else:
                    text = str(result)
            else:
                text = str(result)

            # crude parse for 'success:' / 'failure:' if JSON not returned
            import re, json
            try:
                parsed = json.loads(text)
                return parsed.get("success", ""), parsed.get("failure", "")
            except Exception:
                s = re.search(r"[Ss]uccess[:\-]?\s*(.+?)(?:[Ff]ailure|$)", text)
                f = re.search(r"[Ff]ailure[:\-]?\s*(.+)", text)
                success = s.group(1).strip() if s else f"Goal '{goal_text}' achieved successfully."
                failure = f.group(1).strip() if f else f"Goal '{goal_text}' not achieved."
                return success, failure

        except Exception:
            # fallback heuristic
            g = goal_text.lower()
            if "ocr" in g:
                return ("Text is correctly extracted from images.",
                        "OCR cannot detect or read text.")
            elif "memory" in g:
                return ("System stores and retrieves information reliably.",
                        "Information is lost or corrupted.")
            else:
                return (f"Goal '{goal_text}' achieved as described.",
                        f"Goal '{goal_text}' not achieved or produced errors.")


        # ------------------- Automatic Plan Decomposition -------------------

    def decompose_goal(self, goal_text: str) -> list[str]:
        """
        Generate actionable step descriptions for a given goal.
        Uses ReflectionInterpreter (LLM) reasoning, with structured parsing and safe fallbacks.
        """
        g = goal_text.lower()

        # --- Fast heuristic matches ---
        if "ocr" in g:
            return ["Load image", "Run OCR", "Extract text", "Store extracted text"]
        if "reminder" in g:
            return ["Create reminder entry", "Schedule trigger", "Notify user"]
        if "memory" in g:
            return ["Capture input", "Store entry", "Retrieve entry", "Verify correctness"]

        # --- Reasoning-based path ---
        try:
            prompt = (
                f"You are Tyrone's planning assistant. Your task is to convert this goal into a numbered list of concrete steps.\n\n"
                f"Goal: {goal_text}\n\n"
                "Output format:\n"
                "1. Step one\n"
                "2. Step two\n"
                "3. Step three\n"
                "...\n\n"
                "Rules:\n"
                "- Use exactly 3 to 6 short actionable steps.\n"
                "- Do NOT explain, reflect, or restate the goal.\n"
                "- Do NOT add commentary or insights.\n"
                "- Return ONLY the numbered list."
            )


            result = self.reasoner.generate_steps(goal_text)
            text = str(result)


            # Convert to plain text
            if isinstance(result, dict):
                text = result.get("summary", "") or result.get("insight", "") or str(result)
            else:
                text = str(result)

            # --- Parse for bullet or numbered patterns ---
            import re, json
            lines = re.findall(r"(?:\d+\.|\-|\•)\s*(.+)", text)
            if lines:
                return [l.strip() for l in lines if l.strip()]

            # --- Try JSON list if it ever appears ---
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(s).strip() for s in parsed if isinstance(s, str)]
            except Exception:
                pass

            # --- Sentence split fallback ---
            if "." in text:
                parts = [p.strip() for p in text.split(".") if len(p.strip()) > 3]
                if 2 <= len(parts) <= 8:
                    return parts

        except Exception as e:
            print(f"[WARN] LLM decomposition failed: {e}")

        # --- Final fallback ---
        return ["Define objective", "Execute objective", "Verify outcome"]


