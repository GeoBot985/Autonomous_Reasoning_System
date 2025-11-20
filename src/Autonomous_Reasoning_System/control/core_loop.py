import time
import logging
from datetime import datetime
from Autonomous_Reasoning_System.control.dispatcher import Dispatcher
from Autonomous_Reasoning_System.control.router import Router
from Autonomous_Reasoning_System.cognition.intent_analyzer import IntentAnalyzer
from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface
from Autonomous_Reasoning_System.llm.reflection_interpreter import ReflectionInterpreter
from Autonomous_Reasoning_System.memory.confidence_manager import ConfidenceManager
from Autonomous_Reasoning_System.cognition.self_validator import SelfValidator
from Autonomous_Reasoning_System.cognition.learning_manager import LearningManager
from Autonomous_Reasoning_System.control.scheduler import start_heartbeat_with_plans
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder
from Autonomous_Reasoning_System.planning.plan_executor import PlanExecutor
from Autonomous_Reasoning_System.control.attention_manager import attention
from Autonomous_Reasoning_System.tools.standard_tools import register_tools
from Autonomous_Reasoning_System.llm.context_adapter import ContextAdapter

logger = logging.getLogger(__name__)

class CoreLoop:
    def __init__(self):
        # 1. Initialize Dispatcher first
        self.dispatcher = Dispatcher()

        # 2. Initialize Components
        self.memory = MemoryInterface()
        self.intent_analyzer = IntentAnalyzer()
        self.reflector = ReflectionInterpreter()
        self.confidence = ConfidenceManager()
        self.validator = SelfValidator()
        self.learner = LearningManager()
        self.plan_builder = PlanBuilder()
        self.context_adapter = ContextAdapter()

        # 3. Register Tools
        components = {
            "intent_analyzer": self.intent_analyzer,
            "memory": self.memory,
            "reflector": self.reflector,
            "plan_builder": self.plan_builder,
            "context_adapter": self.context_adapter
        }
        register_tools(self.dispatcher, components)

        # 4. Initialize Control & Execution
        self.router = Router(self.dispatcher)
        self.plan_executor = PlanExecutor(self.plan_builder, self.dispatcher, self.router)

        # 5. Start Background Tasks
        start_heartbeat_with_plans(
            self.learner, self.confidence, self.plan_builder, interval_seconds=10, test_mode=True
        )
        self.running = False

    def run_once(self, text: str):
        """
        Executes the Full Reasoning Loop:
        1. Router (resolve pipeline)
        2. Plan Builder (create plan)
        3. Dispatcher (execute plan via PlanExecutor)
        4. Return Output
        5. Update Memory
        6. Reflection
        """
        print(f"\n[CORE LOOP] Received input: {text}")
        start_time = time.time()

        # --- Step 1: Use Router to determine pipeline ---
        route_decision = self.router.resolve(text)
        intent = route_decision["intent"]
        pipeline = route_decision["pipeline"]
        print(f"[ROUTER] Intent: {intent} | Pipeline: {pipeline}")

        # --- Step 2: Build a plan ---
        # If intent requires complex planning, we decompose.
        # Otherwise, we wrap the simple pipeline/input as a single-step plan.
        if intent == "plan" or intent == "complex_task":
            goal, plan = self.plan_builder.new_goal_with_plan(text)
            print(f"[PLANNER] Created multi-step plan: {plan.id}")
        else:
            # Simple execution: Treat the original input as the step description
            # The PlanExecutor will route this step, effectively executing the pipeline determined by Router
            goal = self.plan_builder.new_goal(text)
            plan = self.plan_builder.build_plan(goal, [text])
            print(f"[PLANNER] Created single-step execution plan: {plan.id}")

        # --- Step 3: Execute via Dispatcher (PlanExecutor uses Dispatcher/Router) ---
        execution_result = self.plan_executor.execute_plan(plan.id)

        final_output = ""
        if execution_result["status"] == "success":
            # Extract summary or final output
            # PlanExecutor returns summary in data usually
            summary = execution_result.get("summary", {})
            if isinstance(summary, dict):
                 # If it's a plan summary, we might want the last step's output
                 # But PlanExecutor logic stores results in memory/step history.
                 # We can try to retrieve the result from the plan's last completed step.
                 # Or use what PlanExecutor returns.
                 pass

            # For simple plans (single step), the result of that step is the answer.
            # PlanExecutor doesn't return the raw output of the last step easily in the return dict.
            # We must look at the plan steps.
            if len(plan.steps) > 0:
                last_step = plan.steps[-1]
                final_output = last_step.result or "Done."
            else:
                final_output = "Plan completed with no steps."
        else:
            final_output = f"Execution failed: {execution_result.get('errors')}"
            print(f"[EXEC] Failed: {final_output}")

        # --- Step 4: Return output ---
        # (We prepare it here, returns at end)
        print(f"Tyrone: {final_output}")

        # --- Step 5: Update Memory (Episodic + Semantic) ---
        # The tools likely updated specific memories.
        # We add an episodic memory of this interaction cycle.
        interaction_summary = f"User: {text} | Intent: {intent} | Result: {final_output}"
        self.memory.store(interaction_summary, memory_type="episodic", importance=0.5)
        print("[MEMORY] Interaction stored.")

        # --- Step 6: Store Reflection if enabled ---
        reflection_data = None
        # We reflect if the interaction was significant or if specifically requested (already handled by intent)
        # Or we can do a post-interaction reflection
        if intent not in ["deterministic", "fact_stored"] and len(text) > 10:
             reflection_data = self.reflector.interpret(f"Reflect on this interaction: {interaction_summary}")
             if reflection_data:
                 print(f"[REFLECTION] {reflection_data}")
                 # Store reflection
                 self.memory.store(str(reflection_data), memory_type="reflection", importance=0.3)

        duration = time.time() - start_time
        return {
            "summary": final_output,
            "decision": route_decision,
            "plan_id": plan.id,
            "duration": duration,
            "reflection": reflection_data
        }

    def run_interactive(self):
        self.running = True
        print("\nTyrone Core Loop is running. Type 'exit' to stop.\n")

        while self.running:
            attention.set_silent(True)
            try:
                text = input("You: ").strip()
            finally:
                attention.set_silent(False)

            if not text:
                continue

            if text.lower() in {"exit", "quit"}:
                print("Exiting core loop.")
                self.running = False
                break

            attention.acquire()
            try:
                self.run_once(text)
            except Exception as e:
                logger.error(f"Error in run_once: {e}", exc_info=True)
                print(f"Error: {e}")
            finally:
                attention.release()

            print("\n---\n")


if __name__ == "__main__":
    tyrone = CoreLoop()
    tyrone.run_interactive()
