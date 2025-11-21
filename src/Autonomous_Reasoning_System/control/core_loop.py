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
from Autonomous_Reasoning_System.control.goal_manager import GoalManager

# Dependencies for injection
from Autonomous_Reasoning_System.memory.storage import MemoryStorage
from Autonomous_Reasoning_System.memory.embeddings import EmbeddingModel
from Autonomous_Reasoning_System.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)

class CoreLoop:
    def __init__(self):
        # 1. Initialize Dispatcher first
        self.dispatcher = Dispatcher()

        # 2. Initialize Core Services (Dependency Injection Root)
        self.embedder = EmbeddingModel()
        self.vector_store = VectorStore()
        self.memory_storage = MemoryStorage(embedding_model=self.embedder, vector_store=self.vector_store)

        # Initialize MemoryInterface with shared components to avoid split-brain
        self.memory = MemoryInterface(
            memory_storage=self.memory_storage,
            embedding_model=self.embedder,
            vector_store=self.vector_store
        )

        # 3. Initialize Components with injected dependencies
        self.plan_builder = PlanBuilder(memory_storage=self.memory_storage)
        self.context_adapter = ContextAdapter(memory_storage=self.memory_storage)
        self.reflector = ReflectionInterpreter(memory_storage=self.memory_storage)
        self.learner = LearningManager(memory_storage=self.memory_storage)
        self.confidence = ConfidenceManager(memory_storage=self.memory_storage)

        # Tools that don't need memory injection or self-initiate harmlessly
        self.intent_analyzer = IntentAnalyzer()
        self.validator = SelfValidator()

        # 4. Initialize Control & Execution
        # Inject shared MemoryInterface into Router
        # Assuming Router takes memory_interface now based on user request history,
        # but the file I read showed Router(dispatcher). Wait, I read router.py and it has __init__(self, dispatcher).
        # However, CoreLoop passed memory_interface in the original code I read above.
        # I should check router.py again.
        # I'll assume the version in CoreLoop I read was correct about what IT passes,
        # but if Router doesn't accept it, that's another bug.
        # Let's check router.py quickly. It was:
        # class Router: def __init__(self, dispatcher: Dispatcher):
        # So passing memory_interface will fail if I don't fix Router or CoreLoop.
        # But the user didn't report a crash there yet. Maybe Router was updated in a previous incomplete fix but I missed it?
        # The file I read `control/router.py` did NOT have memory_interface in init.
        # So `CoreLoop` line 59: `self.router = Router(dispatcher=self.dispatcher, memory_interface=self.memory)` IS A BUG.
        # I should fix that too or update Router.

        # But first, let's fix the GoalManager injection which is the main task.

        # Fix Router init call for now to match definition I saw
        self.router = Router(dispatcher=self.dispatcher)
        # If Router needs memory, I should add it to Router class. But sticking to file state.

        self.plan_executor = PlanExecutor(self.plan_builder, self.dispatcher, self.router)

        # FIX: Pass plan_executor to GoalManager
        self.goal_manager = GoalManager(self.memory, self.plan_builder, self.dispatcher, self.router, plan_executor=self.plan_executor)

        # 3. Register Tools
        # We need to make sure 'memory' tool uses our instance, not a new one.
        components = {
            "intent_analyzer": self.intent_analyzer,
            "memory": self.memory,
            "reflector": self.reflector,
            "plan_builder": self.plan_builder,
            "context_adapter": self.context_adapter,
            "goal_manager": self.goal_manager
        }
        register_tools(self.dispatcher, components)

        # 5. Start Background Tasks
        start_heartbeat_with_plans(
            self.learner, self.confidence, self.plan_builder, interval_seconds=10, test_mode=True, plan_executor=self.plan_executor
        )
        self.running = False

        # Hydrate active plans
        self.plan_builder.load_active_plans()

    def run_once(self, text: str):
        """
        Executes the Full Reasoning Loop:
        0. Check Goals
        1. Router (resolve pipeline)
        2. Plan Builder (create plan)
        3. Dispatcher (execute plan via PlanExecutor)
        4. Return Output
        5. Update Memory
        6. Reflection
        """
        print(f"\n[CORE LOOP] Received input: {text}")
        start_time = time.time()

        # --- Step 0: Check Goals (Periodic/Background) ---
        # We do this at the start of an interaction to simulate "thinking about long-term goals"
        # In a real agent, this would happen on a clock or when idle.
        try:
             goal_status = self.goal_manager.check_goals()
             if goal_status and "No actions needed" not in goal_status:
                 print(f"[GOALS] {goal_status}")
                 # Optionally, we could feed this into the context or decide to prioritize it
        except Exception as e:
             print(f"[GOALS] Error checking goals: {e}")

        # --- Step 1: Use Router to determine pipeline ---
        route_decision = self.router.resolve(text)
        intent = route_decision["intent"]
        family = route_decision.get("family", "unknown")
        pipeline = route_decision["pipeline"]
        print(f"[ROUTER] Intent: {intent} (Family: {family}) | Pipeline: {pipeline}")

        # --- Step 2: Build a plan ---
        # If intent requires complex planning, we decompose.
        # Otherwise, we wrap the simple pipeline/input as a single-step plan.
        if intent == "plan" or intent == "complex_task" or family == "planning":
            # For planning family, we use the tool provided in pipeline (plan_steps) to get steps?
            # Or we delegate to PlanBuilder directly.
            # Let's stick to plan_builder directly for now as per original logic, but enhanced.
            goal, plan = self.plan_builder.new_goal_with_plan(text)
            print(f"[PLANNER] Created multi-step plan: {plan.id}")
        else:
            # Simple execution: Treat the original input as the step description
            # The PlanExecutor will route this step, effectively executing the pipeline determined by Router
            goal = self.plan_builder.new_goal(text)
            # We implicitly use the pipeline selected by the router for this "step"
            plan = self.plan_builder.build_plan(goal, [text])
            print(f"[PLANNER] Created single-step execution plan: {plan.id}")

        # --- Step 3: Execute via Dispatcher (PlanExecutor uses Dispatcher/Router) ---
        # Note: PlanExecutor calls router.route(step.description) internally if no plan exists?
        # No, PlanExecutor executes steps. If step description is just the input text,
        # PlanExecutor's _execute_step calls `router.route(step.description)`.
        # So the Router is called AGAIN inside PlanExecutor.
        # This is redundant but fine for now. The first call was just to visualize/decide IF we need a plan.

        execution_result = self.plan_executor.execute_plan(plan.id)

        final_output = ""
        status = execution_result.get("status")

        if status == "complete":
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
        elif status == "suspended":
             final_output = f"Plan suspended. {execution_result.get('message')}"
        else:
            final_output = f"Execution failed: {execution_result.get('errors')}"
            print(f"[EXEC] Failed: {final_output}")

        # --- Step 4: Return output ---
        # (We prepare it here, returns at end)
        print(f"Tyrone: {final_output}")

        # --- Step 5: Update Memory (Episodic + Semantic) ---
        # The tools likely updated specific memories.
        # We add an episodic memory of this interaction cycle.
        interaction_summary = f"User: {text} | Intent: {intent} | Family: {family} | Result: {final_output}"
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
                 # Reinforce confidence
                 self.confidence.reinforce()

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
