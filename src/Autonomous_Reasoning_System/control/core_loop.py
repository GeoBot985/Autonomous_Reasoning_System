import time
import logging
import asyncio
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
from Autonomous_Reasoning_System.tools.system_tools import get_current_time, get_current_location

# Dependencies for injection
from Autonomous_Reasoning_System.memory.storage import MemoryStorage
from Autonomous_Reasoning_System.memory.embeddings import EmbeddingModel
from Autonomous_Reasoning_System.memory.vector_store import DuckVSSVectorStore

logger = logging.getLogger(__name__)

class CoreLoop:
    def __init__(self, verbose: bool = False):
        # 1. Initialize Dispatcher first
        self.dispatcher = Dispatcher()
        self.verbose = verbose
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        self._stream_subscribers: dict[str, set[asyncio.Queue]] = {}

        # 2. Initialize Core Services (Dependency Injection Root)
        self.embedder = EmbeddingModel()
        self.vector_store = DuckVSSVectorStore()
        self.memory_storage = MemoryStorage(embedding_model=self.embedder, vector_store=self.vector_store)

        # Initialize MemoryInterface with shared components to avoid split-brain
        self.memory = MemoryInterface(
            memory_storage=self.memory_storage,
            embedding_model=self.embedder,
            vector_store=self.vector_store
        )

        # 3. Initialize Components with injected dependencies
        self.plan_builder = PlanBuilder(
            memory_storage=self.memory_storage,
            embedding_model=self.embedder
        )
        self.context_adapter = ContextAdapter(memory_storage=self.memory_storage, embedding_model=self.embedder)

        self.reflector = ReflectionInterpreter(memory_storage=self.memory_storage, embedding_model=self.embedder)
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
        self.clear_stale_state()

    def run_once(self, text: str, plan_id: str | None = None):
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
        logger.info(f"[CORE LOOP] Received input: {text}")
        start_time = time.time()

        # Add metrics
        from Autonomous_Reasoning_System.infrastructure.observability import Metrics
        Metrics().increment("core_loop_cycles")

        # --- Step 0: Check Goals (Periodic/Background) ---
        # We do this at the start of an interaction to simulate "thinking about long-term goals"
        # In a real agent, this would happen on a clock or when idle.
        try:
            goal_status = self.goal_manager.check_goals()
            if goal_status and "No actions needed" not in goal_status:
                logger.debug(f"[GOALS] {goal_status}")
                # Optionally, we could feed this into the context or decide to prioritize it
        except Exception as e:
            logger.error(f"[GOALS] Error checking goals: {e}")

        # --- Step 1: Use Router to determine pipeline ---
        route_decision = self.router.resolve(text)
        intent = route_decision["intent"]
        family = route_decision.get("family", "unknown")
        pipeline = route_decision["pipeline"]
        logger.debug(f"[ROUTER] Intent: {intent} (Family: {family}) | Pipeline: {pipeline}")

        # --- Step 2: Build a plan ---
        # If intent requires complex planning, we decompose.
        # Otherwise, we wrap the simple pipeline/input as a single-step plan.
        if intent == "plan" or intent == "complex_task" or family == "planning":
            # For planning family, we use the tool provided in pipeline (plan_steps) to get steps?
            # Or we delegate to PlanBuilder directly.
            # Let's stick to plan_builder directly for now as per original logic, but enhanced.
            goal, plan = self.plan_builder.new_goal_with_plan(text, plan_id=plan_id)
            logger.debug(f"[PLANNER] Created multi-step plan: {plan.id}")
            self._broadcast_thought(plan.id, f"Plan created with {len(plan.steps)} steps.")
        else:
            # Simple execution: Treat the original input as the step description
            # The PlanExecutor will route this step, effectively executing the pipeline determined by Router
            goal = self.plan_builder.new_goal(text)
            # We implicitly use the pipeline selected by the router for this "step"
            plan = self.plan_builder.build_plan(goal, [text], plan_id=plan_id)
            logger.debug(f"[PLANNER] Created single-step execution plan: {plan.id}")
            self._broadcast_thought(plan.id, "Single-step plan created.")

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
            logger.warning(f"[EXEC] Failed: {final_output}")

        # --- Step 4: Return output ---
        # (We prepare it here, returns at end)
        # print(f"Tyrone: {final_output}") # Using print for UI response as per original design, kept for user visibility in CLI
        # Ideally, this should be handled by the caller (main.py), but CoreLoop prints it directly.
        # The user request was to replace tons of print logs. The final output is arguably UI.
        # I will keep it as print if it's the main response, but maybe log it too.
        logger.info(f"Tyrone response: {final_output}")
        print(f"Tyrone: {final_output}")

        # --- Step 5: Update Memory (Episodic + Semantic) ---
        # The tools likely updated specific memories.
        # We add an episodic memory of this interaction cycle.
        interaction_summary = f"User: {text} | Intent: {intent} | Family: {family} | Result: {final_output}"
        self.memory.store(interaction_summary, memory_type="episodic", importance=0.5)
        logger.debug("[MEMORY] Interaction stored.")

        # --- Step 6: Store Reflection if enabled ---
        reflection_data = None
        # We reflect if the interaction was significant or if specifically requested (already handled by intent)
        # Or we can do a post-interaction reflection
        if intent not in ["deterministic", "fact_stored"] and len(text) > 10:
            reflection_data = self.reflector.interpret(f"Reflect on this interaction: {interaction_summary}")
            if reflection_data:
                logger.debug(f"[REFLECTION] {reflection_data}")
                # Store reflection
                self.memory.store(str(reflection_data), memory_type="reflection", importance=0.3)
                # Reinforce confidence
                self.confidence.reinforce()

        # User corrections get stored explicitly
        lowered_text = text.lower()
        if any(term in lowered_text for term in ["no ", "wrong", "not "]):
            self.memory.remember(
                text=f"USER CORRECTION: {text} → previous answer was wrong",
                metadata={"type": "correction", "importance": 1.0}
            )

        duration = time.time() - start_time
        Metrics().record_time("core_loop_duration", duration)

        result = {
            "summary": final_output,
            "decision": route_decision,
            "plan_id": plan.id,
            "duration": duration,
            "reflection": reflection_data,
            # Legacy keys for tests
            "reflection_data": reflection_data
        }

        self._broadcast_thought(plan.id, f"Plan status: {status}. Output: {final_output}")
        self._send_to_user(final_output)
        return result

    def initialize_context(self):
        """Initializes the context with system information (time, location)."""
        # Find Feet (Initialize Context)
        try:
            current_time = get_current_time()
            current_location = get_current_location()
            logger.info(f"[STARTUP] Feet found: {current_location} at {current_time}")
            self.context_adapter.set_startup_context({
                "Current Time": current_time,
                "Current Location": current_location
            })
        except Exception as e:
            logger.error(f"[STARTUP] Failed to find feet: {e}")

    def run_interactive(self):
        self.initialize_context()
        self.running = True
        logger.info("Tyrone Core Loop is running. Type 'exit' to stop.")

        while self.running:
            attention.set_silent(True)
            try:
                text = input("You: ").strip()
            finally:
                attention.set_silent(False)

            if not text:
                continue

            if text.lower() in {"exit", "quit"}:
                logger.info("Exiting core loop.")
                self.running = False
                break

            attention.acquire()
            try:
                self.run_once(text)
            except Exception as e:
                logger.error(f"Error in run_once: {e}", exc_info=True)
            finally:
                attention.release()

            logger.debug("---")

    # ------------------------------------------------------------------
    # API helpers for background execution and streaming
    # ------------------------------------------------------------------
    def clear_stale_state(self):
        """Remove stale/unfinished goals from prior sessions to start clean."""
        try:
            with self.memory_storage._write_lock:
                self.memory_storage.con.execute(
                    "DELETE FROM goals WHERE status NOT IN ('completed', 'failed')"
                )
            logger.info("Cleared stale plans from previous sessions.")
        except Exception as e:
            logger.error(f"Failed to clear stale plans: {e}")

    def _send_to_user(self, message: str):
        """Send user-facing messages while filtering internal progress spam."""
        if not message:
            return
        spam_markers = [
            "Plan update",
            "step",
            "Current step: None",
            "Reminder: Continue plan",
            "Last action result",
            "0/1 steps complete",
            "%. Current step:",
        ]
        if any(marker in message for marker in spam_markers):
            logger.debug(f"[INTERNAL] {message}")
            return
        print(f"Tyrone> {message}")
        # Broadcast to any stream queues if present
        for queues in self._stream_subscribers.values():
            for q in queues:
                try:
                    q.put_nowait(message)
                except Exception:
                    continue

    def run_background(self, goal: str, plan_id: str):
        """Run a goal asynchronously for API calls."""
        asyncio.get_event_loop().create_task(self._run_goal_async(goal, plan_id))

    async def _run_goal_async(self, goal: str, plan_id: str):
        result = await asyncio.to_thread(self.run_once, goal, plan_id)
        self._broadcast_thought(plan_id, f"Completed plan {plan_id}")
        # Signal end
        self._broadcast_thought(plan_id, None)
        return result

    def get_plan_status(self, plan_id: str):
        """Return plan progress summary if available."""
        summary = self.plan_builder.get_plan_summary(plan_id)
        if summary and "error" not in summary:
            return summary
        return None

    def subscribe_stream(self, plan_id: str, queue: asyncio.Queue):
        self._stream_subscribers.setdefault(plan_id, set()).add(queue)

    def unsubscribe_stream(self, plan_id: str):
        if plan_id in self._stream_subscribers:
            for q in list(self._stream_subscribers[plan_id]):
                try:
                    q.put_nowait(None)
                except Exception:
                    pass
            del self._stream_subscribers[plan_id]

    def _broadcast_thought(self, plan_id: str | None, thought: str | None):
        """Push messages to SSE subscribers."""
        if not plan_id or plan_id not in self._stream_subscribers:
            return
        # Filter spammy internal chatter
        if thought:
            spam_markers = [
                "Plan update",
                "step",
                "Current step: None",
                "Reminder: Continue plan",
                "Last action result",
                "0/1 steps complete",
                "%. Current step:",
            ]
            if any(marker in thought for marker in spam_markers):
                logger.debug(f"[INTERNAL] {thought}")
                return
        for q in list(self._stream_subscribers.get(plan_id, [])):
            try:
                q.put_nowait(thought)
            except Exception:
                continue


if __name__ == "__main__":
    tyrone = CoreLoop()
    tyrone.run_interactive()
