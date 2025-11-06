import time
from datetime import datetime
from Autonomous_Reasoning_System.cognition.router import Router
from Autonomous_Reasoning_System.cognition.intent_analyzer import IntentAnalyzer
from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface
from Autonomous_Reasoning_System.llm.reflection_interpreter import ReflectionInterpreter
from Autonomous_Reasoning_System.memory.confidence_manager import ConfidenceManager
from Autonomous_Reasoning_System.cognition.self_validator import SelfValidator
from Autonomous_Reasoning_System.cognition.learning_manager import LearningManager
from Autonomous_Reasoning_System.control.scheduler import start_heartbeat_with_plans
from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder
from Autonomous_Reasoning_System.control.attention_manager import attention


class CoreLoop:
    """
    Tyrone's central cognitive cycle.
    - Waits for input (from console or external receiver)
    - Routes it via the Router
    - Executes each module in the chosen pipeline
    - Logs and reflects on results
    """

    def __init__(self):
        self.router = Router()
        self.memory = MemoryInterface()
        self.intent_analyzer = IntentAnalyzer()
        self.reflector = ReflectionInterpreter()
        self.confidence = ConfidenceManager()
        self.validator = SelfValidator()
        self.learner = LearningManager()
        self.plan_builder = PlanBuilder()

        # Start background heartbeat for autonomous summarisation
        start_heartbeat_with_plans(
            self.learner, self.confidence, self.plan_builder, interval_seconds=10, test_mode=True
        )
        self.running = False

    def run_once(self, text: str):
        """Process a single input string through Tyrone’s cognition cycle."""
        print(f"\n[🧠 CORE LOOP] Received input: {text}")

        # 1️⃣ Route input
        decision = self.router.route(text)
        print(f"[🧭 ROUTER] Intent: {decision['intent']} | Pipeline: {decision['pipeline']}")

        # 2️⃣ Execute modules sequentially
        results = []
        intent_data = None
        reflection_data = None

        for module in decision["pipeline"]:
            print(f"[⚙️ EXEC] {module}() ...")
            time.sleep(0.2)

            if module == "IntentAnalyzer":
                intent_data = self.intent_analyzer.analyze(text)
                print(f"[🧩 IntentAnalyzer] → {intent_data}")
                results.append(f"IntentAnalyzer → {intent_data['intent']}")

            elif module == "MemoryInterface":
                # Record what just happened
                if intent_data:
                    summary = f"Intent: {intent_data['intent']} | Entities: {intent_data.get('entities', {})}"
                else:
                    summary = f"Intent: {decision['intent']} | Modules: {decision['pipeline']}"
                self.memory.store(summary)
                print(f"[💾 MEMORY] Logged experience → {summary}")
                results.append("Memory stored")

            elif module == "RetrievalOrchestrator":
                from Autonomous_Reasoning_System.memory.retrieval_orchestrator import RetrievalOrchestrator
                retriever = RetrievalOrchestrator()
                retrieved = retriever.retrieve(text)

                # --- Safe handling for lists, DataFrames, or None ---
                count = 0
                if retrieved is not None:
                    if hasattr(retrieved, "shape"):  # likely a DataFrame
                        count = len(retrieved.index)
                    elif isinstance(retrieved, (list, tuple, set)):
                        count = len(retrieved)
                    else:
                        try:
                            count = len(retrieved)
                        except Exception:
                            count = 1

                if count > 0:
                    # Extract sample text for reflection
                    try:
                        self.last_retrieved_text = " ".join(
                            [r[0] if isinstance(r, tuple) else str(r) for r in list(retrieved)[:5]]
                        )
                    except Exception:
                        self.last_retrieved_text = str(retrieved)
                    print(f"[🧠 RETRIEVAL RESULTS] {count} items found.")
                else:
                    self.last_retrieved_text = None
                    print("[🧠 RETRIEVAL RESULTS] None found.")

                results.append(f"RetrievalOrchestrator → {count} results")

            elif module == "ContextAdapter":
                # 🗣️ Tyrone's speaking stage
                from Autonomous_Reasoning_System.llm.context_adapter import ContextAdapter
                adapter = ContextAdapter()
                reply = adapter.run(text)
                print(f"Tyrone: {reply}")
                results.append(f"ContextAdapter → {reply}")

            elif module == "DeterministicResponder":
                from Autonomous_Reasoning_System.tools.deterministic_responder import DeterministicResponder
                responder = DeterministicResponder()
                reply = responder.run(text)
                print(f"Tyrone: {reply}")
                results.append(f"DeterministicResponder → {reply}")


            elif module == "ReflectionInterpreter":
                reflection_input = getattr(self, "last_retrieved_text", None) or text
                reflection_data = self.reflector.interpret(reflection_input)

                if isinstance(reflection_data, dict):
                    summary = reflection_data.get("summary", "")
                    insight = reflection_data.get("insight", "")
                    combined = f"{summary} {insight}".strip()
                else:
                    combined = str(reflection_data)

                # 🧩 Combine with last ContextAdapter reply if available
                try:
                    if results and results[-1].startswith("ContextAdapter"):
                        last_reply = results[-1].split("→", 1)[1].strip()
                        merged_reply = f"{last_reply}\n\n(Reflection) {combined}"
                        print(f"Tyrone: {merged_reply}")
                        results[-1] = f"ContextAdapter+Reflection → {merged_reply}"
                    else:
                        print(f"Tyrone: {combined}")
                except Exception:
                    print(f"Tyrone: {combined}")


                print(f"[🪞 ReflectionInterpreter] → {reflection_data}")
                results.append(f"ReflectionInterpreter → {reflection_data}")

                change = reflection_data.get("confidence_change", "neutral") if isinstance(reflection_data, dict) else "neutral"
                if change == "positive":
                    self.confidence.reinforce()
                    print("[📈 CONFIDENCE] Reinforced based on positive reflection.")
                elif change == "negative":
                    self.confidence.decay_all()
                    print("[📉 CONFIDENCE] Decreased confidence based on negative reflection.")
                else:
                    print("[⚖️ CONFIDENCE] No change from reflection.")

            else:
                results.append(f"{module} executed")

        # 3️⃣ Store final summary of this cycle
        final_summary = (
            f"Intent: {decision['intent']} | Modules: {decision['pipeline']} | "
            f"Reason: {decision.get('reason', '')}"
        )

        # 4️⃣ Self-evaluation and learning integration
        try:
            meta = {
                "intent": decision.get("intent"),
                "confidence": self.confidence.get_overall_confidence()
                if hasattr(self.confidence, "get_overall_confidence")
                else 0.5,
            }

            result = self.validator.evaluate(
                input_text=text,
                output_text=reflection_data.get("summary") if reflection_data else final_summary,
                meta=meta,
            )

            self.learner.ingest(result)

            if len(self.learner.experience_buffer) % 10 == 0:
                summary = self.learner.summarise_recent(window_minutes=120)
                print(f"[📚 LEARNING SUMMARY] {summary['summary']}")
        except Exception as e:
            print(f"[⚠️ LEARNING ERROR] {e}")

        self.memory.store(final_summary)
        print(f"[🧠 SUMMARY] {final_summary}")

        return {
            "decision": decision,
            "intent_data": intent_data,
            "reflection_data": reflection_data,
            "summary": final_summary,
        }

    def run_interactive(self):
        """Run an interactive session in the console."""
        self.running = True
        print("\n🤖 Tyrone Core Loop is running. Type 'exit' to stop.\n")

        while self.running:
            # 🔇 suppress attention/heartbeat prints while typing
            attention.set_silent(True)
            try:
                text = input("You: ").strip()
            finally:
                # once user pressed Enter, re-enable attention messages
                attention.set_silent(False)

            if text.lower() in {"exit", "quit"}:
                print("🛑 Exiting core loop.")
                self.running = False
                break

            # 🧭 acquire attention only after the full input is entered
            attention.acquire()
            try:
                self.run_once(text)
            finally:
                attention.release()

            print("\n---\n")
            time.sleep(0.1)



if __name__ == "__main__":
    from Autonomous_Reasoning_System.control.core_loop import CoreLoop

    tyrone = CoreLoop()
    tyrone.running = True
    print("\n🤖 Tyrone Core Loop ready. Type 'exit' to stop.\n")

    tyrone.run_interactive()
