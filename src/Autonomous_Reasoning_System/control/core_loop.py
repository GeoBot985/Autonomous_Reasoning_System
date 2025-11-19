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
    def __init__(self):
        self.router = Router()
        self.memory = MemoryInterface()
        self.intent_analyzer = IntentAnalyzer()
        self.reflector = ReflectionInterpreter()
        self.confidence = ConfidenceManager()
        self.validator = SelfValidator()
        self.learner = LearningManager()
        self.plan_builder = PlanBuilder()

        start_heartbeat_with_plans(
            self.learner, self.confidence, self.plan_builder, interval_seconds=10, test_mode=True
        )
        self.running = False

    def run_once(self, text: str):
        print(f"\n[CORE LOOP] Received input: {text}")

        decision = self.router.route(text)
        print(f"[ROUTER] Intent: {decision['intent']} | Pipeline: {decision['pipeline']}")

        # SPECIAL CASE: Direct fact storage — we skip everything and just confirm
        if decision["intent"] == "fact_stored":
            reply = "Got it. I'll remember that forever. ❤️"
            print(f"Tyrone: {reply}")
            self.memory.store(f"Stored fact: {text}", memory_type="personal_fact", importance=1.0)
            print(f"[SUMMARY] Intent: fact_stored | Reason: {decision.get('reason')}")
            return {"summary": reply}

        # Normal pipeline execution
        intent_data = None
        reflection_data = None
        reply = None

        for module in decision["pipeline"]:
            print(f"[EXEC] {module}() ...")
            time.sleep(0.2)

            if module == "IntentAnalyzer":
                intent_data = self.intent_analyzer.analyze(text)
                print(f"[IntentAnalyzer] → {intent_data}")

            elif module == "ContextAdapter":
                from Autonomous_Reasoning_System.llm.context_adapter import ContextAdapter
                adapter = ContextAdapter()
                reply = adapter.run(text)
                print(f"Tyrone: {reply}")

            elif module == "ReflectionInterpreter":
                # Only reflect if we have something meaningful
                reflect_on = reply or text
                reflection_data = self.reflector.interpret(reflect_on)
                print(f"[ReflectionInterpreter] → {reflection_data}")

        # Final reply to user
        final_reply = reply or "Done."
        if reflection_data and decision["intent"] != "fact_stored":
            insight = reflection_data.get("insight", "") if isinstance(reflection_data, dict) else str(reflection_data)
            if insight.strip():
                final_reply += f"\n\n(Reflection) {insight}"

        # Store the cycle summary
        final_summary = f"Intent: {decision['intent']} | Reason: {decision.get('reason', '')}"
        self.memory.store(final_summary)
        print(f"[SUMMARY] {final_summary}")

        return {"summary": final_reply}

    def run_interactive(self):
        self.running = True
        print("\nTyrone Core Loop is running. Type 'exit' to stop.\n")

        while self.running:
            attention.set_silent(True)
            try:
                text = input("You: ").strip()
            finally:
                attention.set_silent(False)

            if text.lower() in {"exit", "quit"}:
                print("Exiting core loop.")
                self.running = False
                break

            attention.acquire()
            try:
                result = self.run_once(text)
                if "summary" in result:
                    print(f"Tyrone: {result['summary']}")
            finally:
                attention.release()

            print("\n---\n")


if __name__ == "__main__":
    tyrone = CoreLoop()
    tyrone.run_interactive()