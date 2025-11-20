import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def register_tools(dispatcher, components: Dict[str, Any]):
    """
    Registers standard tools with the dispatcher, binding them to the provided components.

    Args:
        dispatcher: The Dispatcher instance.
        components: A dictionary containing instances of system components:
            - "intent_analyzer"
            - "memory"
            - "reflector"
            - "plan_builder"
            - "deterministic_responder" (optional, if not present, will be created)
            - "context_adapter" (optional)
    """

    # 1. Analyze Intent
    def analyze_intent_handler(text: str, **kwargs):
        analyzer = components.get("intent_analyzer")
        if analyzer:
            return analyzer.analyze(text)
        return {"intent": "unknown"}

    dispatcher.register_tool(
        "analyze_intent",
        analyze_intent_handler,
        schema={"text": {"type": str, "required": True}}
    )

    # 2. Store Memory
    def store_memory_handler(text: str, **kwargs):
        memory = components.get("memory")
        if memory:
            memory.store(f"Stored fact: {text}", memory_type="episodic", importance=1.0)
            return f"Stored: {text}"
        return "Memory component not available."

    dispatcher.register_tool(
        "store_memory",
        store_memory_handler,
        schema={"text": {"type": str, "required": True}}
    )

    # 3. Search Memory
    def search_memory_handler(text: str, **kwargs):
        memory = components.get("memory")
        if memory:
            # Assuming retrieve method exists and returns list
            results = memory.retrieve(text, k=3)
            return results
        return "Memory component not available."

    dispatcher.register_tool(
        "search_memory",
        search_memory_handler,
        schema={"text": {"type": str, "required": True}}
    )

    # 4. Perform Reflection
    def perform_reflection_handler(text: str, **kwargs):
        reflector = components.get("reflector")
        if reflector:
            return reflector.interpret(text)
        return "Reflector component not available."

    dispatcher.register_tool(
        "perform_reflection",
        perform_reflection_handler,
        schema={"text": {"type": str, "required": True}}
    )

    # 5. Summarize Context
    def summarize_context_handler(text: str, **kwargs):
        # This might use memory summarization or just reflection
        reflector = components.get("reflector")
        if reflector:
            return reflector.interpret(f"Summarize this: {text}")
        return "Reflector component not available."

    dispatcher.register_tool(
        "summarize_context",
        summarize_context_handler,
        schema={"text": {"type": str, "required": True}}
    )

    # 6. Deterministic Responder
    def deterministic_responder_handler(text: str, **kwargs):
        responder = components.get("deterministic_responder")
        if not responder:
            from Autonomous_Reasoning_System.tools.deterministic_responder import DeterministicResponder
            responder = DeterministicResponder()
        return responder.run(text)

    dispatcher.register_tool(
        "deterministic_responder",
        deterministic_responder_handler,
        schema={"text": {"type": str, "required": True}}
    )

    # 7. Plan Steps (decompose goal)
    def plan_steps_handler(text: str, goal: str = None, **kwargs):
        plan_builder = components.get("plan_builder")
        target_text = goal or text
        if plan_builder:
            steps = plan_builder.decompose_goal(target_text)
            return steps
        return ["No PlanBuilder available"]

    dispatcher.register_tool(
        "plan_steps",
        plan_steps_handler,
        schema={"text": {"type": str, "required": True}}
    )

    # 8. Answer Question (Generic LLM / ContextAdapter)
    def answer_question_handler(text: str, **kwargs):
        adapter = components.get("context_adapter")
        if not adapter:
            # Lazy import if not provided
            try:
                from Autonomous_Reasoning_System.llm.context_adapter import ContextAdapter
                adapter = ContextAdapter()
            except ImportError:
                pass

        if adapter:
            return adapter.run(text)
        return "I cannot answer that right now."

    dispatcher.register_tool(
        "answer_question",
        answer_question_handler,
        schema={"text": {"type": str, "required": True}}
    )

    logger.info("Standard tools registered successfully.")
