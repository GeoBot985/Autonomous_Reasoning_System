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
            - "goal_manager" (optional)
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
            memory.remember(
                text=f"Stored fact: {text}",
                metadata={"type": "personal_fact", "importance": 1.0, "source": "tool:store_memory"}
            )
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

    # 9. Goal Management
    def create_goal_handler(text: str, priority: int = 1, **kwargs):
        goal_manager = components.get("goal_manager")
        if goal_manager:
            return goal_manager.create_goal(text, priority=priority)
        return "Goal Manager not available."

    dispatcher.register_tool(
        "create_goal",
        create_goal_handler,
        schema={
            "text": {"type": str, "required": True},
            "priority": {"type": int, "required": False}
        }
    )

    def list_goals_handler(status: str = None, **kwargs):
        goal_manager = components.get("goal_manager")
        if goal_manager:
             # We access memory directly or via goal manager helper
             # GoalManager doesn't expose list directly, but memory does
             active_goals = goal_manager.memory.get_active_goals()
             if active_goals.empty:
                 return "No active goals."

             if status:
                 active_goals = active_goals[active_goals['status'] == status]
                 if active_goals.empty:
                     return f"No goals with status '{status}'."

             summary = []
             for _, row in active_goals.iterrows():
                 summary.append(f"[{row['id'][:8]}] {row['text']} (Status: {row['status']})")
             return "\n".join(summary)
        return "Goal Manager not available."

    dispatcher.register_tool(
        "list_goals",
        list_goals_handler,
        schema={
            "status": {"type": str, "required": False}
        }
    )

    # --- NEW CONTROLLER TOOLS FOR FAMILIES ---

    # 10. Handle Memory Ops (Unified)
    def handle_memory_ops_handler(text: str, intent: str = None, context: Dict[str, Any] = None, **kwargs):
        # Check intent from args or context if available
        effective_intent = intent
        if not effective_intent and context:
             effective_intent = context.get("intent", "unknown")

        if not effective_intent:
             effective_intent = "unknown"

        memory = components.get("memory")

        if not memory:
            return "Memory component not available."

        # Dispatch based on intent
        if effective_intent in ["store", "save", "remind", "remember", "memorize"]:
             memory.remember(
                 text=f"Stored fact: {text}",
                 metadata={"type": "personal_fact", "importance": 1.0, "source": "tool:handle_memory_ops"}
             )
             return f"Stored: {text}"
        elif effective_intent in ["search", "recall", "find", "lookup"]:
             results = memory.retrieve(text, k=3)
             return results
        else:
             # Default to search if unknown intent in memory family
             results = memory.retrieve(text, k=3)
             return results

    dispatcher.register_tool(
        "handle_memory_ops",
        handle_memory_ops_handler,
        schema={
            "text": {"type": str, "required": True},
            "intent": {"type": str, "required": False}
        }
    )

    # 11. Handle Goal Ops (Unified)
    def handle_goal_ops_handler(text: str, context: Dict[str, Any] = None, **kwargs):
        intent = context.get("intent", "unknown") if context else "unknown"
        goal_manager = components.get("goal_manager")

        if not goal_manager:
             return "Goal Manager not available."

        if intent in ["list_goals", "goals"]:
             # Reuse logic or call internal method
             # Just calling the helper for now:
             active_goals = goal_manager.memory.get_active_goals()
             if active_goals.empty:
                 return "No active goals."
             summary = []
             for _, row in active_goals.iterrows():
                 summary.append(f"[{row['id'][:8]}] {row['text']} (Status: {row['status']})")
             return "\n".join(summary)
        else:
             # Default to creating goal for other intents (create_goal, achieve, do, task, research, investigate, etc.)
             return goal_manager.create_goal(text)

    dispatcher.register_tool(
        "handle_goal_ops",
        handle_goal_ops_handler,
        schema={"text": {"type": str, "required": True}}
    )

    # 12. Perform Self Analysis
    def perform_self_analysis_handler(text: str, **kwargs):
        reflector = components.get("reflector")
        if reflector:
             # We might want to check system status here
             return reflector.interpret(f"Analyze system status and self: {text}")
        return "Reflector not available."

    dispatcher.register_tool(
        "perform_self_analysis",
        perform_self_analysis_handler,
        schema={"text": {"type": str, "required": True}}
    )


    logger.info("Standard tools registered successfully.")
