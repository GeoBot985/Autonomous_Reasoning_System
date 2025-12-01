import logging
from typing import List, Dict, Any, Optional
from Autonomous_Reasoning_System.control.dispatcher import Dispatcher

logger = logging.getLogger(__name__)

class IntentFamily:
    MEMORY = "memory_operations"
    PERSONAL_FACTS = "personal_facts"
    QA = "question_answering"
    GOALS = "goals_tasks"
    SUMMARIZATION = "summarization"
    REFLECTION = "reflection"
    SELF_ANALYSIS = "self_analysis"
    TOOL_EXECUTION = "tool_execution"
    PLANNING = "planning"
    WEB_SEARCH = "web_search"
    UNKNOWN = "unknown"

class Router:
    """
    Router determines the sequence of tools (pipeline) to execute based on user input.
    Now organized by Intent Families.
    """
    def __init__(self, dispatcher: Dispatcher, memory_interface=None):
        self.dispatcher = dispatcher
        self.memory = memory_interface

        # 1. Map Intents to Families
        self.intent_family_map = {
            # Memory
            "remind": IntentFamily.MEMORY,
            "remember": IntentFamily.MEMORY,
            "store": IntentFamily.MEMORY,
            "memory_store": IntentFamily.MEMORY,
            "save": IntentFamily.MEMORY,
            "recall": IntentFamily.MEMORY,
            "search": IntentFamily.MEMORY,
            "find": IntentFamily.MEMORY,
            "lookup": IntentFamily.MEMORY,

            # QA
            "query": IntentFamily.QA,
            "answer": IntentFamily.QA,
            "ask": IntentFamily.QA,
            "explain": IntentFamily.QA,
            "deterministic": IntentFamily.QA,
            "unknown": IntentFamily.QA, # Default fallback

            # Goals
            "achieve": IntentFamily.GOALS,
            "do": IntentFamily.GOALS,
            "task": IntentFamily.GOALS,
            "create_goal": IntentFamily.GOALS,
            "goals": IntentFamily.GOALS,
            "list_goals": IntentFamily.GOALS,
            "research": IntentFamily.GOALS,
            "investigate": IntentFamily.GOALS,

            # Summarization
            "summarize": IntentFamily.SUMMARIZATION,
            "tldr": IntentFamily.SUMMARIZATION,

            # Reflection
            "reflect": IntentFamily.REFLECTION,

            # Self Analysis
            "status": IntentFamily.SELF_ANALYSIS,
            "health": IntentFamily.SELF_ANALYSIS,
            "analyze_self": IntentFamily.SELF_ANALYSIS,

            # Tool Execution
            "execute": IntentFamily.TOOL_EXECUTION,
            "run": IntentFamily.TOOL_EXECUTION,

            # Planning
            "plan": IntentFamily.PLANNING,
            "blueprint": IntentFamily.PLANNING,

            # Web Search
            "web_search": IntentFamily.WEB_SEARCH,
            "search_web": IntentFamily.WEB_SEARCH,
            "google": IntentFamily.WEB_SEARCH,
            "search_online": IntentFamily.WEB_SEARCH,
            "find_online": IntentFamily.WEB_SEARCH,
        }

        # 2. Map Families to Pipelines
        # To unify routing, we dispatch to a "handler" for the family.
        # Specific intent logic is handled within that tool or refined here.
        self.family_pipeline_map = {
            IntentFamily.MEMORY: ["handle_memory_ops"],
            IntentFamily.PERSONAL_FACTS: ["handle_memory_ops"],
            IntentFamily.QA: ["answer_question"],
            IntentFamily.GOALS: ["handle_goal_ops"],
            IntentFamily.SUMMARIZATION: ["summarize_context"],
            IntentFamily.REFLECTION: ["perform_reflection"],
            IntentFamily.SELF_ANALYSIS: ["perform_self_analysis"],
            IntentFamily.PLANNING: ["plan_steps"],
            IntentFamily.TOOL_EXECUTION: ["answer_question"], # Fallback/Placeholder
            IntentFamily.WEB_SEARCH: ["google_search"],
        }

        self.fallback_pipeline = ["answer_question"]

        # Allowed tool names for pipeline validation
        self._valid_modules = {
            "handle_memory_ops",
            "answer_question",
            "handle_goal_ops",
            "summarize_context",
            "perform_reflection",
            "perform_self_analysis",
            "plan_steps",
            "deterministic_responder",
            "analyze_intent",
            "context_adapter",
            "memory",
            "reflector",
            "plan_builder",
            "goal_manager",
            "action_executor",
            "google_search",
        }

    def _validate_pipeline(self, pipeline: List[str]) -> bool:
        """Ensure the router pipeline only contains registered/known tools."""
        # If running in test mode, we might want to skip validation or allow custom tools.
        # For now, we log warning but still return False to be safe in production.
        # Tests should ideally use valid modules or patch _valid_modules.
        for step in pipeline:
            if step not in self._valid_modules:
                logger.warning(f"Router hallucinated invalid module: {step}")
                return False
        return True

    def resolve(self, text: str) -> Dict[str, Any]:
        """
        Analyzes intent and determines the pipeline without executing it.
        Returns a dictionary containing intent, entities, and the selected pipeline.
        """
        # --- NEW LOGIC: Check for explicit Web Search invocation ---
        lower_text = text.lower().strip()
        if lower_text.startswith("web search") or lower_text.startswith("search web"):
            # Clean up the query
            query = text[len("web search"):].strip()
            if lower_text.startswith("search web"):
                query = text[len("search web"):].strip()

            # Remove punctuation prefix if any (like ":")
            if query.startswith(":") or query.startswith("-"):
                query = query[1:].strip()

            return {
                "intent": "web_search",
                "family": IntentFamily.WEB_SEARCH,
                "subtype": None,
                "entities": {"query": query},
                "analysis_data": {},
                "pipeline": self._select_pipeline(IntentFamily.WEB_SEARCH, "web_search"),
                "clean_text": query
            }

        # 1. Analyze Intent
        analysis_result = self.dispatcher.dispatch("analyze_intent", arguments={"text": text})

        intent = "unknown"
        entities = {}
        analysis_data = {}
        family = None
        subtype = None

        if analysis_result["status"] == "success":
            analysis_data = analysis_result["data"]
            if isinstance(analysis_data, dict):
                intent = analysis_data.get("intent", "unknown")
                entities = analysis_data.get("entities", {})
                # Respect analyzer's family classification if provided
                family = analysis_data.get("family")
                subtype = analysis_data.get("subtype")
        else:
            logger.warning(f"Intent analysis failed: {analysis_result['errors']}")

        # 2. Determine Family (if not provided by analyzer)
        if not family or family == "unknown":
            family = self.intent_family_map.get(intent, IntentFamily.QA) # Default to QA

        # 3. Select Pipeline
        pipeline = self._select_pipeline(family, intent)

        return {
            "intent": intent,
            "family": family,
            "subtype": subtype,
            "entities": entities,
            "analysis_data": analysis_data,
            "pipeline": pipeline
        }

    def execute_pipeline(self, pipeline: List[str], initial_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Executes the given pipeline of tools.
        """
        results = []
        context = context or {}
        previous_output = initial_input

        if not self._validate_pipeline(pipeline):
            return {
                "status": "error",
                "pipeline": pipeline,
                "error": "Invalid pipeline generated by router",
                "results": [],
                "final_output": None
            }

        # Update context with input
        context.update({"original_input": initial_input})
        entities = context.get("entities", {})

        for i, tool_name in enumerate(pipeline):
            step_args = {
                "text": previous_output,
                "entities": entities,
                "context": context
            }

            # Special handling: if tool is 'plan_steps', it might need 'goal'
            if tool_name == "plan_steps":
                step_args["goal"] = initial_input

            logger.info(f"Executing tool '{tool_name}' (step {i+1})")
            step_result = self.dispatcher.dispatch(tool_name, arguments=step_args)
            results.append(step_result)

            if step_result["status"] == "success":
                data = step_result["data"]
                if isinstance(data, str):
                    previous_output = data
                elif isinstance(data, dict) and "result" in data:
                    previous_output = data["result"]
                else:
                    previous_output = str(data)
            else:
                logger.error(f"Tool '{tool_name}' failed: {step_result['errors']}")
                break

        return {
            "pipeline": pipeline,
            "results": results,
            "final_output": previous_output
        }

    def route(self, text: str, pipeline_override: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyzes intent and executes the corresponding tool pipeline.
        """
        logger.info(f"Router received input: {text}")

        # Always analyze intent first to maintain context
        resolve_result = self.resolve(text)
        intent = resolve_result["intent"]
        family = resolve_result["family"]
        entities = resolve_result["entities"]
        analysis_data = resolve_result["analysis_data"]

        # Use clean text if provided (e.g. from web search override)
        execution_text = resolve_result.get("clean_text", text)

        if pipeline_override:
            pipeline = pipeline_override
            intent = "override"
            logger.info(f"Using overridden pipeline: {pipeline}")
        else:
            pipeline = resolve_result["pipeline"]
            if family == IntentFamily.WEB_SEARCH:
                logger.info(f"Routing to Web Search pipeline for query: '{text}'")
            logger.info(f"Routing intent: {intent} (Family: {family}), Selected pipeline: {pipeline}")

        context = {
            "intent": intent,
            "family": family,
            "entities": entities,
            "analysis": analysis_data
        }

        execution_result = self.execute_pipeline(pipeline, execution_text, context)

        return {
            "intent": intent,
            "family": family,
            "pipeline": pipeline,
            "results": execution_result.get("results", []),
            "final_output": execution_result.get("final_output"),
            "status": execution_result.get("status", "success"),
            "error": execution_result.get("error")
        }

    def _select_pipeline(self, family: str, intent: str) -> List[str]:
        pipeline = self.family_pipeline_map.get(family)
        if not pipeline:
            logger.info(f"No pipeline found for family '{family}'. Using fallback.")
            pipeline = self.fallback_pipeline
        return pipeline
