import logging
from typing import List, Dict, Any, Optional
from Autonomous_Reasoning_System.control.dispatcher import Dispatcher

logger = logging.getLogger(__name__)

class Router:
    """
    Router determines the sequence of tools (pipeline) to execute based on user input.
    """
    def __init__(self, dispatcher: Dispatcher):
        self.dispatcher = dispatcher
        self.intent_pipeline_map = {
            "remind": ["store_memory"],
            "remember": ["store_memory"],
            "reflect": ["perform_reflection"],
            "summarize": ["summarize_context"],
            "recall": ["search_memory"],
            "search": ["search_memory"],
            "plan": ["plan_steps"],
            "query": ["answer_question"],
            "answer": ["answer_question"],
            "deterministic": ["deterministic_responder"],
        }
        # Default fallback pipeline
        self.fallback_pipeline = ["answer_question"]

    def resolve(self, text: str) -> Dict[str, Any]:
        """
        Analyzes intent and determines the pipeline without executing it.
        Returns a dictionary containing intent, entities, and the selected pipeline.
        """
        # 1. Analyze Intent
        analysis_result = self.dispatcher.dispatch("analyze_intent", arguments={"text": text})

        intent = "unknown"
        entities = {}
        analysis_data = {}

        if analysis_result["status"] == "success":
            analysis_data = analysis_result["data"]
            if isinstance(analysis_data, dict):
                intent = analysis_data.get("intent", "unknown")
                entities = analysis_data.get("entities", {})
        else:
            logger.warning(f"Intent analysis failed: {analysis_result['errors']}")

        # 2. Select Pipeline
        pipeline = self._select_pipeline(intent)

        return {
            "intent": intent,
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
        This preserves backward compatibility.
        """
        logger.info(f"Router received input: {text}")

        # Always analyze intent first to maintain context
        resolve_result = self.resolve(text)
        intent = resolve_result["intent"]
        entities = resolve_result["entities"]
        analysis_data = resolve_result["analysis_data"]

        if pipeline_override:
            pipeline = pipeline_override
            intent = "override" # Or keep original intent? The test implies we just want the analysis call.
            # But if we override, we probably want to suppress the original intent driving the pipeline.
            # Let's keep intent as "override" or just use the override pipeline.
            logger.info(f"Using overridden pipeline: {pipeline}")
        else:
            pipeline = resolve_result["pipeline"]
            logger.info(f"Routing intent: {intent}, Selected pipeline: {pipeline}")

        context = {
            "intent": intent,
            "entities": entities,
            "analysis": analysis_data
        }

        execution_result = self.execute_pipeline(pipeline, text, context)

        return {
            "intent": intent,
            "pipeline": pipeline,
            "results": execution_result["results"],
            "final_output": execution_result["final_output"]
        }

    def _select_pipeline(self, intent: str) -> List[str]:
        pipeline = self.intent_pipeline_map.get(intent)
        if not pipeline:
            logger.info(f"No pipeline found for intent '{intent}'. Using fallback.")
            pipeline = self.fallback_pipeline
        return pipeline
