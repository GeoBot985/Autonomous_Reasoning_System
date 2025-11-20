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

    def route(self, text: str, pipeline_override: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyzes intent and executes the corresponding tool pipeline.
        Args:
            text: The user input string.
            pipeline_override: Optional list of tool names to force execute, bypassing intent analysis pipeline selection.
        """
        logger.info(f"Router received input: {text}")

        # 1. Analyze Intent
        # We execute "analyze_intent" as a tool via the dispatcher.
        # We assume this tool is registered and returns a dict with "intent" key.
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
            # If intent analysis fails, we might still want to proceed with a fallback or unknown intent

        logger.info(f"Routing intent: {intent}")

        # 2. Select Pipeline
        if pipeline_override:
            pipeline = pipeline_override
            logger.info(f"Using overridden pipeline: {pipeline}")
        else:
            pipeline = self._select_pipeline(intent)
            logger.info(f"Selected pipeline: {pipeline}")

        # 3. Execute Pipeline
        # We hand the plan (sequence of tools) to the dispatcher one by one.
        results = []
        # Initial context includes the original input and analysis
        context = {
            "original_input": text,
            "intent": intent,
            "entities": entities,
            "analysis": analysis_data
        }

        previous_output = text # Default input for first tool

        for i, tool_name in enumerate(pipeline):
            # Determine arguments for the tool
            # Use previous_output. For the first step, it is 'text'.

            step_args = {
                "text": previous_output,
                "entities": entities,
                "context": context
            }

            # Special handling: if intent is 'plan' and tool is 'plan_steps', it might need 'goal'
            if intent == "plan" and tool_name == "plan_steps":
                step_args["goal"] = text

            logger.info(f"Executing tool '{tool_name}' (step {i+1})")
            step_result = self.dispatcher.dispatch(tool_name, arguments=step_args)
            results.append(step_result)

            if step_result["status"] == "success":
                # Update previous_output for the next step in the chain
                # If the tool returns a string, use it. If dict, try to find a relevant field.
                data = step_result["data"]
                if isinstance(data, str):
                    previous_output = data
                elif isinstance(data, dict) and "result" in data:
                    previous_output = data["result"]
                else:
                    previous_output = str(data)
            else:
                logger.error(f"Tool '{tool_name}' failed: {step_result['errors']}")
                # Stop pipeline on failure
                break

        return {
            "intent": intent,
            "pipeline": pipeline,
            "results": results,
            "final_output": previous_output
        }

    def _select_pipeline(self, intent: str) -> List[str]:
        pipeline = self.intent_pipeline_map.get(intent)
        if not pipeline:
            logger.info(f"No pipeline found for intent '{intent}'. Using fallback.")
            pipeline = self.fallback_pipeline
        return pipeline
