import pytest
from unittest.mock import MagicMock, ANY
from Autonomous_Reasoning_System.control.router import Router

class TestControlRouter:

    @pytest.fixture
    def dispatcher(self):
        return MagicMock()

    @pytest.fixture
    def router(self, dispatcher):
        return Router(dispatcher)

    def test_route_unknown_intent_fallback(self, router, dispatcher):
        # Mock Dispatcher.dispatch behavior
        def dispatch_side_effect(tool_name, arguments=None, **kwargs):
            if tool_name == "analyze_intent":
                return {"status": "success", "data": {"intent": "unknown", "entities": {}}}
            elif tool_name == "answer_question":
                return {"status": "success", "data": "This is a generic answer."}
            return {"status": "error", "errors": ["Unknown tool"]}

        dispatcher.dispatch.side_effect = dispatch_side_effect

        # Execute
        result = router.route("What is the meaning of life?")

        # Verify
        assert result["intent"] == "unknown"
        assert result["pipeline"] == ["answer_question"]
        assert len(result["results"]) == 1
        assert result["results"][0]["status"] == "success"
        assert result["final_output"] == "This is a generic answer."

        # Check tool calls
        assert dispatcher.dispatch.call_count == 2
        dispatcher.dispatch.assert_any_call("analyze_intent", arguments={"text": "What is the meaning of life?"})
        # Use ANY for context argument validation as it's complex
        dispatcher.dispatch.assert_any_call("answer_question", arguments=ANY)

    def test_route_plan_intent(self, router, dispatcher):
        def dispatch_side_effect(tool_name, arguments=None, **kwargs):
            if tool_name == "analyze_intent":
                return {"status": "success", "data": {"intent": "plan", "entities": {}}}
            elif tool_name == "plan_steps":
                return {"status": "success", "data": "Plan created successfully."}
            return {"status": "error", "errors": ["Unknown tool"]}

        dispatcher.dispatch.side_effect = dispatch_side_effect

        # Execute
        result = router.route("Plan a surprise party")

        # Verify
        assert result["intent"] == "plan"
        assert result["pipeline"] == ["plan_steps"]

        # Verify arguments
        # Last call should be plan_steps
        args, kwargs = dispatcher.dispatch.call_args
        assert args[0] == "plan_steps"
        assert kwargs['arguments']['goal'] == "Plan a surprise party"

    def test_route_remember_intent(self, router, dispatcher):
        def dispatch_side_effect(tool_name, arguments=None, **kwargs):
            if tool_name == "analyze_intent":
                return {"status": "success", "data": {"intent": "remember", "entities": {}}}
            elif tool_name == "store_memory":
                return {"status": "success", "data": "Memory stored."}
            return {"status": "error", "errors": ["Unknown tool"]}

        dispatcher.dispatch.side_effect = dispatch_side_effect

        # Execute
        result = router.route("Remember to buy milk")

        # Verify
        assert result["intent"] == "remember"
        assert result["pipeline"] == ["store_memory"]

    def test_pipeline_execution_failure(self, router, dispatcher):
        def dispatch_side_effect(tool_name, arguments=None, **kwargs):
            if tool_name == "analyze_intent":
                return {"status": "success", "data": {"intent": "search", "entities": {}}}
            elif tool_name == "search_memory":
                return {"status": "error", "errors": ["Search failed"]}
            return {"status": "error", "errors": ["Unknown tool"]}

        dispatcher.dispatch.side_effect = dispatch_side_effect

        # Execute
        result = router.route("Search for keys")

        # Verify
        assert result["intent"] == "search"
        assert len(result["results"]) == 1
        assert result["results"][0]["status"] == "error"
        assert "Search failed" in result["results"][0]["errors"][0]

    def test_analyze_intent_failure(self, router, dispatcher):
        # Analyze intent returns error
        def dispatch_side_effect(tool_name, arguments=None, **kwargs):
            if tool_name == "analyze_intent":
                return {"status": "error", "errors": ["Analyzer offline"]}
            elif tool_name == "answer_question":
                return {"status": "success", "data": "Fallback answer"}
            return {"status": "error", "errors": ["Unknown tool"]}

        dispatcher.dispatch.side_effect = dispatch_side_effect

        # Execute
        result = router.route("Hello")

        # Verify logic: should proceed with unknown intent (due to failure) -> fallback
        assert result["intent"] == "unknown"
        assert result["pipeline"] == ["answer_question"]

    def test_pipeline_override(self, router, dispatcher):
        # Test explicit pipeline override
        def dispatch_side_effect(tool_name, arguments=None, **kwargs):
            if tool_name == "analyze_intent":
                return {"status": "success", "data": {"intent": "remember", "entities": {}}}
            elif tool_name == "custom_tool_1":
                return {"status": "success", "data": "Result 1"}
            elif tool_name == "custom_tool_2":
                return {"status": "success", "data": "Result 2"}
            return {"status": "error", "errors": [f"Unknown tool {tool_name}"]}

        dispatcher.dispatch.side_effect = dispatch_side_effect

        override = ["custom_tool_1", "custom_tool_2"]
        result = router.route("Input", pipeline_override=override)

        assert result["pipeline"] == override
        assert len(result["results"]) == 2
        assert result["results"][0]["data"] == "Result 1"
        assert result["results"][1]["data"] == "Result 2"
        assert result["final_output"] == "Result 2"

        # Verify intent was still analyzed (unless we want to skip it? Implementation calls it first)
        dispatcher.dispatch.assert_any_call("analyze_intent", arguments={"text": "Input"})

    def test_pipeline_chaining(self, router, dispatcher):
        # Test that output of step 1 is passed to step 2
        def dispatch_side_effect(tool_name, arguments=None, **kwargs):
            if tool_name == "analyze_intent":
                return {"status": "success", "data": {"intent": "test_chain", "entities": {}}}
            elif tool_name == "step_1":
                return {"status": "success", "data": "Output from Step 1"}
            elif tool_name == "step_2":
                # Check if input was correct
                input_text = arguments.get("text")
                return {"status": "success", "data": f"Received: {input_text}"}
            return {"status": "error", "errors": [f"Unknown tool {tool_name}"]}

        dispatcher.dispatch.side_effect = dispatch_side_effect

        # Manually inject pipeline for testing chaining
        router.intent_pipeline_map["test_chain"] = ["step_1", "step_2"]

        result = router.route("Original Input")

        assert result["intent"] == "test_chain"
        assert result["results"][0]["data"] == "Output from Step 1"
        assert result["results"][1]["data"] == "Received: Output from Step 1"
        assert result["final_output"] == "Received: Output from Step 1"
