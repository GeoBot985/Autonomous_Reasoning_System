import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.control.router import Router, IntentFamily
from Autonomous_Reasoning_System.control.dispatcher import Dispatcher

class TestControlRouter:
    @pytest.fixture
    def dispatcher(self):
        return MagicMock(spec=Dispatcher)

    @pytest.fixture
    def router(self, dispatcher):
        return Router(dispatcher)

    def test_router_initialization(self, router):
        assert router.intent_family_map["remember"] == IntentFamily.MEMORY
        assert router.intent_family_map["query"] == IntentFamily.QA
        assert "handle_memory_ops" in router.family_pipeline_map[IntentFamily.MEMORY]

    def test_resolve_intent(self, router, dispatcher):
        # Mock Intent Analysis
        dispatcher.dispatch.return_value = {
            "status": "success",
            "data": {"intent": "remember", "entities": {"item": "keys"}}
        }

        result = router.resolve("Remember my keys")

        assert result["intent"] == "remember"
        assert result["family"] == IntentFamily.MEMORY
        assert result["pipeline"] == ["handle_memory_ops"]
        dispatcher.dispatch.assert_called_with("analyze_intent", arguments={"text": "Remember my keys"})

    def test_resolve_fallback(self, router, dispatcher):
        dispatcher.dispatch.return_value = {
            "status": "success",
            "data": {"intent": "unknown_intent"}
        }

        result = router.resolve("Something weird")

        assert result["intent"] == "unknown_intent"
        assert result["family"] == IntentFamily.QA # Default
        assert result["pipeline"] == ["answer_question"] # Default pipeline

    def test_execute_pipeline(self, router, dispatcher):
        pipeline = ["handle_memory_ops"]
        dispatcher.dispatch.return_value = {"status": "success", "data": "Memory Stored"}

        result = router.execute_pipeline(pipeline, "Remember this")

        assert len(result["results"]) == 1
        assert result["final_output"] == "Memory Stored"
        dispatcher.dispatch.assert_called_with(
            "handle_memory_ops",
            arguments={"text": "Remember this", "entities": {}, "context": {'original_input': 'Remember this'}}
        )

    def test_route_end_to_end(self, router, dispatcher):
        # Analyze intent mock
        def dispatch_side_effect(tool_name, arguments=None, **kwargs):
            if tool_name == "analyze_intent":
                return {"status": "success", "data": {"intent": "query", "entities": {}}}
            elif tool_name == "answer_question":
                return {"status": "success", "data": "Paris"}
            return {"status": "error", "errors": ["Unknown tool"]}

        dispatcher.dispatch.side_effect = dispatch_side_effect

        result = router.route("Capital of France?")

        assert result["intent"] == "query"
        assert result["final_output"] == "Paris"

    def test_pipeline_override(self, router, dispatcher):
        # Test explicit pipeline override
        def dispatch_side_effect(tool_name, arguments=None, **kwargs):
            if tool_name == "analyze_intent":
                return {"status": "success", "data": {"intent": "remember", "entities": {}}}
            elif tool_name == "handle_memory_ops":
                 return {"status": "success", "data": "Result 1"}
            return {"status": "error", "errors": [f"Unknown tool {tool_name}"]}

        dispatcher.dispatch.side_effect = dispatch_side_effect

        # Use valid tools for override
        override = ["handle_memory_ops"]
        result = router.route("Input", pipeline_override=override)

        assert result["pipeline"] == override
        assert result["intent"] == "override"
        assert result["final_output"] == "Result 1"

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

        # Register valid modules for test
        router._valid_modules.add("step_1")
        router._valid_modules.add("step_2")

        # Use family mapping to inject pipeline
        router.intent_family_map["test_chain"] = "test_family"
        router.family_pipeline_map["test_family"] = ["step_1", "step_2"]

        result = router.route("Original Input")

        assert result["final_output"] == "Received: Output from Step 1"
        assert len(result["results"]) == 2
