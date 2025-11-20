
import pytest
from src.Autonomous_Reasoning_System.control.dispatcher import Dispatcher

def mock_tool_add(x, y):
    return x + y

def mock_tool_greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

def mock_tool_fail():
    raise ValueError("Something went wrong")

class TestDispatcher:
    def test_register_and_dispatch_success(self):
        dispatcher = Dispatcher()
        dispatcher.register_tool("add", mock_tool_add, schema={"x": {"type": int, "required": True}, "y": {"type": int, "required": True}})

        result = dispatcher.dispatch("add", {"x": 5, "y": 3})

        assert result["status"] == "success"
        assert result["data"] == 8
        assert result["errors"] == []
        assert result["meta"]["tool_name"] == "add"

    def test_missing_argument_validation(self):
        dispatcher = Dispatcher()
        dispatcher.register_tool("add", mock_tool_add, schema={"x": {"type": int, "required": True}, "y": {"type": int, "required": True}})

        result = dispatcher.dispatch("add", {"x": 5})

        assert result["status"] == "error"
        assert "Missing required argument: y" in result["errors"]

    def test_wrong_type_validation(self):
        dispatcher = Dispatcher()
        dispatcher.register_tool("add", mock_tool_add, schema={"x": {"type": int, "required": True}, "y": {"type": int, "required": True}})

        result = dispatcher.dispatch("add", {"x": 5, "y": "3"})

        assert result["status"] == "error"
        assert any("expected type" in e for e in result["errors"])

    def test_unknown_tool(self):
        dispatcher = Dispatcher()
        result = dispatcher.dispatch("unknown_tool", {})

        assert result["status"] == "error"
        assert "Tool 'unknown_tool' not found" in result["errors"]

    def test_tool_execution_exception(self):
        dispatcher = Dispatcher()
        dispatcher.register_tool("fail", mock_tool_fail)

        result = dispatcher.dispatch("fail", {})

        assert result["status"] == "error"
        assert "Something went wrong" in result["errors"][0]

    def test_dry_run(self):
        dispatcher = Dispatcher()
        dispatcher.register_tool("add", mock_tool_add, schema={"x": {"type": int, "required": True}})

        result = dispatcher.dispatch("add", {"x": 10, "y": 20}, dry_run=True)

        assert result["status"] == "success"
        assert "Dry run successful" in str(result["data"])
        # Ensure it didn't actually run the tool?
        # The mock is pure, so no side effects to check, but result data confirms dry run path taken.

    def test_metadata_propagation(self):
        dispatcher = Dispatcher()
        dispatcher.register_tool("greet", mock_tool_greet)

        context = {"user_id": "123", "session_id": "abc"}
        result = dispatcher.dispatch("greet", {"name": "Alice"}, context=context)

        assert result["status"] == "success"
        assert result["meta"]["context"] == context
        assert "duration" in result["meta"]
        assert result["meta"]["timestamp"] > 0

    def test_lineage_tracking(self):
        dispatcher = Dispatcher()
        dispatcher.register_tool("add", mock_tool_add)

        dispatcher.dispatch("add", {"x": 1, "y": 2})
        dispatcher.dispatch("add", {"x": 3, "y": 4})

        history = dispatcher.get_history()
        assert len(history) == 2
        assert history[0]["tool_name"] == "add"
        assert history[0]["input_summary"] == str({"x": 1, "y": 2})
        assert history[0]["output_summary"] == "3"

        assert history[1]["tool_name"] == "add"
        assert history[1]["output_summary"] == "7"
