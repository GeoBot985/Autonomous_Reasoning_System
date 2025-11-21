
import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.tools.standard_tools import register_tools
import pandas as pd

class MockDispatcher:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, handler, schema):
        self.tools[name] = handler

@pytest.fixture
def mock_components():
    components = {
        "intent_analyzer": MagicMock(),
        "memory": MagicMock(),
        "reflector": MagicMock(),
        "plan_builder": MagicMock(),
        "deterministic_responder": MagicMock(),
        "context_adapter": MagicMock(),
        "goal_manager": MagicMock(),
    }
    components["goal_manager"].memory = components["memory"]
    return components

@pytest.fixture
def dispatcher():
    return MockDispatcher()

def test_register_tools(dispatcher, mock_components):
    register_tools(dispatcher, mock_components)

    assert "analyze_intent" in dispatcher.tools
    assert "store_memory" in dispatcher.tools
    assert "search_memory" in dispatcher.tools
    assert "perform_reflection" in dispatcher.tools
    assert "summarize_context" in dispatcher.tools
    assert "deterministic_responder" in dispatcher.tools
    assert "plan_steps" in dispatcher.tools
    assert "answer_question" in dispatcher.tools
    assert "create_goal" in dispatcher.tools
    assert "list_goals" in dispatcher.tools
    assert "handle_memory_ops" in dispatcher.tools
    assert "handle_goal_ops" in dispatcher.tools
    assert "perform_self_analysis" in dispatcher.tools

def test_analyze_intent(dispatcher, mock_components):
    register_tools(dispatcher, mock_components)
    mock_components["intent_analyzer"].analyze.return_value = {"intent": "test"}

    result = dispatcher.tools["analyze_intent"](text="test input")
    assert result == {"intent": "test"}

def test_store_memory(dispatcher, mock_components):
    register_tools(dispatcher, mock_components)

    result = dispatcher.tools["store_memory"](text="test fact")
    assert "Stored: test fact" in result
    mock_components["memory"].remember.assert_called_once()

def test_search_memory(dispatcher, mock_components):
    register_tools(dispatcher, mock_components)
    mock_components["memory"].retrieve.return_value = [{"text": "result 1"}]

    result = dispatcher.tools["search_memory"](text="query")
    assert "- result 1" in result

def test_perform_reflection(dispatcher, mock_components):
    register_tools(dispatcher, mock_components)
    mock_components["reflector"].interpret.return_value = "reflection result"

    result = dispatcher.tools["perform_reflection"](text="reflect on this")
    assert result == "reflection result"

def test_deterministic_responder(dispatcher, mock_components):
    register_tools(dispatcher, mock_components)
    mock_components["deterministic_responder"].run.return_value = "4"

    result = dispatcher.tools["deterministic_responder"](text="2+2")
    assert result == "4"

def test_plan_steps(dispatcher, mock_components):
    register_tools(dispatcher, mock_components)
    mock_components["plan_builder"].decompose_goal.return_value = ["step 1", "step 2"]

    result = dispatcher.tools["plan_steps"](text="goal")
    assert "1. step 1" in result
    assert "2. step 2" in result

def test_answer_question(dispatcher, mock_components):
    register_tools(dispatcher, mock_components)
    mock_components["context_adapter"].run.return_value = "Answer"

    result = dispatcher.tools["answer_question"](text="Question?")
    assert result == "Answer"

def test_create_goal(dispatcher, mock_components):
    register_tools(dispatcher, mock_components)
    mock_components["goal_manager"].create_goal.return_value = "goal_id"

    result = dispatcher.tools["create_goal"](text="new goal")
    assert result == "goal_id"

def test_list_goals(dispatcher, mock_components):
    register_tools(dispatcher, mock_components)

    df = pd.DataFrame([
        {"id": "123456789", "text": "goal 1", "status": "pending"},
        {"id": "987654321", "text": "goal 2", "status": "completed"}
    ])
    mock_components["memory"].get_active_goals.return_value = df

    result = dispatcher.tools["list_goals"]()
    assert "[12345678] goal 1 (Status: pending)" in result
    assert "[98765432] goal 2 (Status: completed)" in result

def test_handle_memory_ops_store(dispatcher, mock_components):
    register_tools(dispatcher, mock_components)

    result = dispatcher.tools["handle_memory_ops"](text="some fact", intent="store")
    assert "Stored: some fact" in result
    mock_components["memory"].remember.assert_called()

def test_handle_memory_ops_search(dispatcher, mock_components):
    register_tools(dispatcher, mock_components)
    mock_components["memory"].retrieve.return_value = [{"text": "found"}]

    result = dispatcher.tools["handle_memory_ops"](text="query", intent="search")
    assert "- found" in result

def test_handle_goal_ops_list(dispatcher, mock_components):
    register_tools(dispatcher, mock_components)
    df = pd.DataFrame([{"id": "1", "text": "goal", "status": "pending"}])
    mock_components["memory"].get_active_goals.return_value = df

    result = dispatcher.tools["handle_goal_ops"](text="list goals", context={"intent": "list_goals"})
    assert "[1] goal" in result

def test_handle_goal_ops_create(dispatcher, mock_components):
    register_tools(dispatcher, mock_components)
    mock_components["goal_manager"].create_goal.return_value = "new_goal_id"

    result = dispatcher.tools["handle_goal_ops"](text="new task", context={"intent": "create_goal"})
    assert result == "new_goal_id"

def test_perform_self_analysis(dispatcher, mock_components):
    register_tools(dispatcher, mock_components)
    mock_components["reflector"].interpret.return_value = "analysis"

    result = dispatcher.tools["perform_self_analysis"](text="status")
    assert result == "analysis"
