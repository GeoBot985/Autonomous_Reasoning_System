
import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.control.core_loop import CoreLoop
from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface
from Autonomous_Reasoning_System.cognition.intent_analyzer import IntentAnalyzer

# Mock call_llm to avoid actual LLM calls
@pytest.fixture(autouse=True)
def mock_llm():
    with patch('Autonomous_Reasoning_System.cognition.intent_analyzer.call_llm') as mock:
        yield mock

@pytest.fixture
def core_loop():
    # Mock components that require DB or heavy initialization
    with patch('Autonomous_Reasoning_System.memory.storage.MemoryStorage'), \
         patch('Autonomous_Reasoning_System.memory.embeddings.EmbeddingModel'), \
         patch('Autonomous_Reasoning_System.memory.vector_store.DuckVSSVectorStore'):
        loop = CoreLoop(verbose=True)
        # Further mock internal components to isolate logic
        loop.memory = MagicMock(spec=MemoryInterface)
        loop.router = MagicMock()
        loop.plan_builder = MagicMock()
        loop.plan_executor = MagicMock()
        loop.reflector = MagicMock()
        loop.confidence = MagicMock()
        return loop

def test_birthday_short_circuit(core_loop, mock_llm):
    # Setup Intent Analyzer mock return via Router (since CoreLoop calls Router)
    # CoreLoop calls router.resolve(text)
    # We need router to return the specific family/subtype

    core_loop.router.resolve.return_value = {
        "intent": "memory_store",
        "family": "personal_facts",
        "subtype": "birthday",
        "entities": {"name": "Nina", "date": "11 January"},
        "pipeline": ["handle_memory_ops"],
        "response_override": None
    }

    text = "Nina's birthday is 11 January"
    result = core_loop.run_once(text)

    # Verify Short Circuit
    assert result["plan_id"] == "birthday_shortcut"
    assert "saved" in result["summary"].lower()

    # Verify Memory Store was called
    core_loop.memory.remember.assert_called()

    # Verify KG insertion attempted (if we implemented it to use entities)
    # In my implementation I tried to extract subject/date from entities
    core_loop.memory.insert_kg_triple.assert_called_with("Nina", "has_birthday", "11 January")

    # Verify NO Planning
    core_loop.plan_builder.new_goal.assert_not_called()
    core_loop.plan_executor.execute_plan.assert_not_called()

    # Verify NO Reflection
    core_loop.reflector.interpret.assert_not_called()

def test_reflection_guard_memory_store(core_loop):
    core_loop.router.resolve.return_value = {
        "intent": "memory_store",
        "family": "memory_operations",
        "subtype": None,
        "pipeline": ["handle_memory_ops"],
        "response_override": None
    }

    # Mock execution result
    core_loop.plan_executor.execute_plan.return_value = {"status": "complete", "summary": "Stored."}

    text = "Remind me to buy milk"
    core_loop.run_once(text)

    # Reflection should be skipped
    core_loop.reflector.interpret.assert_not_called()

def test_reflection_guard_kg_answer(core_loop):
    core_loop.router.resolve.return_value = {
        "intent": "query",
        "family": "question_answering",
        "subtype": None,
        "pipeline": ["answer_question"],
        "response_override": None
    }

    # Mock execution result starting with "Fact:"
    core_loop.plan_executor.execute_plan.return_value = {
        "status": "complete",
        "summary": {"result": "Fact: Nina has_birthday 11 January"} # PlanExecutor returns result inside summary usually?
        # Wait, in CoreLoop: final_output = last_step.result or "Done."
        # So we need to mock what execute_plan returns.
        # CoreLoop:
        # execution_result = self.plan_executor.execute_plan(plan.id)
        # if status == "complete": summary = execution_result.get("summary", {}) ... last_step.result
    }

    # We need to mock the plan structure too because CoreLoop accesses plan.steps[-1].result
    mock_plan = MagicMock()
    mock_step = MagicMock()
    mock_step.result = "Fact: Nina has_birthday 11 January"
    mock_plan.steps = [mock_step]

    core_loop.plan_builder.new_goal.return_value = (MagicMock(), mock_plan)
    core_loop.plan_builder.build_plan.return_value = mock_plan

    core_loop.plan_executor.execute_plan.return_value = {"status": "complete"}

    text = "When is Nina's birthday?"
    core_loop.run_once(text)

    # Reflection should be skipped
    core_loop.reflector.interpret.assert_not_called()
