import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.llm.context_adapter import ContextAdapter

@patch("Autonomous_Reasoning_System.llm.context_adapter.call_llm")
@patch("Autonomous_Reasoning_System.llm.context_adapter.RetrievalOrchestrator")
@patch("Autonomous_Reasoning_System.llm.context_adapter.ReasoningConsolidator")
@patch("Autonomous_Reasoning_System.llm.context_adapter.ContextBuilder")
def test_context_adapter_run(
    mock_ContextBuilder,
    mock_ReasoningConsolidator,
    mock_RetrievalOrchestrator,
    mock_call_llm
):
    # Setup mocks
    mock_retriever = MagicMock()
    mock_RetrievalOrchestrator.return_value = mock_retriever
    mock_retriever.retrieve.return_value = ["Fact 1", "Fact 2"]

    mock_memory = MagicMock()

    mock_call_llm.return_value = "This is a mock response from Ollama."

    adapter = ContextAdapter(memory_storage=mock_memory)
    response = adapter.run("Hello world")

    assert response == "This is a mock response from Ollama."

    # Verify memory storage
    mock_memory.add_memory.assert_called_once()
    args, kwargs = mock_memory.add_memory.call_args
    assert "User: Hello world" in kwargs['text']
    assert "Tyrone: This is a mock response from Ollama." in kwargs['text']
