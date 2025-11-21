from Autonomous_Reasoning_System.llm.reflection_interpreter import ReflectionInterpreter
from unittest.mock import MagicMock, patch

def test_reflection_interpreter():
    mock_memory = MagicMock()
    # Mock get_all_memories
    import pandas as pd
    mock_memory.get_all_memories.return_value = pd.DataFrame([
        {"text": "I did good today.", "memory_type": "reflection", "created_at": "2023-01-01"}
    ])

    # Inject memory
    ri = ReflectionInterpreter(memory_storage=mock_memory)

    # Mock retrieval orchestrator inside RI (which we created in init)
    ri.retriever = MagicMock()
    ri.retriever.retrieve.return_value = ["Fact 1"]
    ri.retriever._semantic_retrieve.return_value = ["Fact 1"]

    query = "What patterns do you see in my recent work?"

    # We mock call_llm to avoid real network calls
    with patch("Autonomous_Reasoning_System.llm.reflection_interpreter.call_llm") as mock_call:
        mock_call.return_value = '{"summary": "You work hard", "insight": "Keep it up", "confidence_change": "positive"}'

        result = ri.interpret(query)

        print(f"🧩 Query: {query}")
        print(f"🪞 Summary: {result['summary']}")
        print(f"💡 Insight: {result['insight']}")
        print(f"📈 Confidence Change: {result['confidence_change']}\n")

        assert result["summary"] == "You work hard"
        assert result["confidence_change"] == "positive"

if __name__ == "__main__":
    test_reflection_interpreter()
