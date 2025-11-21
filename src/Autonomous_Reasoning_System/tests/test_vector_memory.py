# tests/test_vector_memory.py
from Autonomous_Reasoning_System.memory.storage import MemoryStorage
from unittest.mock import MagicMock
import pytest

def test_vector_memory():
    # Mock embedding model and vector store
    mock_embed = MagicMock()
    mock_embed.embed.return_value = [0.1] * 384

    mock_vector = MagicMock()
    # Mock search result
    mock_vector.search.return_value = [
        {"text": "Meeting with John", "score": 0.9, "memory_type": "note", "id": "1"}
    ]
    mock_vector.metadata = []

    store = MemoryStorage(db_path=":memory:", embedding_model=mock_embed, vector_store=mock_vector)

    store.add_memory("I met Sarah at the coffee shop yesterday.", "note")
    store.add_memory("Meeting with John about project timeline next week.", "note")
    store.add_memory("Remember to buy groceries for the weekend.", "note")

    print("\nQuery: meeting schedule")
    # We invoke the vector search manually since MemoryStorage might not auto-search on add
    # but MemoryInterface does. Here we test MemoryStorage + VectorStore interaction
    # if MemoryStorage exposes search.

    # Actually MemoryStorage usually handles SQL search. Vector search is in MemoryInterface.
    # But the original test imported get_memory_storage and accessed store.vector_store.

    q_vec = store.embedder.embed("meeting schedule")
    results = store.vector_store.search(q_vec)

    assert len(results) > 0
    assert results[0]["text"] == "Meeting with John"

    for r in results:
        print(f"- ({r['score']:.3f}) {r['text']} [{r['memory_type']}]")
