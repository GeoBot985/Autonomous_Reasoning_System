import pytest
import shutil
import tempfile
import os
from unittest.mock import patch, MagicMock
from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface
from Autonomous_Reasoning_System.memory.storage import MemoryStorage
from Autonomous_Reasoning_System.memory.singletons import get_memory_storage

@pytest.fixture
def temp_storage():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_memory.parquet")

    # We need to ensure singletons use this path.
    # Since we can't easily pass args to the singleton getter, we might need to manually instantiate MemoryStorage
    # and set it in singletons module, or patch the singleton getter.

    storage = MemoryStorage(db_path=db_path)

    # Patch the singleton
    with patch("Autonomous_Reasoning_System.memory.singletons._memory_storage", storage):
        yield storage

    shutil.rmtree(temp_dir)

def test_memory_interface_integration(temp_storage):
    # Mock EmbeddingModel and VectorStore to avoid heavy loads for this integration test
    # unless we really want to test FAISS integration too.
    # Let's mock the heavy parts but keep the DB logic.

    with patch("Autonomous_Reasoning_System.memory.singletons.get_embedding_model") as mock_get_embed, \
         patch("Autonomous_Reasoning_System.memory.singletons.get_vector_store") as mock_get_vector:

        mock_embed = MagicMock()
        mock_embed.embed.return_value = [0.1] * 384
        mock_get_embed.return_value = mock_embed

        mock_vector = MagicMock()
        mock_get_vector.return_value = mock_vector
        mock_vector.search.return_value = [{"text": "Analyzed the memory system", "score": 0.9}]

        mem = MemoryInterface()

        # Start new episode
        eid = mem.start_episode("Morning reasoning session")
        assert eid is not None

        # Store some memories
        uid1 = mem.store("Analyzed the memory system design for Tyrone.")
        uid2 = mem.store("Implemented vector and episodic layers successfully.")

        assert uid1 is not None
        assert uid2 is not None

        # Verify they are in DuckDB
        df = temp_storage.get_all_memories()
        assert len(df) >= 2
        assert "Analyzed the memory system design for Tyrone." in df["text"].values

        # Query recall (mocks vector store search but logic flows through)
        recall_result = mem.recall("memory integration")
        assert "Analyzed the memory system" in recall_result

        # End the episode (mocks LLM summary)
        with patch("Autonomous_Reasoning_System.memory.llm_summarizer.summarize_with_local_llm", return_value="Mock Summary"):
            summary = mem.end_episode("summarize key events")
            assert summary == "Mock Summary"
