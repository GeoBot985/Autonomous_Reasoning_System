import pytest
import shutil
import tempfile
import os
from unittest.mock import patch, MagicMock
from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface
from Autonomous_Reasoning_System.memory.storage import MemoryStorage
# from Autonomous_Reasoning_System.memory.singletons import get_memory_storage # Removed

@pytest.fixture
def temp_storage():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_memory.duckdb") # Changed to duckdb to match default

    # Create storage instance directly
    storage = MemoryStorage(db_path=db_path)

    yield storage

    try:
        shutil.rmtree(temp_dir)
    except:
        pass

def test_memory_interface_integration(temp_storage):
    # Mock EmbeddingModel and VectorStore to avoid heavy loads for this integration test

    mock_embed = MagicMock()
    mock_embed.embed.return_value = [0.1] * 384

    mock_vector = MagicMock()
    mock_vector.search.return_value = [{"text": "Analyzed the memory system", "score": 0.9, "id": "123"}]
    mock_vector.metadata = []

    # We need to patch persistence loading in MemoryInterface __init__
    with patch("Autonomous_Reasoning_System.memory.memory_interface.get_persistence_service") as mock_get_persist:
         mock_persist_svc = MagicMock()
         # Mock loads
         mock_persist_svc.load_vector_index.return_value = None
         mock_persist_svc.load_vector_metadata.return_value = []
         mock_persist_svc.load_episodic_memory.return_value = None # or empty DF
         mock_get_persist.return_value = mock_persist_svc

         mem = MemoryInterface(
             memory_storage=temp_storage,
             embedding_model=mock_embed,
             vector_store=mock_vector
         )

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
         # Retrieve returns list of dicts or summary string depending on helper
         # recall() returns summary string
         assert "Analyzed the memory system" in recall_result

         # End the episode (mocks LLM summary)
         with patch("Autonomous_Reasoning_System.memory.llm_summarizer.summarize_with_local_llm", return_value="Mock Summary"):
             summary = mem.end_episode("summarize key events")
             assert summary == "Mock Summary"
