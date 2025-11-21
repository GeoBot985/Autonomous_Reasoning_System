import pytest
import shutil
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import sys

# Patch the missing 'ocr' module globally before any other imports
mock_ocr = MagicMock()
sys.modules["Autonomous_Reasoning_System.tools.ocr"] = mock_ocr

@pytest.fixture(scope="function")
def temp_db_path():
    # Use in-memory database to support HNSW index without experimental persistence flag
    return ":memory:"

@pytest.fixture(scope="function")
def mock_embedding_model():
    mock = MagicMock()
    # Mock embed to return a list of floats of correct dimension (384 for all-MiniLM-L6-v2)
    mock.embed.return_value = [0.1] * 384
    return mock

@pytest.fixture(scope="function")
def mock_vector_store():
    mock = MagicMock()
    mock.metadata = []
    mock.search.return_value = []
    return mock

@pytest.fixture(scope="function")
def memory_storage(temp_db_path, mock_embedding_model, mock_vector_store):
    # Create real MemoryStorage with temp DB
    from Autonomous_Reasoning_System.memory.storage import MemoryStorage
    storage = MemoryStorage(
        db_path=temp_db_path,
        embedding_model=mock_embedding_model,
        vector_store=mock_vector_store
    )
    return storage

@pytest.fixture(scope="function")
def memory_interface(memory_storage, mock_embedding_model, mock_vector_store):
    # We need to patch get_persistence_service because MemoryInterface uses it
    with patch("Autonomous_Reasoning_System.memory.memory_interface.get_persistence_service") as mock_get_persist:
        mock_persist = MagicMock()
        # Mock loading methods to return empty data or minimal valid data
        mock_persist.load_deterministic_memory.return_value = pd.DataFrame(columns=["id", "text", "memory_type", "created_at", "last_accessed", "importance", "scheduled_for", "status", "source"])
        mock_persist.load_goals.return_value = pd.DataFrame(columns=["id", "text", "priority", "status", "steps", "metadata", "created_at", "updated_at"])
        mock_persist.load_episodic_memory.return_value = pd.DataFrame()
        mock_persist.load_vector_index.return_value = None
        mock_persist.load_vector_metadata.return_value = []

        mock_get_persist.return_value = mock_persist

        from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface

        # Instantiate with injected dependencies
        interface = MemoryInterface(
            memory_storage=memory_storage,
            embedding_model=mock_embedding_model,
            vector_store=mock_vector_store
        )
        yield interface

@pytest.fixture(scope="function")
def mock_plan_builder(memory_storage):
    from Autonomous_Reasoning_System.planning.plan_builder import PlanBuilder
    return PlanBuilder(memory_storage=memory_storage)
