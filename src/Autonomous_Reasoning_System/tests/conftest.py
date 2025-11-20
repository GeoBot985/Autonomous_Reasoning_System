import pytest
import shutil
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

@pytest.fixture(scope="function")
def temp_db_path():
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_memory.parquet")
    yield db_path
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def mock_singletons(temp_db_path):
    # We want to control the initialization of singletons.
    # Since get_memory_storage calls MemoryStorage(), which uses "data" dir by default,
    # we need to patch where MemoryStorage looks, OR we need to reset the singleton
    # and instantiate it with our path.

    # But get_memory_storage doesn't take arguments.
    # So we should probably patch MemoryStorage class in storage.py

    # Reset singletons first
    with patch("Autonomous_Reasoning_System.memory.singletons._memory_storage", new=None), \
         patch("Autonomous_Reasoning_System.memory.singletons._vector_store", new=None), \
         patch("Autonomous_Reasoning_System.memory.singletons._embedding_model", new=None):

         # We also need to patch EmbeddingModel to avoid loading it
         with patch("Autonomous_Reasoning_System.memory.singletons.EmbeddingModel") as MockEmbed, \
              patch("Autonomous_Reasoning_System.memory.storage.MemoryStorage") as RealMemoryStorage: # Wait, if we patch it we replace it.

              # Actually, we want to use the REAL MemoryStorage but with a different path.
              # But the singleton function doesn't pass the path.
              # We can patch the 'Path("data")' inside storage.py or simply patch the class to default to temp path?

              # Easier: Mock the singletons entirely for UNIT tests.
              # For INTEGRATION tests, we might want the real ones but pointed to temp dir.

              # Let's provide a fixture that mocks them for unit tests.
              mock_embed = MagicMock()
              mock_embed.embed.return_value = [0.1] * 384
              MockEmbed.return_value = mock_embed

              yield
