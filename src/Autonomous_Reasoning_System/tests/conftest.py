import pytest
import sys
from unittest.mock import MagicMock
import numpy as np

# --- Mock circular import ---
# This must happen before any imports that might trigger the circular dependency
sys.modules["Autonomous_Reasoning_System.tools.ocr"] = MagicMock()

# --- Mock FastEmbed ---
# We mock fastembed.TextEmbedding to avoid model downloads and rate limits.
# This needs to be done before MemoryStorage is imported in tests.

class MockTextEmbedding:
    def __init__(self, model_name=None, **kwargs):
        self.model_name = model_name

    def embed(self, documents):
        # Return a generator of dummy embeddings
        # The dimension should match config.VECTOR_DIMENSION (384)
        for _ in documents:
            yield np.random.rand(384).astype(np.float32)

@pytest.fixture(autouse=True)
def mock_fastembed(monkeypatch):
    """
    Globally mock fastembed to prevent model downloads during tests.
    """
    monkeypatch.setattr("fastembed.TextEmbedding", MockTextEmbedding)

@pytest.fixture
def mock_memory_storage(monkeypatch):
    """
    Fixture to provide a MemoryStorage instance with mocked internals if needed specifically.
    """
    # This might not be needed if the global mock works, but good to have as backup
    pass
