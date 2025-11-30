import sys
from unittest.mock import MagicMock
import pytest
import numpy as np

# Mock 'ocr' module globally to prevent circular imports as per memory instructions
sys.modules['Autonomous_Reasoning_System.tools.ocr'] = MagicMock()
# Also mock 'pypdf' if it's not installed in the environment but used in the code
# sys.modules['pypdf'] = MagicMock()

@pytest.fixture
def mock_embedding_model():
    """Returns a mock embedding model that returns fixed vectors."""
    mock = MagicMock()
    # Mock embed to return a generator of numpy arrays (embeddings)
    # We use a fixed size 384 as per config
    def side_effect(texts):
        for text in texts:
            # Return numpy array, which has .tolist()
            yield np.array([0.1] * 384)
    mock.embed.side_effect = side_effect
    return mock
