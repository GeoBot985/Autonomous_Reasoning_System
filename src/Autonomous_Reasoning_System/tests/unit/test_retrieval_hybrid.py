
import pytest
import unittest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.memory.retrieval_orchestrator import RetrievalOrchestrator
from Autonomous_Reasoning_System.memory.storage import MemoryStorage

class TestRetrievalOrchestrator:

    @pytest.fixture
    def mock_storage(self):
        storage = MagicMock(spec=MemoryStorage)
        storage.search_text.return_value = []
        # Mock vector store inside storage
        storage.vector_store = MagicMock()
        storage.vector_store.search.return_value = []
        return storage

    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock()
        embedder.embed.return_value = [0.1, 0.2, 0.3]
        return embedder

    @pytest.fixture
    def mock_extractor(self):
        with patch('Autonomous_Reasoning_System.memory.retrieval_orchestrator.EntityExtractor') as MockExtractor:
            instance = MockExtractor.return_value
            instance.extract.return_value = ["Cornelia", "birthday"]
            yield instance

    def test_deterministic_priority(self, mock_storage, mock_embedder, mock_extractor):
        """Test that deterministic results are prioritized when confidence is high."""
        orchestrator = RetrievalOrchestrator(memory_storage=mock_storage, embedding_model=mock_embedder)

        # Setup deterministic match
        mock_storage.search_text.return_value = [("Cornelia's birthday is November 21st.", 1.0)]

        # Setup semantic match (irrelevant or lower priority)
        mock_storage.vector_store.search.return_value = [{"text": "Cornelia likes cake."}]

        results = orchestrator.retrieve("When is Cornelia's birthday?")

        # Should return ONLY the deterministic match because score 1.0 >= 0.9
        assert len(results) == 1
        assert results[0] == "Cornelia's birthday is November 21st."

        # Verify extraction was called
        mock_extractor.extract.assert_called_once()
        # Verify storage search was called with extracted keywords
        mock_storage.search_text.assert_called_with(["Cornelia", "birthday"], top_k=3)

    def test_hybrid_fallback(self, mock_storage, mock_embedder, mock_extractor):
        """Test fallback when deterministic search fails."""
        orchestrator = RetrievalOrchestrator(memory_storage=mock_storage, embedding_model=mock_embedder)

        # No deterministic match
        mock_storage.search_text.return_value = []

        # Semantic match found
        mock_storage.vector_store.search.return_value = [{"text": "Cornelia likes cake."}]

        results = orchestrator.retrieve("When is Cornelia's birthday?")

        # Should return semantic results
        assert len(results) == 1
        assert results[0] == "Cornelia likes cake."

    def test_combined_results(self, mock_storage, mock_embedder, mock_extractor):
        """Test mixing when deterministic score is low (though currently search_text returns 1.0,
           if we had a scenario where it returned lower, or we want to test simply that it falls back if empty).

           Actually, logic says: if det_results[0][1] >= 0.9 return ONLY det.
           So let's simulate a case where we might want mixed results?
           The current logic is strict: if ANY high confidence deterministic, return ONLY that.

           If search_text returns nothing, we get semantic.
           If search_text returns something with score < 0.9 (unlikely with current hardcoded 1.0, but for completeness):
        """
        orchestrator = RetrievalOrchestrator(memory_storage=mock_storage, embedding_model=mock_embedder)

        # Low confidence deterministic match
        mock_storage.search_text.return_value = [("Maybe Cornelia?", 0.5)]
        mock_storage.vector_store.search.return_value = [{"text": "Cornelia likes cake."}]

        results = orchestrator.retrieve("Who is Cornelia?")

        # Should combine both
        assert "Maybe Cornelia?" in results
        assert "Cornelia likes cake." in results
        assert len(results) == 2
