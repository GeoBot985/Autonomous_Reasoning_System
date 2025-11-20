import pytest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface

class TestMemoryInterface:
    @pytest.fixture
    def interface(self):
        with patch("Autonomous_Reasoning_System.memory.memory_interface.get_memory_storage") as mock_storage, \
             patch("Autonomous_Reasoning_System.memory.memory_interface.EpisodicMemory") as mock_episodes, \
             patch("Autonomous_Reasoning_System.memory.memory_interface.get_embedding_model") as mock_embedder, \
             patch("Autonomous_Reasoning_System.memory.memory_interface.get_vector_store") as mock_vector_store:

            # Setup mocks
            mock_storage_instance = MagicMock()
            mock_storage.return_value = mock_storage_instance

            mock_episodes_instance = MagicMock()
            mock_episodes.return_value = mock_episodes_instance

            mock_embedder_instance = MagicMock()
            mock_embedder.return_value = mock_embedder_instance

            mock_vector_store_instance = MagicMock()
            mock_vector_store.return_value = mock_vector_store_instance

            interface = MemoryInterface()
            return interface

    def test_init(self, interface):
        assert interface.storage is not None
        assert interface.episodes is not None
        assert interface.embedder is not None
        assert interface.vector_store is not None

    def test_store(self, interface):
        interface.storage.add_memory.return_value = "test_uid"
        interface.episodes.active_episode_id = "ep1"

        uid = interface.store("test text", "note", 0.8)

        assert uid == "test_uid"
        interface.storage.add_memory.assert_called_with("test text", "note", 0.8)

    def test_recall_with_results(self, interface):
        interface.embedder.embed.return_value = [0.1, 0.2]
        interface.storage.vector_store.search.return_value = [
            {"score": 0.9, "text": "memory 1"},
            {"score": 0.8, "text": "memory 2"}
        ]

        summary = interface.recall("query", k=2)

        assert "memory 1" in summary
        assert "memory 2" in summary
        interface.embedder.embed.assert_called_with("query")
        interface.storage.vector_store.search.assert_called_with([0.1, 0.2], 2)

    def test_recall_no_results(self, interface):
        interface.storage.vector_store.search.return_value = []
        summary = interface.recall("query")
        assert summary == "No relevant memories found."

    def test_start_episode(self, interface):
        interface.episodes.begin_episode.return_value = "ep_123"

        eid = interface.start_episode("starting test")

        assert eid == "ep_123"
        interface.episodes.begin_episode.assert_called_once()
        # verify store was called for the description
        interface.storage.add_memory.assert_called_with("Episode started: starting test", "context", 0.4)

    def test_end_episode_no_active(self, interface):
        interface.episodes.active_episode_id = None
        summary = interface.end_episode("hint")
        assert summary is None
        interface.episodes.end_episode.assert_not_called()

    @patch("Autonomous_Reasoning_System.memory.llm_summarizer.summarize_with_local_llm")
    def test_end_episode_with_active(self, mock_summarizer, interface):
        interface.episodes.active_episode_id = "ep_123"
        # Mock dataframe for recent memories
        import pandas as pd
        mock_df = pd.DataFrame({"text": ["mem1", "mem2"]})
        interface.storage.get_all_memories.return_value = mock_df
        mock_summarizer.return_value = "Summary text"

        summary = interface.end_episode("hint")

        assert summary == "Summary text"
        mock_summarizer.assert_called_once()
        interface.episodes.end_episode.assert_called_with("Summary text")

    def test_search_similar_vector(self, interface):
        interface.embedder.embed.return_value = [0.1]
        interface.vector_store.search.return_value = [{"text": "res", "score": 0.9}]

        results = interface.search_similar("query")

        assert len(results) == 1
        assert results[0]["text"] == "res"
        interface.vector_store.search.assert_called_with([0.1], top_k=3)

    def test_summarize_day(self, interface):
        interface.episodes.summarize_day.return_value = "Day summary"
        res = interface.summarize_day()
        assert res == "Day summary"
        interface.episodes.summarize_day.assert_called_once()
