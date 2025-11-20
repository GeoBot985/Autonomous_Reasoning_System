
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from Autonomous_Reasoning_System.memory.memory_interface import MemoryInterface

class TestMemoryInterface(unittest.TestCase):
    def setUp(self):
        # Mock singletons to avoid real DB/Embedding usage
        self.mock_storage = MagicMock()
        self.mock_episodes = MagicMock()
        self.mock_embedder = MagicMock()
        self.mock_vector_store = MagicMock()

        with patch("Autonomous_Reasoning_System.memory.memory_interface.get_memory_storage", return_value=self.mock_storage), \
             patch("Autonomous_Reasoning_System.memory.memory_interface.EpisodicMemory", return_value=self.mock_episodes), \
             patch("Autonomous_Reasoning_System.memory.memory_interface.get_embedding_model", return_value=self.mock_embedder), \
             patch("Autonomous_Reasoning_System.memory.memory_interface.get_vector_store", return_value=self.mock_vector_store):

            self.mi = MemoryInterface()

    def test_remember(self):
        self.mock_storage.add_memory.return_value = "uuid-123"
        self.mock_episodes.active_episode_id = None

        uid = self.mi.remember("Test memory", {"type": "fact", "importance": 0.8})

        self.mock_storage.add_memory.assert_called_with("Test memory", "fact", 0.8, "unknown")
        self.assertEqual(uid, "uuid-123")

    def test_retrieve_vector(self):
        # Setup mock for vector search success
        self.mock_vector_store.search.return_value = [
            {"text": "Found memory", "score": 0.9, "id": "123"}
        ]

        results = self.mi.retrieve("query")

        self.mock_embedder.embed.assert_called_with("query")
        self.mock_vector_store.search.assert_called()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Found memory")

    def test_retrieve_fallback(self):
        # Setup mock for vector search failure (empty)
        self.mock_vector_store.search.return_value = []
        self.mock_storage.search_text.return_value = [("Fallback memory", 0.5)]

        results = self.mi.retrieve("query")

        self.mock_storage.search_text.assert_called_with("query", top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Fallback memory")

    def test_update(self):
        self.mock_storage.update_memory.return_value = True

        result = self.mi.update("uuid-123", "New content")

        self.mock_storage.update_memory.assert_called_with("uuid-123", "New content")
        self.assertTrue(result)

    def test_summarize_and_compress(self):
        self.mock_episodes.summarize_day.return_value = "Summary of the day"

        summary = self.mi.summarize_and_compress()

        self.mock_episodes.summarize_day.assert_called()
        self.assertEqual(summary, "Summary of the day")

if __name__ == "__main__":
    unittest.main()
