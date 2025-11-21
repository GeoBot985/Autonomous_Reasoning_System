# Autonomous_Reasoning_System/memory/vector_memory.py
from Autonomous_Reasoning_System.memory.storage import MemoryStorage
from Autonomous_Reasoning_System.memory.embeddings import EmbeddingModel
from Autonomous_Reasoning_System.memory.vector_store import VectorStore
import numpy as np


class VectorMemory:
    """
    Combines symbolic (DuckDB) memory with vector-based semantic recall.
    """

    def __init__(self, memory_storage=None, embedding_model=None, vector_store=None):
        print("🧠 Vector Memory initialized.")
        self.storage = memory_storage or MemoryStorage()
        self.embedder = embedding_model or EmbeddingModel()
        self.vectors = vector_store or VectorStore()

    def add(self, text, memory_type="note", importance=0.5):
        """
        Store a text memory both in the structured (DuckDB) store
        and in the semantic vector index.
        """
        uid = self.storage.add_memory(text, memory_type, importance)
        vec = self.embedder.embed(text)
        self.vectors.add(uid, text, vec, {"memory_type": memory_type})
        self.vectors.save()
        return uid

    def recall(self, query, k=5):
        """
        Retrieve top-k semantically similar memories.
        """
        q_vec = self.embedder.embed(query)
        results = self.vectors.search(q_vec, k)
        return results
