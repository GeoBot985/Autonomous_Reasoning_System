# Autonomous_Reasoning_System/memory/vector_memory.py

from Autonomous_Reasoning_System.memory.singletons import (
    get_embedding_model,
    get_memory_storage,
    get_vector_store,
)
import numpy as np


class VectorMemory:
    """
    Combines symbolic (DuckDB) memory with vector-based semantic recall.
    """

    def __init__(self):
        print("🧠 Vector Memory initialized.")
        self.storage = get_memory_storage()
        self.embedder = get_embedding_model()
        self.vectors = get_vector_store()

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
