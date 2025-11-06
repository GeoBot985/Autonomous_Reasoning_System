# memory/vector_memory.py
from .storage import MemoryStorage
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
import numpy as np

class VectorMemory:
    def __init__(self):
        self.storage = MemoryStorage()
        self.embedder = EmbeddingModel()
        self.vectors = VectorStore()

    def add(self, text, memory_type="note", importance=0.5):
        # 1. Add to symbolic store
        uid = self.storage.add_memory(text, memory_type, importance)
        # 2. Add to vector store
        vec = self.embedder.embed(text)
        self.vectors.add(uid, text, vec, {"memory_type": memory_type})
        self.vectors.save()
        return uid

    def recall(self, query, k=5):
        q_vec = self.embedder.embed(query)
        results = self.vectors.search(q_vec, k)
        return results
