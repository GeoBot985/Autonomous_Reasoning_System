# memory/vector_memory.py
from .singletons import (
    get_embedding_model,
    get_memory_storage,
    get_vector_store,
)
import numpy as np

class VectorMemory:
    def __init__(self):
        self.storage = get_memory_storage()
        self.embedder = get_embedding_model()
        self.vectors = get_vector_store()

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
