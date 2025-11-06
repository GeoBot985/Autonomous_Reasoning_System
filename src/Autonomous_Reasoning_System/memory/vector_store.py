# Autonomous_Reasoning_System/memory/vector_store.py
import faiss
import numpy as np
import os
import pickle

class VectorStore:
    """
    Manages FAISS index for semantic similarity search.
    Stores metadata alongside the index for recall and persistence.
    """
    def __init__(self, dim=384, path="data/vector_index.faiss", meta_path="data/vector_meta.pkl"):
        self.dim = dim
        self.path = path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatIP(dim)   # cosine similarity (after normalization)
        self.metadata = []

        if os.path.exists(self.path) and os.path.exists(self.meta_path):
            self.load()

    def add(self, uid: str, text: str, vector: np.ndarray, meta: dict = None):
        if vector.ndim == 1:
            vector = np.expand_dims(vector, axis=0)
        self.index.add(vector.astype(np.float32))
        entry = {"id": uid, "text": text, **(meta or {})}
        self.metadata.append(entry)

    def search(self, query_vec: np.ndarray, k=5):
        if len(self.metadata) == 0:
            return []
        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, axis=0)
        scores, idxs = self.index.search(query_vec.astype(np.float32), k)
        results = []
        for i, score in zip(idxs[0], scores[0]):
            if 0 <= i < len(self.metadata):
                item = self.metadata[i].copy()
                item["score"] = float(score)
                results.append(item)
        return results

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        faiss.write_index(self.index, self.path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(self.path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)
