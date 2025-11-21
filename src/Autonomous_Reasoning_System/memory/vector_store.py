# Autonomous_Reasoning_System/memory/vector_store.py
import faiss
import numpy as np
import os
import pickle

class VectorStore:
    """
    Manages FAISS index for semantic similarity search.
    Stores metadata alongside the index for recall and persistence.
    Persistence is handled by MemoryInterface via PersistenceService.
    """
    def __init__(self, dim=384, index=None, metadata=None):
        self.dim = dim
        self.index = index if index is not None else faiss.IndexFlatIP(dim)   # cosine similarity (after normalization)
        self.metadata = metadata if metadata is not None else []
        self.id_map = {} # Maps uid -> list of indices in metadata/faiss
        self._rebuild_id_map()

    def _rebuild_id_map(self):
        """Rebuilds the internal ID map from metadata."""
        self.id_map = {}
        for idx, item in enumerate(self.metadata):
            if item.get("deleted"):
                continue
            uid = item.get("id")
            if uid:
                if uid not in self.id_map:
                    self.id_map[uid] = []
                self.id_map[uid].append(idx)

    def add(self, uid: str, text: str, vector: np.ndarray, meta: dict = None):
        if vector.ndim == 1:
            vector = np.expand_dims(vector, axis=0)
        self.index.add(vector.astype(np.float32))

        # The new item is at the end of the list
        idx = len(self.metadata)

        entry = {"id": uid, "text": text, **(meta or {})}
        self.metadata.append(entry)

        if uid:
            if uid not in self.id_map:
                self.id_map[uid] = []
            self.id_map[uid].append(idx)

    def soft_delete(self, uid: str):
        """
        Marks all entries with the given UID as deleted.
        They will be filtered out during search.
        """
        if uid in self.id_map:
            for idx in self.id_map[uid]:
                if 0 <= idx < len(self.metadata):
                    self.metadata[idx]["deleted"] = True
            # Remove from map so we don't track it anymore
            del self.id_map[uid]
            return True
        return False

    def search(self, query_vec: np.ndarray, k=5):
        if len(self.metadata) == 0:
            return []
        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, axis=0)

        # We might need to search more than k if there are deleted items
        # Heuristic: search k * 2 + deleted_count?
        # Or just search larger k and filter.
        # Since we don't know how many deleted items are in top-k, we'll fetch more.
        search_k = min(len(self.metadata), k * 3)

        scores, idxs = self.index.search(query_vec.astype(np.float32), search_k)
        results = []

        for i, score in zip(idxs[0], scores[0]):
            if 0 <= i < len(self.metadata):
                item = self.metadata[i]
                if item.get("deleted"):
                    continue

                # Create a copy to return
                res_item = item.copy()
                res_item["score"] = float(score)
                results.append(res_item)

                if len(results) >= k:
                    break

        return results

    def reset(self):
        """Clear the index and metadata."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        self.id_map = {}
