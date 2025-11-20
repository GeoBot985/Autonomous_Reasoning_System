# Autonomous_Reasoning_System/memory/singletons.py


from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .embeddings import EmbeddingModel
    from .storage import MemoryStorage
    from .vector_store import VectorStore

_embedding_model: Optional[EmbeddingModel] = None
_memory_storage: Optional[MemoryStorage] = None
_vector_store: Optional[VectorStore] = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model ONCE (singleton)")
        from .embeddings import EmbeddingModel  # ← deferred import!
        _embedding_model = EmbeddingModel()
    return _embedding_model

# Modified to allow injection of loaded data or returning None if not initialized
def get_memory_storage(initial_df=None):
    global _memory_storage
    if _memory_storage is None:
        print("Initializing MemoryStorage ONCE (singleton)")
        from .storage import MemoryStorage  # ← deferred import!
        _memory_storage = MemoryStorage(initial_df=initial_df)
    return _memory_storage

# Modified to allow injection of loaded data or returning None if not initialized
def get_vector_store(index=None, metadata=None):
    global _vector_store
    if _vector_store is None:
        print("Initializing VectorStore ONCE (singleton)")
        from .vector_store import VectorStore  # ← deferred import!
        _vector_store = VectorStore(index=index, metadata=metadata)
    return _vector_store
