# memory/embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer for vector generation.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(384, dtype=np.float32)
        vec = self.model.encode([text], normalize_embeddings=True)
        return vec[0]
