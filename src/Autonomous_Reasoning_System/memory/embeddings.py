# Autonomous_Reasoning_System/memory/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    """
    Lightweight wrapper for generating semantic embeddings.
    Uses a local SentenceTransformer model for efficient inference.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"🔤 Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """
        Return a 384-dimensional normalized vector for the given text.
        """
        if not text or not text.strip():
            return np.zeros(384, dtype=np.float32)
        vec = self.model.encode([text], normalize_embeddings=True)
        return vec[0].astype(np.float32)
