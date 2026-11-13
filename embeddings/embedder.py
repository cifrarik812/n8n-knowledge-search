"""Local multilingual embeddings; long documents use the mean of line vectors."""
import os
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding
from tokenizers import Tokenizer

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class Embedder:
    def __init__(self, model=None, api_key=None):
        self.model = MODEL
        self.api_key = None
        self.encoder = TextEmbedding(model_name=MODEL, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH", str(Path(__file__).resolve().parents[2] / "model-cache")), threads=4, local_files_only=True)
        self.tokenizer = Tokenizer.from_str(self.encoder.model.tokenizer.to_str())
        self.tokenizer.no_truncation()

    def embed_text(self, text):
        if not text.strip():
            raise ValueError("Empty text")
        parts = [line.strip() for line in text.splitlines() if line.strip()]
        if any(len(self.tokenizer.encode(line).ids) > 128 for line in parts):
            raise ValueError("A line exceeds the model's 128-token limit")
        vectors = np.asarray(list(self.encoder.embed(parts)))
        vector = vectors.mean(axis=0)
        return (vector / np.linalg.norm(vector)).tolist()

    def embed_batch(self, texts):
        return [self.embed_text(text) for text in texts]

    def get_embedding_dimension(self):
        return 384
