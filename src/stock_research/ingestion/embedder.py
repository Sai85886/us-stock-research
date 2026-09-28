"""Local text embeddings via sentence-transformers."""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class TextEmbedder:
    """Embed text with a local SentenceTransformer model."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name.strip()
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed one or more documents."""
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        vectors = self.embed_documents([text])
        return vectors[0]
