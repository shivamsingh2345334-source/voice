"""Local multilingual embeddings via fastembed (ONNX). 384-dim -> matches vector(384) in schema.sql."""
import asyncio
import threading

from app.config import settings
from app.schemas import MemoryTag


def build_embed_text(raw_text: str, tag: MemoryTag) -> str:
    """Canonical English-heavy string so Hinglish memories match English queries (and vice-versa)."""
    parts = [f"Event: {tag.Event}"]
    if tag.Place_to_go:
        parts.append(f"Place: {tag.Place_to_go}")
    if tag.Time:
        parts.append(f"Time: {tag.Time}")
    parts.append(f"Original: {raw_text}")
    return ". ".join(parts)


class Embedder:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _embed_sync(self, text: str) -> list[float]:
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=settings.embedding_model)
            return next(iter(self._model.embed([text]))).tolist()

    async def embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._embed_sync, text)


embedder = Embedder()
