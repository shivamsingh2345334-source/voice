"""Stage 3: Supabase (Postgres + pgvector) persistence and semantic retrieval."""
import asyncio
from typing import Optional

from supabase import AsyncClient, acreate_client

from app.config import settings
from app.schemas import MemoryTag


class MemoryStore:
    def __init__(self) -> None:
        self._client: Optional[AsyncClient] = None
        self._lock = asyncio.Lock()

    async def _db(self) -> AsyncClient:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = await acreate_client(settings.supabase_url, settings.supabase_service_key)
        return self._client

    async def save(self, user_id: str, raw_text: str, tag: MemoryTag, embedding: list[float]) -> dict:
        db = await self._db()
        row = {
            "user_id": user_id,
            "raw_text": raw_text,
            "tags": tag.model_dump(),  # -> JSONB
            "embedding": embedding,    # -> vector(384)
        }
        res = await db.table("memories").insert(row).execute()
        return res.data[0]

    async def search(self, user_id: str, embedding: list[float], k: int, min_similarity: float) -> list[dict]:
        db = await self._db()
        res = await db.rpc(
            "match_memories",
            {
                "query_embedding": embedding,
                "p_user_id": user_id,
                "match_count": k,
                "min_similarity": min_similarity,
            },
        ).execute()
        return res.data or []


memory_store = MemoryStore()
