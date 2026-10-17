"""Stage 4: Upstash Redis hot-cache for High-priority memories.

Layout: one capped Redis LIST per user -> key `mem:high:{user_id}`; each element is a JSON blob that also
carries the embedding, so a cache lookup is a cosine-similarity scan over <= CACHE_MAX_ITEMS entries
(microseconds) with zero Postgres round-trips.
"""
import json
import logging

import numpy as np
from upstash_redis.asyncio import Redis

from app.config import settings

log = logging.getLogger("cache")


class HotCache:
    def __init__(self) -> None:
        self._redis = Redis(url=settings.upstash_redis_rest_url, token=settings.upstash_redis_rest_token)

    @staticmethod
    def _key(user_id: str) -> str:
        return f"mem:high:{user_id}"

    async def push_high_priority(self, user_id: str, row: dict, embedding: list[float]) -> None:
        item = {
            "id": row["id"],
            "raw_text": row["raw_text"],
            "tags": row["tags"],
            "created_at": row.get("created_at"),
            "embedding": [round(x, 5) for x in embedding],
        }
        key = self._key(user_id)
        pipe = self._redis.pipeline()
        pipe.lpush(key, json.dumps(item, ensure_ascii=False))
        pipe.ltrim(key, 0, settings.cache_max_items - 1)
        pipe.expire(key, settings.cache_ttl_seconds)
        await pipe.execute()

    async def lookup(self, user_id: str, query_embedding: list[float], k: int, threshold: float) -> list[dict]:
        raw_items = await self._redis.lrange(self._key(user_id), 0, -1)
        if not raw_items:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-9

        hits: list[dict] = []
        for raw in raw_items:
            item = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
            v = np.asarray(item.pop("embedding"), dtype=np.float32)
            sim = float(np.dot(q, v / (np.linalg.norm(v) + 1e-9)))
            if sim >= threshold:
                item["similarity"] = round(sim, 4)
                hits.append(item)
        hits.sort(key=lambda m: m["similarity"], reverse=True)
        return hits[:k]


hot_cache = HotCache()
