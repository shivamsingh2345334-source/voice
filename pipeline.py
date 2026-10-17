"""Ingestion orchestration: transcript -> tags -> embedding -> Supabase -> (High) Upstash."""
import logging
import time

from app.cache import hot_cache
from app.embeddings import build_embed_text, embedder
from app.llm_engine import llm_engine
from app.memory_db import memory_store
from app.schemas import IngestResponse

log = logging.getLogger("pipeline")


async def ingest_transcript(transcript: str, user_id: str) -> IngestResponse:
    t0 = time.perf_counter()

    tag = await llm_engine.extract_tags(transcript)                       # Stage 2
    embedding = await embedder.embed(build_embed_text(transcript, tag))   # vectorize
    row = await memory_store.save(user_id, transcript, tag, embedding)    # Stage 3

    cached = False
    if tag.Priority == "High":                                            # Stage 4
        try:
            await hot_cache.push_high_priority(user_id, row, embedding)
            cached = True
        except Exception as e:  # noqa: BLE001 - DB write already succeeded; cache is best-effort
            log.warning("Upstash push failed: %s", e)

    return IngestResponse(
        status="stored",
        transcript=transcript,
        tags=tag,
        memory_id=str(row["id"]),
        cached_in_redis=cached,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
