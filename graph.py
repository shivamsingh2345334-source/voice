"""Stage 5: LangGraph retrieval agent.

START -> embed_query -> cache_lookup --hit--> synthesize --> tts --> END
                              |                   ^
                              miss                |
                              v                   |
                          db_lookup --found-------+
                              |
                              not found --> not_found --> tts
"""
import logging
from typing import Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.cache import hot_cache
from app.config import settings
from app.embeddings import embedder
from app.llm_engine import llm_engine
from app.memory_db import memory_store
from app.tts import tts_engine

log = logging.getLogger("graph")

NOT_FOUND_MSG = "Mujhe is bare mein kuch saved nahi mila. I couldn't find anything saved about that."


class RetrievalState(TypedDict, total=False):
    user_id: str
    query: str
    speak: bool
    query_embedding: list[float]
    memories: list[dict]
    source: Literal["cache", "supabase", "none"]
    answer: str
    audio_file: Optional[str]
    trace: list[str]


def _t(state: RetrievalState, step: str) -> list[str]:
    return [*state.get("trace", []), step]


# ---------------- Nodes ----------------
async def embed_query(state: RetrievalState) -> dict:
    emb = await embedder.embed(state["query"])
    return {"query_embedding": emb, "trace": _t(state, "embed_query")}


async def cache_lookup(state: RetrievalState) -> dict:
    try:
        hits = await hot_cache.lookup(
            state["user_id"], state["query_embedding"], settings.top_k, settings.cache_similarity_threshold
        )
    except Exception as e:  # noqa: BLE001 - cache outage must degrade to DB, not fail
        log.warning("Redis lookup failed: %s", e)
        hits = []
    if hits:
        return {"memories": hits, "source": "cache", "trace": _t(state, "cache_HIT")}
    return {"memories": [], "trace": _t(state, "cache_MISS")}


async def db_lookup(state: RetrievalState) -> dict:
    rows = await memory_store.search(
        state["user_id"], state["query_embedding"], settings.top_k, settings.db_similarity_threshold
    )
    if rows:
        return {"memories": rows, "source": "supabase", "trace": _t(state, "supabase_FOUND")}
    return {"memories": [], "source": "none", "trace": _t(state, "supabase_EMPTY")}


async def synthesize(state: RetrievalState) -> dict:
    answer = await llm_engine.synthesize_answer(state["query"], state["memories"])
    return {"answer": answer, "trace": _t(state, "llm_synthesize")}


async def not_found(state: RetrievalState) -> dict:
    return {"answer": NOT_FOUND_MSG, "trace": _t(state, "not_found")}


async def tts(state: RetrievalState) -> dict:
    if not state.get("speak"):
        return {"audio_file": None}
    return {"audio_file": await tts_engine.speak(state["answer"]), "trace": _t(state, "tts")}


# ---------------- Routers ----------------
def route_after_cache(state: RetrievalState) -> str:
    return "synthesize" if state.get("source") == "cache" else "db_lookup"


def route_after_db(state: RetrievalState) -> str:
    return "synthesize" if state.get("memories") else "not_found"


def build_graph():
    g = StateGraph(RetrievalState)
    g.add_node("embed_query", embed_query)
    g.add_node("cache_lookup", cache_lookup)
    g.add_node("db_lookup", db_lookup)
    g.add_node("synthesize", synthesize)
    g.add_node("not_found", not_found)
    g.add_node("tts", tts)

    g.add_edge(START, "embed_query")
    g.add_edge("embed_query", "cache_lookup")
    g.add_conditional_edges("cache_lookup", route_after_cache, {"synthesize": "synthesize", "db_lookup": "db_lookup"})
    g.add_conditional_edges("db_lookup", route_after_db, {"synthesize": "synthesize", "not_found": "not_found"})
    g.add_edge("synthesize", "tts")
    g.add_edge("not_found", "tts")
    g.add_edge("tts", END)
    return g.compile()


retrieval_graph = build_graph()
