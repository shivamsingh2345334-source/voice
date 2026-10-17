"""FastAPI entrypoint.  Run:  uvicorn app.main:app --reload --port 8000"""
import logging
import os
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from app.audio_processor import audio_processor
from app.config import settings
from app.graph import retrieval_graph
from app.llm_engine import TaggingError
from app.pipeline import ingest_transcript
from app.schemas import (
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    RetrievedMemory,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Voice-to-Memory AI Agent (POC)", version="3.0")
os.makedirs(settings.audio_out_dir, exist_ok=True)
app.mount("/audio", StaticFiles(directory=settings.audio_out_dir), name="audio")


async def _read_audio(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty audio file")
    return data


async def _run_query(query: str, user_id: str, speak: bool, t0: float) -> QueryResponse:
    state = await retrieval_graph.ainvoke(
        {"query": query, "user_id": user_id, "speak": speak, "trace": []}
    )
    audio_file = state.get("audio_file")
    return QueryResponse(
        query=query,
        answer=state["answer"],
        source=state.get("source", "none"),
        memories=[
            RetrievedMemory(
                id=str(m["id"]), raw_text=m["raw_text"], tags=m["tags"], similarity=float(m.get("similarity", 0))
            )
            for m in state.get("memories", [])
        ],
        audio_url=f"/audio/{audio_file}" if audio_file else None,
        trace=state.get("trace", []),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "llm": settings.groq_llm_model, "stt": settings.groq_stt_model}


# ---------- Ingestion ----------
@app.post("/ingest_audio", response_model=IngestResponse)
async def ingest_audio(file: UploadFile = File(...), user_id: Optional[str] = Form(None)):
    """Audio -> Silero VAD -> Groq Whisper -> Llama tagging -> Supabase -> (High) Upstash."""
    t0 = time.perf_counter()
    audio = await _read_audio(file)
    speech = await audio_processor.filter_speech(audio, file.filename or "audio.wav")
    if speech is None:
        return IngestResponse(status="no_speech_detected", latency_ms=int((time.perf_counter() - t0) * 1000))

    transcript = await audio_processor.transcribe(speech)
    if not transcript:
        return IngestResponse(status="no_speech_detected", latency_ms=int((time.perf_counter() - t0) * 1000))
    try:
        result = await ingest_transcript(transcript, user_id or settings.default_user_id)
    except TaggingError as e:
        raise HTTPException(422, str(e)) from e
    result.latency_ms = int((time.perf_counter() - t0) * 1000)
    return result


@app.post("/ingest_text", response_model=IngestResponse)
async def ingest_text(req: IngestTextRequest):
    """Same pipeline minus audio - handy for testing without a microphone."""
    try:
        return await ingest_transcript(req.text.strip(), req.user_id or settings.default_user_id)
    except TaggingError as e:
        raise HTTPException(422, str(e)) from e


# ---------- Retrieval ----------
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Text question -> LangGraph (Redis -> Supabase) -> Groq answer -> TTS."""
    return await _run_query(req.query, req.user_id or settings.default_user_id, req.speak, time.perf_counter())


@app.post("/query_audio", response_model=QueryResponse)
async def query_audio(file: UploadFile = File(...), user_id: Optional[str] = Form(None), speak: bool = Form(True)):
    """Spoken question -> VAD -> Whisper -> same LangGraph agent."""
    t0 = time.perf_counter()
    speech = await audio_processor.filter_speech(await _read_audio(file), file.filename or "audio.wav")
    if speech is None:
        raise HTTPException(400, "No speech detected in audio")
    question = await audio_processor.transcribe(speech)
    if not question:
        raise HTTPException(400, "Could not transcribe audio")
    return await _run_query(question, user_id or settings.default_user_id, speak, t0)
