"""Stage 2 + 5b: Groq LLM -> strict Pydantic JSON tagging, and natural-language answer synthesis."""
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from app.clients import groq_client
from app.config import settings
from app.schemas import MemoryTag

log = logging.getLogger("llm")


class TaggingError(RuntimeError):
    pass


TAGGING_SYSTEM = """You are a precise information-extraction engine for a personal voice-memory app.
The user's speech was transcribed from Hindi/English/Hinglish. Extract structured data.

Current datetime: {{NOW}}

Return ONLY one JSON object (no markdown, no commentary) that validates against this JSON Schema:
{{SCHEMA}}

Rules:
- Keys must be exactly: Place_to_go, Time, Event, Priority.
- Translate Place_to_go and Event to concise English.
- Resolve relative times (kal=tomorrow, parso=day after tomorrow, aaj=today, subah=morning, shaam=evening,
  raat=night, "2 baje" = 2 o'clock; choose PM for travel/meetings unless clearly morning) into ISO-8601
  'YYYY-MM-DDTHH:MM' using the current datetime. If it cannot be resolved, keep the original phrase.
- Use null (not "") for anything not mentioned.
- Priority: High if time-critical within ~24 hours or involves travel, appointments, deadlines, medicine, money.
  Medium if within about a week. Low otherwise."""

SYNTH_SYSTEM = """You are a warm, concise personal memory assistant speaking out loud.
Current datetime: {{NOW}}

Answer the user's question using ONLY the saved memories provided. Rules:
- Reply in the same language/style as the question (Hinglish in Roman script if they used Hinglish).
- At most 2 short sentences. Plain text only: no markdown, lists, or emojis (it will be read by a TTS voice).
- Convert ISO times into natural speech (e.g. "kal dopahar 2 baje" / "tomorrow at 2 PM").
- If the memories do not answer the question, say you don't have that saved."""


def _now() -> str:
    return datetime.now(ZoneInfo(settings.timezone)).strftime("%A, %Y-%m-%d %H:%M %Z")


def _chat_kwargs() -> dict:
    kw: dict = {"model": settings.groq_llm_model}
    if settings.groq_reasoning_effort:
        kw["reasoning_effort"] = settings.groq_reasoning_effort
    return kw


class LLMEngine:
    async def extract_tags(self, raw_text: str, max_attempts: int = 3) -> MemoryTag:
        system = TAGGING_SYSTEM.replace("{{NOW}}", _now()).replace(
            "{{SCHEMA}}", json.dumps(MemoryTag.model_json_schema(), indent=2)
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Transcript: {raw_text}"},
        ]
        last_err: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            resp = await groq_client.chat.completions.create(
                messages=messages,
                temperature=0,
                max_tokens=1024,  # reasoning models spend tokens thinking; keep headroom
                response_format={"type": "json_object"},
                **_chat_kwargs(),
            )
            content = (resp.choices[0].message.content or "").strip()
            try:
                return MemoryTag.model_validate_json(content)
            except (ValidationError, ValueError) as e:
                last_err = e
                log.warning("Tagging attempt %d failed validation: %s", attempt, e)
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Invalid. Fix it and return ONLY valid JSON. Error: {str(e)[:400]}"},
                ]
        raise TaggingError(f"LLM failed to produce valid tags after {max_attempts} attempts: {last_err}")

    async def synthesize_answer(self, query: str, memories: list[dict]) -> str:
        context = "\n".join(
            f'- saved_at={m.get("created_at")} | said="{m["raw_text"]}" | structured={json.dumps(m["tags"], ensure_ascii=False)}'
            for m in memories
        )
        resp = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYNTH_SYSTEM.replace("{{NOW}}", _now())},
                {"role": "user", "content": f"Saved memories:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=0.3,
            max_tokens=1024,
            **_chat_kwargs(),
        )
        return (resp.choices[0].message.content or "").strip()


llm_engine = LLMEngine()
