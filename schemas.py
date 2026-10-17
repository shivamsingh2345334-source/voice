"""Pydantic models: the strict auto-tagging contract + API request/response types."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NULLISH = {"", "null", "none", "n/a", "na", "unknown", "-"}


class MemoryTag(BaseModel):
    """Zero-shot extraction target. Key names are EXACTLY what the architecture specifies."""

    model_config = ConfigDict(extra="ignore")

    Place_to_go: Optional[str] = Field(
        default=None, description="Destination / location mentioned, in English (e.g. 'Delhi'). null if none."
    )
    Time: Optional[str] = Field(
        default=None,
        description="When it happens. ISO-8601 'YYYY-MM-DDTHH:MM' if resolvable from the current datetime, "
        "otherwise the original time phrase. null if none.",
    )
    Event: str = Field(description="Short English description of the activity/event (<= 12 words).")
    Priority: Literal["High", "Medium", "Low"] = Field(
        description="High = time-critical within ~24h (travel, appointments, deadlines, medicine, money). "
        "Medium = within ~a week. Low = casual / no time pressure."
    )

    @field_validator("Place_to_go", "Time", mode="before")
    @classmethod
    def _nullish_to_none(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() in _NULLISH:
            return None
        return v

    @field_validator("Priority", mode="before")
    @classmethod
    def _normalise_priority(cls, v: Any) -> Any:
        return str(v).strip().capitalize() if v else "Medium"


class IngestTextRequest(BaseModel):
    text: str = Field(min_length=1)
    user_id: Optional[str] = None


class IngestResponse(BaseModel):
    status: Literal["stored", "no_speech_detected"]
    transcript: Optional[str] = None
    tags: Optional[MemoryTag] = None
    memory_id: Optional[str] = None
    cached_in_redis: bool = False
    latency_ms: int = 0


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    user_id: Optional[str] = None
    speak: bool = True


class RetrievedMemory(BaseModel):
    id: str
    raw_text: str
    tags: dict
    similarity: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    source: Literal["cache", "supabase", "none"]
    memories: list[RetrievedMemory] = []
    audio_url: Optional[str] = None
    trace: list[str] = []
    latency_ms: int = 0
