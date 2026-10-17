"""Central typed configuration loaded from environment / .env."""
from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Groq
    groq_api_key: str
    groq_stt_model: str = "whisper-large-v3"
    groq_llm_model: str = "openai/gpt-oss-20b"
    groq_reasoning_effort: Optional[str] = "low"  # set empty for non-reasoning models
    # Prompt biases Whisper towards Roman-script Hinglish instead of Devanagari/Urdu script.
    stt_prompt: str = "Mujhe kal 2 baje Delhi nikalna hai. Hinglish, Roman script."
    stt_language: Optional[str] = None  # None = auto-detect

    # Supabase
    supabase_url: str
    supabase_service_key: str

    # Upstash
    upstash_redis_rest_url: str
    upstash_redis_rest_token: str
    cache_ttl_seconds: int = 86_400
    cache_max_items: int = 50

    # Retrieval
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384-dim
    cache_similarity_threshold: float = 0.45
    db_similarity_threshold: float = 0.20
    top_k: int = 3

    # TTS
    tts_provider: Literal["gtts", "elevenlabs"] = "gtts"
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    gtts_lang: str = "en"
    gtts_tld: str = "co.in"  # Indian-English accent, reads Roman Hinglish well
    audio_out_dir: str = "audio_out"

    # Edge VAD
    vad_enabled: bool = True
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 500

    # Misc
    timezone: str = "Asia/Kolkata"
    default_user_id: str = "demo_user"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
