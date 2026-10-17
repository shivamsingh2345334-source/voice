"""Stage 5c: Text -> speech. ElevenLabs (premium) with automatic gTTS fallback."""
import asyncio
import io
import logging
import os
import uuid
from typing import Optional

import httpx

from app.config import settings

log = logging.getLogger("tts")
os.makedirs(settings.audio_out_dir, exist_ok=True)


class TTSEngine:
    async def _elevenlabs(self, text: str) -> bytes:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{settings.elevenlabs_voice_id}"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                url,
                headers={"xi-api-key": settings.elevenlabs_api_key or "", "accept": "audio/mpeg"},
                json={"text": text, "model_id": settings.elevenlabs_model_id},
            )
            r.raise_for_status()
            return r.content

    @staticmethod
    def _gtts_sync(text: str) -> bytes:
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(text=text, lang=settings.gtts_lang, tld=settings.gtts_tld).write_to_fp(buf)
        return buf.getvalue()

    async def speak(self, text: str) -> Optional[str]:
        """Synthesize `text`, save an mp3 and return its filename (served at /audio/<filename>)."""
        if not text:
            return None
        audio: Optional[bytes] = None
        if settings.tts_provider == "elevenlabs" and settings.elevenlabs_api_key:
            try:
                audio = await self._elevenlabs(text)
            except Exception as e:  # noqa: BLE001 - fall back, never fail the request over TTS
                log.warning("ElevenLabs failed (%s); falling back to gTTS", e)
        if audio is None:
            try:
                audio = await asyncio.to_thread(self._gtts_sync, text)
            except Exception as e:  # noqa: BLE001
                log.error("gTTS failed: %s", e)
                return None
        name = f"{uuid.uuid4().hex}.mp3"
        with open(os.path.join(settings.audio_out_dir, name), "wb") as f:
            f.write(audio)
        return name


tts_engine = TTSEngine()
