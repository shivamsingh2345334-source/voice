"""Stage 1: Silero VAD (noise/silence filtering) + Groq Whisper STT."""
import asyncio
import io
import logging
import os
import tempfile
import threading
import wave
from typing import Optional

from app.clients import groq_client
from app.config import settings

log = logging.getLogger("audio")
SAMPLE_RATE = 16_000


class AudioProcessor:
    def __init__(self) -> None:
        self._vad_model = None
        self._vad_lock = threading.Lock()  # torch model is not thread-safe

    # ---------- Silero VAD (CPU, runs in a worker thread) ----------
    def _filter_speech_sync(self, audio: bytes, suffix: str) -> tuple[Optional[bytes], float]:
        import torch
        from silero_vad import collect_chunks, get_speech_timestamps, load_silero_vad, read_audio

        with self._vad_lock:
            if self._vad_model is None:
                self._vad_model = load_silero_vad()

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio)
                path = f.name
            try:
                wav = read_audio(path, sampling_rate=SAMPLE_RATE)  # mono float32 tensor @16k
            finally:
                os.unlink(path)

            stamps = get_speech_timestamps(
                wav,
                self._vad_model,
                sampling_rate=SAMPLE_RATE,
                min_speech_duration_ms=settings.vad_min_speech_ms,
                min_silence_duration_ms=settings.vad_min_silence_ms,
            )
            if not stamps:
                return None, 0.0

            speech = collect_chunks(stamps, wav)  # concatenates ONLY the speech segments
            pcm = (speech.clamp(-1, 1) * 32767).to(torch.int16).numpy().tobytes()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
        return buf.getvalue(), len(speech) / SAMPLE_RATE

    async def filter_speech(self, audio: bytes, filename: str = "audio.wav") -> Optional[bytes]:
        """Return WAV bytes containing only speech, or None if no speech was detected."""
        if not settings.vad_enabled:
            return audio
        suffix = os.path.splitext(filename)[1] or ".wav"
        speech, seconds = await asyncio.to_thread(self._filter_speech_sync, audio, suffix)
        log.info("VAD: %.2fs of speech kept", seconds)
        return speech

    # ---------- Groq Whisper ----------
    async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
        kwargs = dict(
            file=(filename, audio),
            model=settings.groq_stt_model,
            prompt=settings.stt_prompt,
            response_format="json",
            temperature=0.0,
        )
        if settings.stt_language:
            kwargs["language"] = settings.stt_language
        result = await groq_client.audio.transcriptions.create(**kwargs)
        return (result.text or "").strip()


audio_processor = AudioProcessor()
