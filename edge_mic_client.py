"""Edge client: continuous mic capture + LOCAL Silero VAD; only speech segments are uploaded.
    pip install sounddevice
    python scripts/edge_mic_client.py --mode ingest     # speak memories
    python scripts/edge_mic_client.py --mode query      # ask questions (prints answer + mp3 URL)
"""
import argparse
import io
import queue
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import sounddevice as sd
import torch
from silero_vad import VADIterator, load_silero_vad

SR, CHUNK = 16_000, 512  # Silero requires 512-sample frames @16 kHz


def to_wav(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ingest", "query"], default="ingest")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--user", default="demo_user")
    args = ap.parse_args()
    endpoint = "/ingest_audio" if args.mode == "ingest" else "/query_audio"

    vad = VADIterator(load_silero_vad(), sampling_rate=SR, threshold=0.5, min_silence_duration_ms=700, speech_pad_ms=200)
    pool = ThreadPoolExecutor(max_workers=2)
    frames: "queue.Queue[np.ndarray]" = queue.Queue()

    def send(samples: np.ndarray) -> None:
        if len(samples) < SR * 0.4:  # ignore blips < 0.4 s
            return
        r = httpx.post(args.base + endpoint, files={"file": ("seg.wav", to_wav(samples))}, data={"user_id": args.user}, timeout=120)
        print("\n<<", r.status_code, r.text)

    print(f"Listening ({args.mode} mode). Speak naturally; Ctrl+C to stop.")
    preroll: deque = deque(maxlen=8)  # ~256 ms so word onsets aren't clipped
    buf: list[np.ndarray] = []
    speaking = False
    with sd.InputStream(samplerate=SR, channels=1, blocksize=CHUNK, dtype="float32",
                        callback=lambda d, f, t, s: frames.put(d[:, 0].copy())):
        while True:
            chunk = frames.get()
            ev = vad(torch.from_numpy(chunk), return_seconds=False)
            if ev and "start" in ev:
                speaking, buf = True, list(preroll)
            if speaking:
                buf.append(chunk)
            else:
                preroll.append(chunk)
            if ev and "end" in ev and speaking:
                pool.submit(send, np.concatenate(buf))
                speaking, buf = False, []
                vad.reset_states()


if __name__ == "__main__":
    main()
