"""End-to-end smoke test.  Usage:
    python scripts/test_pipeline.py                  # text-only (no mic / audio file needed)
    python scripts/test_pipeline.py --audio my.wav   # also test /ingest_audio with a real recording
"""
import argparse
import json
import sys

import httpx

BASE = "http://127.0.0.1:8000"
USER = "demo_user"


def show(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", help="path to a WAV/MP3 recording to send to /ingest_audio")
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()

    with httpx.Client(base_url=args.base, timeout=120) as c:
        print("health:", c.get("/health").json())

        # 1) INGEST: one High (-> Redis) and one Low (Supabase only)
        for text in ["Mujhe kal 2 baje Delhi nikalna hai", "Kabhi time mile toh naya Netflix show dekhna hai"]:
            r = c.post("/ingest_text", json={"text": text, "user_id": USER})
            r.raise_for_status()
            show(f"INGEST: {text}", r.json())

        if args.audio:
            with open(args.audio, "rb") as f:
                r = c.post("/ingest_audio", files={"file": (args.audio, f)}, data={"user_id": USER})
            r.raise_for_status()
            show("INGEST AUDIO", r.json())

        # 2) QUERY: first should be a Redis cache HIT, second goes to Supabase
        for q in ["Main Delhi kab ja raha hoon?", "What show did I want to watch?"]:
            r = c.post("/query", json={"query": q, "user_id": USER, "speak": True})
            r.raise_for_status()
            body = r.json()
            show(f"QUERY: {q}", body)
            print(f"-> source={body['source']}  trace={' > '.join(body['trace'])}")
            if body.get("audio_url"):
                print(f"-> voice reply: {args.base}{body['audio_url']}")


if __name__ == "__main__":
    try:
        main()
    except httpx.HTTPError as e:
        sys.exit(f"HTTP error: {e}")
