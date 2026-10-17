# Voice-to-Memory AI Agent System (POC v3.0)

```
Mic ─► Silero VAD ─► Groq Whisper ─► Groq LLM (Pydantic JSON) ─► Supabase (JSONB + pgvector)
                                                                     └─ if Priority=High ─► Upstash Redis
Query ─► LangGraph: embed ─► Redis hit? ──yes──► Groq synth ─► gTTS/ElevenLabs ─► mp3
                               └─no─► Supabase vector search ─► Groq synth ─► TTS
```

## 1. Supabase
1. Create a project at https://supabase.com (free tier is fine).
2. SQL Editor -> paste all of `sql/schema.sql` -> Run.
3. Project Settings -> API: copy **Project URL** -> `SUPABASE_URL`, and the **service_role** key -> `SUPABASE_SERVICE_KEY`.

## 2. Upstash Redis
1. https://console.upstash.com -> Create Database (Redis), pick a region near you.
2. Open the **REST API** section -> copy `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` into `.env`.
   (Use the REST credentials, not the `redis://` TCP URL.)

## 3. Groq (+ optional ElevenLabs)
- Key from https://console.groq.com/keys -> `GROQ_API_KEY`.
- ElevenLabs is optional: set `TTS_PROVIDER=elevenlabs` and `ELEVENLABS_API_KEY`. Otherwise gTTS is used (no key).

## 4. Run
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env     # fill in keys
uvicorn app.main:app --reload --port 8000
```
Swagger UI: http://127.0.0.1:8000/docs
First request is slower: Silero VAD + the embedding model (~120 MB) load/download lazily.
Non-WAV uploads (mp3/webm/m4a) need `ffmpeg` installed for the VAD step.

## 5. Test
```bash
python scripts/test_pipeline.py                 # text-only end-to-end
python scripts/test_pipeline.py --audio my.wav  # + real audio through VAD/Whisper
```
curl:
```bash
# ingest (no mic needed)
curl -s -X POST localhost:8000/ingest_text -H 'Content-Type: application/json' \
  -d '{"text":"Mujhe kal 2 baje Delhi nikalna hai","user_id":"demo_user"}' | python -m json.tool

# ingest real audio
curl -s -X POST localhost:8000/ingest_audio -F file=@recording.wav -F user_id=demo_user | python -m json.tool

# query -> expect "source":"cache" and trace containing cache_HIT
curl -s -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query":"Main Delhi kab ja raha hoon?","user_id":"demo_user","speak":true}' | python -m json.tool

# play the voice answer
curl -s localhost:8000/audio/<file>.mp3 -o answer.mp3
```
Live mic (edge VAD): `pip install sounddevice && python scripts/edge_mic_client.py --mode ingest`

## Troubleshooting
- `model_decommissioned` / 404 from Groq: update `GROQ_LLM_MODEL` (list live IDs: `GET https://api.groq.com/openai/v1/models`).
- `reasoning_effort` error: you switched to a non-reasoning model -> set `GROQ_REASONING_EFFORT=` (empty).
- Query returns "not found": lower `DB_SIMILARITY_THRESHOLD` (0.10-0.20) - thresholds need tuning on your data.
- Changing `EMBEDDING_MODEL` to a different dimension requires changing `vector(384)` in the SQL.
