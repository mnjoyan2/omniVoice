# Speech API

**Version:** 0.1.0 (FastAPI)

**Interactive docs:** `GET /docs` (Swagger UI), `GET /redoc`

## Run the server

From the `speech-api` directory:

1. `uv sync`
2. `./run.sh`

**Listen address:** `HOST` (default `0.0.0.0`), **port:** `PORT` (default `8000`).

Example: `HOST=127.0.0.1 PORT=9000 ./run.sh` → base URL `http://127.0.0.1:9000`.

`run.sh` sets `LD_LIBRARY_PATH` so `sherpa-onnx` can load ONNX Runtime. Use it for normal runs, or set the same `LD_LIBRARY_PATH` if you invoke `uvicorn` directly.

---

## `GET /health`

**Response:** JSON

| Field | Type | Meaning |
|--------|------|--------|
| `status` | string | `"ok"` |
| `stt_model_dir` | string | Directory used for STT (see `OMNILINGUAL_ASR_DIR`) |
| `tts_model` | string | TTS model id/path (see `OMNIVOICE_MODEL`) |

---

## `POST /api/stt`

Speech-to-text.

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |
| **Part name** | `audio` (required) — uploaded file |

**Response:** `200` JSON

```json
{ "text": "<transcript>" }
```

Audio is read with `soundfile` (WAV is typical).

---

## `POST /api/tts`

Text-to-speech (OmniVoice).

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |

| Field | Required | Type | Default | Description |
|--------|----------|------|---------|-------------|
| `text` | yes | string | — | Text to synthesize |
| `language` | no | string | — | Language name or code (e.g. English / `en`) |
| `instruct` | no | string | — | Voice-design instruction |
| `ref_text` | no | string | — | Transcript for reference audio (voice cloning) |
| `ref_audio` | no | file | — | Reference clip for cloning |
| `num_step` | no | int | `32` | Diffusion steps |
| `guidance_scale` | no | float | `2.0` | Guidance scale |
| `speed` | no | float | `1.0` | Speed factor |

**Response:** `200` — raw WAV body, `Content-Type: audio/wav`.

Modes: plain TTS (`text` only), voice design (`instruct`), or cloning (`ref_audio` + `ref_text`).

---

## Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OMNILINGUAL_ASR_DIR` | Folder containing `model.int8.onnx` and `tokens.txt` | Path under OmniVoice `asr_models` |
| `OMNILINGUAL_ASR_THREADS` | STT ONNX threads | `4` |
| `OMNILINGUAL_ASR_PROVIDER` | STT provider (e.g. `cpu`, `cuda` if supported) | `cpu` |
| `OMNIVOICE_MODEL` | Hugging Face id or local path for TTS | `k2-fsa/OmniVoice` |
| `OMNIVOICE_DEVICE` | Force TTS device (`cuda`, `cpu`, …); if unset, auto | auto |
| `CORS_ORIGINS` | Comma-separated allowed browser origins | `*` |
| `PRELOAD_STT` | If `1` / `true` / `yes`, load STT at startup | off |
| `PRELOAD_TTS` | If `1` / `true` / `yes`, load TTS at startup | off |

---

## Examples

```bash
curl -s http://127.0.0.1:8000/health
```

```bash
curl -s -X POST http://127.0.0.1:8000/api/stt -F "audio=@recording.wav"
```

```bash
curl -s -X POST http://127.0.0.1:8000/api/tts -F "text=Hello" -o speech.wav
```
