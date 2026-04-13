import io
import logging
import os
import sys
import ctypes
from pathlib import Path
import re
import tempfile
import threading
import time
import uuid
import zipfile
from typing import Optional

for _p in sys.path:
    if _p.endswith("site-packages"):
        _ort = Path(_p) / "onnxruntime" / "capi" / "libonnxruntime.so"
        if _ort.is_file():
            ctypes.CDLL(str(_ort), mode=ctypes.RTLD_GLOBAL)
            break

import numpy as np
import sherpa_onnx
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydub import AudioSegment
from starlette.background import BackgroundTask

from omnivoice.cli.infer import get_best_device
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAME_TO_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_itn_cache: dict[str, object] = {}
_tn_cache: dict[str, object] = {}


def get_itn(lang: str):
    key = lang.lower()
    if key not in _itn_cache:
        from nemo_text_processing.inverse_text_normalization import InverseNormalizer
        input_case = "cased" if key == "hy" else "lower_cased"
        _itn_cache[key] = InverseNormalizer(lang=key, input_case=input_case)
        logger.info("Loaded ITN for %s", key)
    return _itn_cache[key]


def get_tn(lang: str):
    key = lang.lower()
    if key not in _tn_cache:
        from nemo_text_processing.text_normalization import Normalizer
        input_case = "cased" if key == "hy" else "lower_cased"
        _tn_cache[key] = Normalizer(input_case=input_case, lang=key)
        logger.info("Loaded TN for %s", key)
    return _tn_cache[key]


ITN_SUPPORTED_LANGS = {"en", "es", "pt", "ru", "de", "fr", "hy"}
TN_SUPPORTED_LANGS = {"en", "es", "pt", "ru", "de", "fr", "hy"}

STT_MODEL_DIR = os.environ.get(
    "OMNILINGUAL_ASR_DIR",
    "/workspace/OmniVoice/asr_models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12",
)
TTS_MODEL_ID = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
STT_NUM_THREADS = int(os.environ.get("OMNILINGUAL_ASR_THREADS", "4"))
STT_PROVIDER = os.environ.get("OMNILINGUAL_ASR_PROVIDER", "cpu")

VOICES_DIR = os.environ.get("VOICES_DIR", "/workspace/voices")
CUSTOM_VOICES_DIR = os.environ.get("CUSTOM_VOICES_DIR", "/workspace/voices/custom")

VOICES = {
    "angry_marcus": {"emotion": "angry", "gender": "male", "pitch": "low", "age": "young adult"},
    "angry_diana": {"emotion": "angry", "gender": "female", "pitch": "moderate", "age": "young adult"},
    "angry_victor": {"emotion": "angry", "gender": "male", "pitch": "moderate", "age": "middle-aged"},
    "sad_elena": {"emotion": "sad", "gender": "female", "pitch": "low", "age": "young adult"},
    "sad_james": {"emotion": "sad", "gender": "male", "pitch": "low", "age": "middle-aged"},
    "sad_margaret": {"emotion": "sad", "gender": "female", "pitch": "low", "age": "elderly"},
    "happy_aria": {"emotion": "happy", "gender": "female", "pitch": "high", "age": "young adult"},
    "happy_leo": {"emotion": "happy", "gender": "male", "pitch": "high", "age": "young adult"},
    "happy_sophie": {"emotion": "happy", "gender": "female", "pitch": "moderate", "age": "teenager"},
    "calm_oliver": {"emotion": "calm", "gender": "male", "pitch": "moderate", "age": "middle-aged"},
    "calm_nina": {"emotion": "calm", "gender": "female", "pitch": "low", "age": "young adult"},
    "calm_henry": {"emotion": "calm", "gender": "male", "pitch": "low", "age": "elderly"},
    "excited_luna": {"emotion": "excited", "gender": "female", "pitch": "very high", "age": "young adult"},
    "excited_ryan": {"emotion": "excited", "gender": "male", "pitch": "high", "age": "young adult"},
    "excited_mia": {"emotion": "excited", "gender": "female", "pitch": "high", "age": "teenager"},
    "serious_grant": {"emotion": "serious", "gender": "male", "pitch": "low", "age": "middle-aged"},
    "serious_claire": {"emotion": "serious", "gender": "female", "pitch": "moderate", "age": "middle-aged"},
    "serious_walter": {"emotion": "serious", "gender": "male", "pitch": "very low", "age": "elderly"},
    "gentle_emma": {"emotion": "gentle", "gender": "female", "pitch": "low", "age": "young adult"},
    "gentle_thomas": {"emotion": "gentle", "gender": "male", "pitch": "moderate", "age": "middle-aged"},
    "gentle_rose": {"emotion": "gentle", "gender": "female", "pitch": "moderate", "age": "elderly"},
    "confident_alex": {"emotion": "confident", "gender": "male", "pitch": "moderate", "age": "young adult"},
    "confident_sarah": {"emotion": "confident", "gender": "female", "pitch": "high", "age": "young adult"},
    "confident_daniel": {"emotion": "confident", "gender": "male", "pitch": "low", "age": "middle-aged"},
    "tired_nathan": {"emotion": "tired", "gender": "male", "pitch": "low", "age": "young adult"},
    "tired_lisa": {"emotion": "tired", "gender": "female", "pitch": "low", "age": "young adult"},
    "tired_george": {"emotion": "tired", "gender": "male", "pitch": "low", "age": "elderly"},
    "neutral_sam": {"emotion": "neutral", "gender": "male", "pitch": "moderate", "age": "young adult"},
    "neutral_kate": {"emotion": "neutral", "gender": "female", "pitch": "moderate", "age": "young adult"},
    "neutral_david": {"emotion": "neutral", "gender": "male", "pitch": "moderate", "age": "middle-aged"},
}

CUSTOM_VOICES: dict[str, dict] = {}

JOB_TTL_SECONDS = 600
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

_recognizer = None
_tts_model = None
_sfx_model = None
def get_sfx():
    global _sfx_model
    if _sfx_model is None:
        from audiocraft.models import AudioGen
        logger.info("Loading AudioGen model (facebook/audiogen-medium)...")
        _sfx_model = AudioGen.get_pretrained("facebook/audiogen-medium")
        logger.info("AudioGen model ready")
    return _sfx_model


def load_audio(path):
    samples, sample_rate = sf.read(path, dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]
    return samples, sample_rate


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        tokens_path = os.path.join(STT_MODEL_DIR, "tokens.txt")
        model_path = os.path.join(STT_MODEL_DIR, "model.int8.onnx")
        _recognizer = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=model_path,
            tokens=tokens_path,
            num_threads=STT_NUM_THREADS,
            decoding_method="greedy_search",
            debug=False,
            provider=STT_PROVIDER,
        )
        logger.info("STT recognizer ready")
    return _recognizer


def get_tts():
    global _tts_model
    if _tts_model is None:
        device = os.environ.get("OMNIVOICE_DEVICE") or get_best_device()
        load_asr = os.environ.get("OMNIVOICE_LOAD_ASR", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        logger.info("Loading TTS model %s on %s", TTS_MODEL_ID, device)
        _tts_model = OmniVoice.from_pretrained(
            TTS_MODEL_ID,
            device_map=device,
            dtype=torch.float16,
            load_asr=load_asr,
        )
        logger.info("TTS model ready")
    return _tts_model


def _resolve_language(language: str | None) -> str | None:
    if language is None:
        return None
    s = language.strip()
    if not s or s.lower() == "auto":
        return None
    return s


def _cleanup_task(*paths):
    def _clean():
        for p in paths:
            if p and os.path.exists(p):
                os.unlink(p)
    return BackgroundTask(_clean)


def _all_voices() -> dict[str, dict]:
    merged = {}
    for name, meta in VOICES.items():
        merged[name] = {**meta, "custom": False}
    for name, meta in CUSTOM_VOICES.items():
        merged[name] = {**meta, "custom": True}
    return merged


def _resolve_voice_path(voice_name: str) -> str:
    if voice_name in VOICES:
        return os.path.join(VOICES_DIR, f"{voice_name}.wav")
    if voice_name in CUSTOM_VOICES:
        return os.path.join(CUSTOM_VOICES_DIR, f"{voice_name}.wav")
    raise HTTPException(
        status_code=400,
        detail=f"Unknown voice '{voice_name}'. Use GET /api/voices to list available voices.",
    )


def _is_library_path(path: str | None) -> bool:
    if not path:
        return False
    return path.startswith(VOICES_DIR) or path.startswith(CUSTOM_VOICES_DIR)


def _generate_audio(
    text: str,
    voice: str | None = None,
    language: str | None = None,
    instruct: str | None = None,
    ref_text: str | None = None,
    ref_audio_path: str | None = None,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    t_shift: float = 0.1,
    denoise: bool = True,
    speed: float = 1.0,
    duration: float | None = None,
    preprocess_prompt: bool = True,
    postprocess_output: bool = True,
    layer_penalty_factor: float = 5.0,
    position_temperature: float = 5.0,
    class_temperature: float = 0.0,
    audio_chunk_duration: float = 15.0,
    audio_chunk_threshold: float = 30.0,
) -> tuple[np.ndarray, int]:
    model = get_tts()

    ref_path = ref_audio_path
    if not ref_path and voice and voice.strip():
        vp = _resolve_voice_path(voice.strip())
        if not os.path.isfile(vp):
            raise HTTPException(status_code=500, detail=f"Voice file for '{voice}' is missing on disk")
        ref_path = vp

    gen_config = OmniVoiceGenerationConfig(
        num_step=num_step,
        guidance_scale=guidance_scale,
        t_shift=t_shift,
        denoise=denoise,
        preprocess_prompt=preprocess_prompt,
        postprocess_output=postprocess_output,
        layer_penalty_factor=layer_penalty_factor,
        position_temperature=position_temperature,
        class_temperature=class_temperature,
        audio_chunk_duration=audio_chunk_duration,
        audio_chunk_threshold=audio_chunk_threshold,
    )

    kw: dict = {
        "text": text.strip(),
        "language": _resolve_language(language),
        "generation_config": gen_config,
    }

    if ref_path is not None:
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
            ref_audio=ref_path,
            ref_text=ref_text or None,
            preprocess_prompt=gen_config.preprocess_prompt,
        )
    elif instruct and instruct.strip():
        kw["instruct"] = instruct.strip()

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    audios = model.generate(**kw)
    waveform = audios[0].squeeze(0).cpu().numpy()
    waveform_int16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
    return waveform_int16, model.sampling_rate


def _save_wav(waveform: np.ndarray, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, waveform, sr, format="WAV", subtype="PCM_16")
    return tmp.name


def _wav_to_mp3(wav_path: str) -> str:
    mp3_path = wav_path.rsplit(".", 1)[0] + ".mp3"
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3", bitrate="192k")
    return mp3_path


def _purge_expired_jobs():
    now = time.time()
    expired = [jid for jid, j in _jobs.items() if now - j["created_at"] > JOB_TTL_SECONDS]
    for jid in expired:
        j = _jobs.pop(jid, None)
        if j and j.get("file") and os.path.exists(j["file"]):
            os.unlink(j["file"])


def _load_custom_voices():
    os.makedirs(CUSTOM_VOICES_DIR, exist_ok=True)
    for fname in os.listdir(CUSTOM_VOICES_DIR):
        if not fname.endswith(".wav"):
            continue
        name = fname[:-4]
        if name not in VOICES:
            CUSTOM_VOICES[name] = {"emotion": "custom", "gender": "unknown"}


app = FastAPI(title="Speech API", version="0.2.0")
_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
_cors_credentials = os.environ.get("CORS_ALLOW_CREDENTIALS", "false").lower() in ("1", "true", "yes")
if "*" in _cors_origins:
    _cors_credentials = False
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "stt_model_dir": STT_MODEL_DIR,
        "tts_model": TTS_MODEL_ID,
    }


@app.get("/api/languages")
def list_languages():
    return {
        "languages": [
            {"name": name, "code": code}
            for name, code in sorted(LANG_NAME_TO_ID.items())
        ]
    }


@app.get("/api/voices")
def list_voices():
    all_v = _all_voices()
    return {
        "voices": [
            {"name": name, **meta}
            for name, meta in all_v.items()
        ]
    }


@app.get("/api/voices/{name}/sample")
def voice_sample(name: str):
    if name in VOICES:
        path = os.path.join(VOICES_DIR, f"{name}.wav")
    elif name in CUSTOM_VOICES:
        path = os.path.join(CUSTOM_VOICES_DIR, f"{name}.wav")
    else:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Sample file for '{name}' not found")
    return FileResponse(path, media_type="audio/wav", filename=f"{name}.wav")


@app.post("/api/voices/clone")
async def clone_voice(
    name: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: str | None = Form(None),
    gender: str | None = Form(None),
    emotion: str | None = Form(None),
):
    name = name.strip()
    if not re.match(r"^[a-z0-9_]+$", name):
        raise HTTPException(status_code=400, detail="Name must be lowercase alphanumeric with underscores only")
    if name in VOICES:
        raise HTTPException(status_code=409, detail=f"'{name}' conflicts with a built-in voice")
    if name in CUSTOM_VOICES:
        raise HTTPException(status_code=409, detail=f"'{name}' already exists. Delete it first.")

    os.makedirs(CUSTOM_VOICES_DIR, exist_ok=True)
    rsfx = os.path.splitext(ref_audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=rsfx) as f:
        f.write(await ref_audio.read())
        tmp_path = f.name

    try:
        samples, sr = sf.read(tmp_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples[:, 0]
        dest = os.path.join(CUSTOM_VOICES_DIR, f"{name}.wav")
        sf.write(dest, samples, sr, format="WAV", subtype="PCM_16")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    meta = {
        "emotion": emotion or "custom",
        "gender": gender or "unknown",
    }
    CUSTOM_VOICES[name] = meta
    logger.info("Custom voice cloned: %s", name)
    return {"name": name, **meta, "custom": True}


@app.delete("/api/voices/{name}")
def delete_voice(name: str):
    if name in VOICES:
        raise HTTPException(status_code=403, detail="Cannot delete built-in voices")
    if name not in CUSTOM_VOICES:
        raise HTTPException(status_code=404, detail=f"Custom voice '{name}' not found")
    path = os.path.join(CUSTOM_VOICES_DIR, f"{name}.wav")
    if os.path.exists(path):
        os.unlink(path)
    CUSTOM_VOICES.pop(name, None)
    logger.info("Custom voice deleted: %s", name)
    return {"deleted": name}


@app.post("/api/stt")
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
    apply_itn: bool = Form(True),
):
    data = await audio.read()
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        path = f.name
    try:
        recognizer = get_recognizer()
        samples, sample_rate = load_audio(path)
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        recognizer.decode_stream(stream)
        raw_text = stream.result.text.strip()

        text = raw_text
        itn_applied = False
        if apply_itn and language and language.lower() in ITN_SUPPORTED_LANGS:
            try:
                itn = get_itn(language.lower())
                text = itn.normalize(raw_text)
                itn_applied = True
            except Exception as e:
                logger.warning("ITN failed for lang=%s: %s", language, e)

        return {"text": text, "raw_text": raw_text, "itn_applied": itn_applied, "engine": "sherpa-onnx"}
    finally:
        os.unlink(path)


@app.post("/api/normalize")
async def normalize_text(
    text: str = Form(...),
    language: str = Form(...),
    mode: str = Form("itn"),
):
    lang = language.lower()
    if mode == "itn":
        if lang not in ITN_SUPPORTED_LANGS:
            raise HTTPException(status_code=400, detail=f"ITN not supported for '{language}'. Supported: {', '.join(sorted(ITN_SUPPORTED_LANGS))}")
        normalizer = get_itn(lang)
    elif mode == "tn":
        if lang not in TN_SUPPORTED_LANGS:
            raise HTTPException(status_code=400, detail=f"TN not supported for '{language}'. Supported: {', '.join(sorted(TN_SUPPORTED_LANGS))}")
        normalizer = get_tn(lang)
    else:
        raise HTTPException(status_code=400, detail="mode must be 'itn' or 'tn'")
    result = normalizer.normalize(text)
    return {"text": result, "original": text, "mode": mode, "language": lang}


@app.post("/api/tts")
@app.post("/tts")
async def synthesize(
    text: str = Form(...),
    voice: str | None = Form(None),
    language: str | None = Form(None),
    instruct: str | None = Form(None),
    ref_text: str | None = Form(None),
    ref_audio: UploadFile | None = File(None),
    format: str = Form("wav"),
    num_step: int = Form(32),
    guidance_scale: float = Form(2.0),
    t_shift: float = Form(0.1),
    denoise: bool = Form(True),
    speed: float = Form(1.0),
    duration: float | None = Form(None),
    preprocess_prompt: bool = Form(True),
    postprocess_output: bool = Form(True),
    layer_penalty_factor: float = Form(5.0),
    position_temperature: float = Form(5.0),
    class_temperature: float = Form(0.0),
    audio_chunk_duration: float = Form(15.0),
    audio_chunk_threshold: float = Form(30.0),
):
    if format not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="format must be 'wav' or 'mp3'")

    ref_path = None
    try:
        if ref_audio is not None:
            rsfx = os.path.splitext(ref_audio.filename or "")[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=rsfx) as f:
                f.write(await ref_audio.read())
                ref_path = f.name

        logger.info(
            "TTS request: text_len=%d, language=%s, voice=%s, format=%s",
            len(text), language, voice, format,
        )

        waveform, sr = _generate_audio(
            text=text,
            voice=voice,
            language=language,
            instruct=instruct,
            ref_text=ref_text,
            ref_audio_path=ref_path,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
            denoise=denoise,
            speed=speed,
            duration=duration,
            preprocess_prompt=preprocess_prompt,
            postprocess_output=postprocess_output,
            layer_penalty_factor=layer_penalty_factor,
            position_temperature=position_temperature,
            class_temperature=class_temperature,
            audio_chunk_duration=audio_chunk_duration,
            audio_chunk_threshold=audio_chunk_threshold,
        )

        wav_path = _save_wav(waveform, sr)
        cleanup = [wav_path]

        if format == "mp3":
            mp3_path = _wav_to_mp3(wav_path)
            cleanup.append(mp3_path)
            if ref_path and not _is_library_path(ref_path):
                cleanup.append(ref_path)
            logger.info("TTS done: duration=%.1fs, format=mp3", len(waveform) / sr)
            return FileResponse(
                mp3_path,
                media_type="audio/mpeg",
                filename="output.mp3",
                background=_cleanup_task(*cleanup),
            )

        if ref_path and not _is_library_path(ref_path):
            cleanup.append(ref_path)
        logger.info("TTS done: duration=%.1fs, format=wav", len(waveform) / sr)
        return FileResponse(
            wav_path,
            media_type="audio/wav",
            filename="output.wav",
            background=_cleanup_task(*cleanup),
        )
    except HTTPException:
        raise
    except Exception:
        if ref_path and not _is_library_path(ref_path) and os.path.exists(ref_path):
            os.unlink(ref_path)
        raise


class BatchItem(BaseModel):
    text: str
    voice: Optional[str] = None
    language: Optional[str] = None
    instruct: Optional[str] = None
    ref_text: Optional[str] = None


class BatchRequest(BaseModel):
    items: list[BatchItem]
    format: str = "wav"
    num_step: int = 32
    guidance_scale: float = 2.0
    speed: float = 1.0


@app.post("/api/tts/batch")
async def synthesize_batch(req: BatchRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="items list is empty")
    if len(req.items) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 items per batch")
    if req.format not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="format must be 'wav' or 'mp3'")

    buf = io.BytesIO()
    cleanup = []
    try:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, item in enumerate(req.items):
                waveform, sr = _generate_audio(
                    text=item.text,
                    voice=item.voice,
                    language=item.language,
                    instruct=item.instruct,
                    ref_text=item.ref_text,
                    num_step=req.num_step,
                    guidance_scale=req.guidance_scale,
                    speed=req.speed,
                )
                wav_path = _save_wav(waveform, sr)
                cleanup.append(wav_path)

                if req.format == "mp3":
                    mp3_path = _wav_to_mp3(wav_path)
                    cleanup.append(mp3_path)
                    zf.write(mp3_path, f"{i:03d}.mp3")
                else:
                    zf.write(wav_path, f"{i:03d}.wav")

        logger.info("Batch TTS done: %d items", len(req.items))
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=batch.zip"},
        )
    finally:
        for p in cleanup:
            if os.path.exists(p):
                os.unlink(p)


@app.post("/api/tts/async")
async def synthesize_async(
    text: str = Form(...),
    voice: str | None = Form(None),
    language: str | None = Form(None),
    instruct: str | None = Form(None),
    ref_text: str | None = Form(None),
    ref_audio: UploadFile | None = File(None),
    format: str = Form("wav"),
    num_step: int = Form(32),
    guidance_scale: float = Form(2.0),
    t_shift: float = Form(0.1),
    denoise: bool = Form(True),
    speed: float = Form(1.0),
    duration: float | None = Form(None),
    preprocess_prompt: bool = Form(True),
    postprocess_output: bool = Form(True),
):
    if format not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="format must be 'wav' or 'mp3'")

    ref_path = None
    if ref_audio is not None:
        rsfx = os.path.splitext(ref_audio.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=rsfx) as f:
            f.write(await ref_audio.read())
            ref_path = f.name

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _purge_expired_jobs()
        _jobs[job_id] = {"status": "pending", "created_at": time.time(), "file": None, "error": None}

    params = dict(
        text=text, voice=voice, language=language, instruct=instruct,
        ref_text=ref_text, ref_audio_path=ref_path,
        num_step=num_step, guidance_scale=guidance_scale, t_shift=t_shift,
        denoise=denoise, speed=speed, duration=duration,
        preprocess_prompt=preprocess_prompt, postprocess_output=postprocess_output,
    )

    def _run():
        try:
            with _jobs_lock:
                _jobs[job_id]["status"] = "processing"
            waveform, sr = _generate_audio(**params)
            wav_path = _save_wav(waveform, sr)
            out_path = wav_path
            if format == "mp3":
                out_path = _wav_to_mp3(wav_path)
                os.unlink(wav_path)
            with _jobs_lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["file"] = out_path
        except Exception as e:
            logger.exception("Async job %s failed", job_id)
            with _jobs_lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(e)
        finally:
            if ref_path and not _is_library_path(ref_path) and os.path.exists(ref_path):
                os.unlink(ref_path)

    threading.Thread(target=_run, daemon=True).start()
    logger.info("Async job created: %s", job_id)
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, request: Request):
    with _jobs_lock:
        _purge_expired_jobs()
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    result = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "done":
        result["download_url"] = str(request.url_for("job_download", job_id=job_id))
    if job["status"] == "failed":
        result["error"] = job["error"]
    return result


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}, not ready for download")
    path = job["file"]
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=410, detail="Output file has been cleaned up")
    ext = os.path.splitext(path)[1]
    media = "audio/mpeg" if ext == ".mp3" else "audio/wav"
    return FileResponse(path, media_type=media, filename=f"output{ext}")


@app.post("/api/sfx")
async def generate_sfx(
    prompt: str = Form(...),
    duration: float = Form(5.0),
    format: str = Form("wav"),
):
    if duration < 0.5 or duration > 30.0:
        raise HTTPException(status_code=400, detail="duration must be between 0.5 and 30 seconds")
    if format not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="format must be 'wav' or 'mp3'")

    model = get_sfx()
    model.set_generation_params(duration=duration)

    logger.info("SFX request: prompt=%r, duration=%.1fs", prompt, duration)
    wav = model.generate([prompt])
    audio_np = wav[0].squeeze(0).cpu().numpy()
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)

    wav_path = _save_wav(audio_int16, model.sample_rate)
    cleanup = [wav_path]

    if format == "mp3":
        mp3_path = _wav_to_mp3(wav_path)
        cleanup.append(mp3_path)
        logger.info("SFX done: duration=%.1fs, format=mp3", duration)
        return FileResponse(
            mp3_path,
            media_type="audio/mpeg",
            filename="sfx.mp3",
            background=_cleanup_task(*cleanup),
        )

    logger.info("SFX done: duration=%.1fs, format=wav", duration)
    return FileResponse(
        wav_path,
        media_type="audio/wav",
        filename="sfx.wav",
        background=_cleanup_task(*cleanup),
    )


class SfxBatchItem(BaseModel):
    prompt: str
    duration: float = 5.0


class SfxBatchRequest(BaseModel):
    items: list[SfxBatchItem]
    format: str = "wav"


@app.post("/api/sfx/batch")
async def generate_sfx_batch(req: SfxBatchRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="items list is empty")
    if len(req.items) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 items per batch")
    if req.format not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="format must be 'wav' or 'mp3'")

    model = get_sfx()
    max_dur = max(item.duration for item in req.items)
    model.set_generation_params(duration=max_dur)

    prompts = [item.prompt for item in req.items]
    logger.info("SFX batch: %d prompts, max_duration=%.1fs", len(prompts), max_dur)
    wavs = model.generate(prompts)

    buf = io.BytesIO()
    cleanup = []
    try:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, (item, wav_tensor) in enumerate(zip(req.items, wavs)):
                audio_np = wav_tensor.squeeze(0).cpu().numpy()
                target_samples = int(item.duration * model.sample_rate)
                audio_np = audio_np[:target_samples]
                audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)

                wav_path = _save_wav(audio_int16, model.sample_rate)
                cleanup.append(wav_path)

                if req.format == "mp3":
                    mp3_path = _wav_to_mp3(wav_path)
                    cleanup.append(mp3_path)
                    zf.write(mp3_path, f"{i:03d}.mp3")
                else:
                    zf.write(wav_path, f"{i:03d}.wav")

        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=sfx_batch.zip"},
        )
    finally:
        for p in cleanup:
            if os.path.exists(p):
                os.unlink(p)


_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")


@app.get("/")
def ui_root():
    return RedirectResponse(url="/ui/")


@app.on_event("startup")
def startup():
    _load_custom_voices()
    if os.environ.get("PRELOAD_STT", "").lower() in ("1", "true", "yes"):
        get_recognizer()
    if os.environ.get("PRELOAD_TTS", "").lower() in ("1", "true", "yes"):
        get_tts()
    if os.environ.get("PRELOAD_SFX", "").lower() in ("1", "true", "yes"):
        get_sfx()
