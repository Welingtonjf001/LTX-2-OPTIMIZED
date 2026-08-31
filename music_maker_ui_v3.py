import gradio as gr
import video_doctor_ui  # aba de diagnóstico temporal pós-geração
import subprocess
import os
import datetime
import threading
import sys
import math
import json
import hashlib
import warnings
import re
import shutil
import numpy as np
import torch
import torchaudio
import wave
import librosa
from collections import deque
import cv2
import audio_fx

# --- Configuration & Defaults ---
DEFAULT_CHECKPOINT = "./models/ltx-2.3-22b-distilled-fp8.safetensors"
DEFAULT_GEMMA = "./models/gemma3"
DEFAULT_UPSAMPLER = "./models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
AUDIO_CLIPS_DIR = "./audio_clips"
OUTPUT_ROOT = os.path.abspath(os.path.join("outputs", "music_video_v3"))
LATENTSYNC_ROOT = os.environ.get("LATENTSYNC_ROOT", r"E:\Users\home\Documents\LatentSync")
LATENTSYNC_PYTHON = os.environ.get(
    "LATENTSYNC_PYTHON", os.path.join(LATENTSYNC_ROOT, ".conda_env", "python.exe")
)

# --- Global State ---
JOB_QUEUE = deque()
QUEUE_LOCK = threading.Lock()
CURRENT_LOG = "System Ready. Waiting for jobs..."
def v2_status(message, phase=None, progress=None):
    global CURRENT_LOG, CURRENT_PHASE, CURRENT_PROGRESS
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}][V3] {message}"
    print(line, flush=True)
    CURRENT_LOG += line + "\n"
    if phase is not None: CURRENT_PHASE = phase
    if progress is not None: CURRENT_PROGRESS = max(0.0, min(100.0, float(progress)))
LATEST_VIDEO_PATH = None
IS_PROCESSING = False
STOP_GENERATION = False
CURRENT_JOB_ID = None
CURRENT_PROCESS = None
CURRENT_OUTPUT_PATH = None
CURRENT_SCENE_INDEX = -1
CURRENT_PHASE = "Idle"
CURRENT_PROGRESS = 0.0
CURRENT_TOTAL_SCENES = 0
LYRICS_ENABLED = True
LIPSYNC_ENABLED = False
MUSIC_AUDIO_PATH = None
CURRENT_RUN_DIR = None
# SCENES_DATA will store: {'prompt': str, 'video_path': str, 'audio_path': str, 'first_frame': str, 'last_frame': str}
SCENES_DATA = [None] * 20  # Increased to 20 scenes support
# Set by the speech/photo slicing worker threads when SCENES_DATA has fresh
# results ready to push into the scene editor. update_ui() reads it once,
# then clears it -- the scene-editor fields are otherwise left untouched on
# every other 2s tick, so a user's manual edit to a scene between slices is
# never clobbered by the live-progress timer.
SCENES_JUST_SLICED = False
CLIP_INTERROGATOR_INSTANCE = None
CLIP_INTERROGATOR_LOCK = threading.Lock()
PHOTO_CAPTION_CACHE_PATH = os.path.abspath("photo_clip_captions.json")

def run_subdir(name):
    base = CURRENT_RUN_DIR or os.getcwd()
    path = os.path.abspath(os.path.join(base, name))
    os.makedirs(path, exist_ok=True)
    return path

def vocal_cache_key(audio_path):
    """Stable cache key so copied inputs reuse the same Demucs result."""
    digest = hashlib.sha1()
    with open(audio_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(audio_path))[0]).strip("_") or "audio"
    return f"{stem}_{digest.hexdigest()[:16]}"

def apply_low_resolution_test_profile():
    """Return the smallest valid V3/LTX two-stage profile for a smoke test."""
    return 4, 24, 320, 128, 9

def create_generation_folder(prompts, audios, start_images):
    """Create an isolated folder and snapshot all inputs for one generation."""
    global CURRENT_RUN_DIR, MUSIC_AUDIO_PATH, CURRENT_LOG
    stem = os.path.splitext(os.path.basename(MUSIC_AUDIO_PATH or "music_video"))[0]
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "music_video"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(OUTPUT_ROOT, f"{stamp}_{safe_stem}")
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(OUTPUT_ROOT, f"{stamp}_{safe_stem}_{suffix}")
        suffix += 1
    CURRENT_RUN_DIR = os.path.abspath(candidate)
    for name in ("input", "audio", "scenes", "frames", "lipsync", "intermediate", "final", "logs"):
        os.makedirs(os.path.join(CURRENT_RUN_DIR, name), exist_ok=True)
    CURRENT_LOG = "System Ready. Waiting for jobs...\n"

    if MUSIC_AUDIO_PATH and os.path.exists(MUSIC_AUDIO_PATH):
        ext = os.path.splitext(MUSIC_AUDIO_PATH)[1] or ".wav"
        master_copy = os.path.join(CURRENT_RUN_DIR, "input", f"reference_music{ext}")
        shutil.copy2(MUSIC_AUDIO_PATH, master_copy)
        MUSIC_AUDIO_PATH = os.path.abspath(master_copy)

    copied_audios = list(audios)
    copied_images = list(start_images)
    for i in range(min(20, len(copied_audios))):
        audio_path = copied_audios[i]
        if audio_path and os.path.exists(audio_path):
            ext = os.path.splitext(audio_path)[1] or ".wav"
            target = os.path.join(CURRENT_RUN_DIR, "audio", f"scene_{i+1:02d}{ext}")
            shutil.copy2(audio_path, target)
            copied_audios[i] = os.path.abspath(target)
            if SCENES_DATA[i]: SCENES_DATA[i]["audio_path"] = copied_audios[i]
        image_path = copied_images[i]
        if isinstance(image_path, dict): image_path = image_path.get("path")
        if image_path and os.path.exists(image_path):
            ext = os.path.splitext(image_path)[1] or ".png"
            target = os.path.join(CURRENT_RUN_DIR, "input", f"scene_{i+1:02d}_reference{ext}")
            shutil.copy2(image_path, target)
            copied_images[i] = os.path.abspath(target)

    manifest = {
        "created_at": datetime.datetime.now().isoformat(),
        "run_directory": CURRENT_RUN_DIR,
        "reference_audio": MUSIC_AUDIO_PATH,
        "prompts": [p for p in prompts if p],
        "scene_count": sum(bool(p) for p in prompts),
    }
    with open(os.path.join(CURRENT_RUN_DIR, "generation_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    v2_status(f"nova pasta de geração: {CURRENT_RUN_DIR}", "Preparando pasta", 1)
    return copied_audios, copied_images

DEFAULT_SCENE_PROMPTS_EN = [
    "A consistent female singer with an acoustic guitar performs on a windswept mountain ridge, dramatic clouds, cinematic tracking shot, natural movement, realistic music video.",
    "The same singer and acoustic guitar perform in a vast golden desert, camels moving softly in the background, warm sunlight, blowing sand, cinematic wide shot.",
    "The same singer performs beside a quiet forest stream, tall trees and filtered morning light, subtle birds in the branches, gentle camera movement, realistic music video.",
    "The same singer and guitar appear in a lush rainforest filled with colorful wild birds, leaves moving in humid air, intimate handheld camera, rich natural detail.",
    "The same singer performs on a tropical beach during a rising tide, wind moving her clothing and hair, turquoise water, golden reflections, smooth dolly shot.",
    "The same singer plays acoustic guitar on a coral reef viewed through clear shallow water, sunlight rays beneath the surface, graceful underwater atmosphere, cinematic realism.",
    "The same singer performs on a rocky coastline under heavy storm clouds, waves crashing behind her, dramatic but consistent lighting, slow lateral camera movement.",
    "The same singer stands in a meadow during light rain, wildflowers bending in the breeze, soft gray sky, emotional performance, close cinematic portrait.",
    "The same singer performs in a snowy mountain valley, gentle snowfall, acoustic guitar held naturally, blue dusk light, slow crane camera, realistic textures.",
    "The same singer plays beneath a massive full moon in a quiet desert at night, stars across the sky, silver moonlight, wide cinematic composition, calm movement.",
    "The same singer performs beside a campfire under a star-filled sky, warm firelight on her face and guitar, dark forest behind her, intimate slow push-in.",
    "The same singer walks through a field of tall grass at sunset while playing guitar, orange sky, glowing dust, graceful tracking shot, coherent character continuity.",
    "The same singer performs on a windswept cliff at sunrise, ocean horizon behind her, bright golden rim light, expressive but natural gestures, cinematic realism.",
    "The same singer plays guitar in an ancient stone village at twilight, lanterns glowing, light mist in the streets, slow steady camera, atmospheric music video.",
    "The same singer performs beside a calm lake beneath dramatic violet clouds, mountains reflected in the water, gentle breeze, elegant wide shot, realistic motion.",
    "The same singer appears in a colorful canyon after rain, water glistening on red rock walls, soft sunlight breaking through clouds, smooth cinematic camera movement.",
    "The same singer performs among tall coastal cliffs with seabirds circling overhead, ocean spray in the air, cool daylight, natural acoustic performance, wide shot.",
    "The same singer plays guitar in a quiet winter forest beneath pale moonlight, frost on the branches, subtle breath in cold air, slow intimate camera movement.",
    "The same singer performs at dawn on a hill above the sea, pastel sky changing from blue to pink, gentle wind, hopeful expression, cinematic closing approach.",
    "The same singer finishes the performance on a peaceful overlook at sunrise, acoustic guitar in hand, vast landscapes behind her, warm light, memorable final music video shot."
]

# --- Logic Functions ---

def load_audio_compatible(audio_path):
    """Decode audio with FFmpeg first; use torchaudio only as a quiet fallback."""
    ffmpeg = os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
    target_sample_rate = 16000
    try:
        proc = subprocess.run(
            [ffmpeg, "-v", "error", "-i", audio_path, "-f", "f32le",
             "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(target_sample_rate), "pipe:1"],
            check=True, capture_output=True
        )
        samples = np.frombuffer(proc.stdout, dtype=np.float32)
        if samples.size == 0 or samples.size % 2:
            raise RuntimeError("FFmpeg returned no valid audio samples")
        waveform = torch.from_numpy(samples.reshape(-1, 2).T.copy())
        return waveform, target_sample_rate, waveform.shape[1] / target_sample_rate
    except Exception as ffmpeg_error:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info = torchaudio.info(audio_path)
                waveform, sample_rate = torchaudio.load(audio_path)
            return waveform, sample_rate, info.num_frames / info.sample_rate
        except Exception as audio_error:
            raise RuntimeError(f"Could not decode audio with FFmpeg or torchaudio: {ffmpeg_error}; {audio_error}")

def save_audio_compatible(audio_path, waveform, sample_rate):
    """Save a torch waveform as PCM WAV without requiring a torchaudio backend."""
    data = waveform.detach().cpu().float().clamp(-1.0, 1.0).numpy()
    if data.ndim == 1:
        data = data[None, :]
    interleaved = (data.T.reshape(-1) * 32767.0).astype(np.int16).tobytes()
    with wave.open(audio_path, "wb") as wav_file:
        wav_file.setnchannels(data.shape[0])
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(interleaved)

def beat_aligned_boundaries(waveform, sample_rate, duration_sec, target_scene_sec, num_scenes):
    """Choose scene boundaries near downbeats while preserving full-track coverage."""
    mono = waveform.float().mean(dim=0).numpy()
    _, beat_frames = librosa.beat.beat_track(
        y=mono, sr=sample_rate, hop_length=512, units="frames", trim=False
    )
    beats = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=512)
    downbeats = [float(t) for t in beats[::4] if 2.0 < t < duration_sec - 2.0]
    boundaries = [0.0]
    for index in range(1, num_scenes):
        desired = index * target_scene_sec
        candidates = [t for t in downbeats if boundaries[-1] + 2.0 <= t <= duration_sec - 2.0]
        if candidates:
            chosen = min(candidates, key=lambda t: abs(t - desired))
            if chosen > boundaries[-1] + 2.0:
                boundaries.append(chosen)
                continue
        boundaries.append(desired)
    boundaries.append(float(duration_sec))
    return boundaries, len(downbeats)

def vocal_aware_boundaries(lyrics_segments, duration_sec, target_scene_sec, num_scenes,
                           min_scene_sec=5.0, max_scene_sec=10.0):
    """Choose 5-10 s boundaries on ASR word/phrase ends, never mid-word when possible."""
    if not lyrics_segments:
        return None, 0
    phrase_ends = []
    word_ends = []
    for segment in lyrics_segments:
        try:
            seg_end = float(segment.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if 0.0 < seg_end < duration_sec:
            text = str(segment.get("text", "")).strip()
            punctuation = 1 if text.endswith((".", "!", "?", ";", ":")) else 0
            phrase_ends.append((seg_end, punctuation))
        for word in segment.get("words") or []:
            try:
                word_end = float(word.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            if 0.0 < word_end < duration_sec:
                word_ends.append(word_end)
    if not phrase_ends and not word_ends:
        return None, 0
    phrase_ends = sorted(set((round(t, 3), p) for t, p in phrase_ends))
    word_ends = sorted(set(round(t, 3) for t in word_ends))
    boundaries = [0.0]
    used = 0
    target = min(max(float(target_scene_sec), min_scene_sec), max_scene_sec)
    for index in range(1, num_scenes):
        previous = boundaries[-1]
        remaining_scenes = num_scenes - index
        lower = max(previous + min_scene_sec, duration_sec - remaining_scenes * max_scene_sec)
        upper = min(previous + max_scene_sec, duration_sec - remaining_scenes * min_scene_sec)
        if upper < lower:
            upper = min(previous + max_scene_sec, duration_sec)
        desired = min(max(previous + target, lower), upper)
        phrase_candidates = [(t, punctuation) for t, punctuation in phrase_ends if lower <= t <= upper]
        if phrase_candidates:
            chosen = min(phrase_candidates, key=lambda item: (abs(item[0] - desired), -item[1]))[0]
            used += 1
        else:
            word_candidates = [t for t in word_ends if lower <= t <= upper]
            if word_candidates:
                chosen = min(word_candidates, key=lambda t: abs(t - desired))
                used += 1
            else:
                chosen = upper
        boundaries.append(round(float(chosen), 3))
    boundaries.append(float(duration_sec))
    return boundaries, used

def load_lyrics_segments(audio_path=None, duration_sec=None):
    """Load timestamps only when their metadata belongs to the selected audio."""
    audio_abs = os.path.normcase(os.path.normpath(os.path.abspath(audio_path))) if audio_path else None
    stem = os.path.splitext(os.path.basename(audio_path))[0] if audio_path else ""
    # transcribe_lyrics_for_audio saves sidecars under a filesystem-safe stem.
    # Gradio upload names commonly contain spaces and parentheses, so looking up
    # only the original stem made a valid ASR result appear as "no speech".
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem).strip("_") or "audio"
    candidates = [
        os.path.join(os.getcwd(), f"lyrics_timing_{stem}.json") if stem else None,
        os.path.join(os.getcwd(), f"lyrics_timing_{safe_stem}.json") if stem else None,
        os.path.join(os.getcwd(), "lyrics_timing.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f"lyrics_timing_{stem}.json") if stem else None,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f"lyrics_timing_{safe_stem}.json") if stem else None,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "lyrics_timing.json"),
    ]
    candidates = [path for path in candidates if path]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            recorded_audio = payload.get("audio") or payload.get("audio_path")
            recorded_duration = payload.get("duration_sec")
            if not recorded_audio:
                print(f"Ignoring legacy lyrics file without audio metadata: {path}")
                continue
            recorded_abs = os.path.normcase(os.path.normpath(os.path.abspath(str(recorded_audio))))
            same_file = recorded_abs == audio_abs
            same_duration = (
                duration_sec is not None and recorded_duration is not None and
                abs(float(recorded_duration) - float(duration_sec)) <= 0.75
            )
            if not same_file and not (same_duration and os.path.basename(recorded_abs) == os.path.basename(audio_abs)):
                print(f"Ignoring stale lyrics file for another audio: {path}")
                continue
            return payload.get("segments", []), path, None
        except Exception as exc:
            print(f"Lyrics timing read failed ({path}): {exc}")
    return [], None, "No matching lyrics_timing.json for the selected audio."

def lyrics_for_window(segments, start_sec, end_sec):
    """Return ASR text overlapping a scene window, capped for prompt safety."""
    text_parts = []
    for segment in segments:
        try:
            seg_start = float(segment.get("start", 0.0))
            seg_end = float(segment.get("end", seg_start))
        except (TypeError, ValueError):
            continue
        if seg_end > start_sec and seg_start < end_sec:
            text = str(segment.get("text", "")).strip()
            if text:
                text_parts.append(text)
    return " ".join(text_parts)[:500]

def transcribe_lyrics_for_audio(audio_path, source_audio=None):
    """Create a per-audio Whisper sidecar, falling back to Demucs vocals."""
    global CURRENT_LOG, CURRENT_PHASE, CURRENT_PROGRESS
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem).strip("_") or "audio"
    output_path = os.path.abspath(os.path.join(os.getcwd(), f"lyrics_timing_{safe_stem}.json"))
    transcriber = os.path.abspath(os.path.join(os.getcwd(), "transcribe_vocals.py"))
    if not os.path.exists(transcriber):
        return None, "transcribe_vocals.py is missing."
    CURRENT_PHASE = "Extracting lyrics with ASR"
    CURRENT_PROGRESS = 1.0
    CURRENT_LOG += f"No matching lyrics found; transcribing {os.path.basename(audio_path)}...\n"
    source_audio = os.path.abspath(source_audio or audio_path)
    command = [sys.executable, transcriber, os.path.abspath(audio_path), "--output", output_path, "--model", "base"]
    if source_audio != os.path.abspath(audio_path):
        command.extend(["--source-audio", source_audio])
    try:
        result = subprocess.run(command, cwd=os.getcwd(), capture_output=True, text=True)
    except Exception as exc:
        return None, f"ASR launch failed: {exc}"
    if result.stdout:
        CURRENT_LOG += result.stdout[-2000:] + "\n"
    direct_ok = result.returncode == 0 and os.path.exists(output_path)
    direct_segments = []
    if direct_ok:
        try:
            with open(output_path, "r", encoding="utf-8") as handle:
                direct_segments = json.load(handle).get("segments", [])
        except Exception:
            direct_segments = []
    if direct_segments:
        CURRENT_LOG += f"Lyrics sidecar created: {output_path}\n"
        return output_path, None

    # Mixed music can hide vocals from VAD. Separate the vocal stem and retry ASR.
    CURRENT_PHASE = "Separating vocals for ASR"
    CURRENT_PROGRESS = 2.0
    CURRENT_LOG += "Direct ASR found no speech; running Demucs vocal separation...\n"
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    vocal_path = os.path.abspath(os.path.join(os.getcwd(), "stems", "htdemucs", stem, "vocals.wav"))
    demucs_command = [
        sys.executable, "-m", "demucs.separate", "-n", "htdemucs",
        "--two-stems=vocals", "-d", "cuda", "-o", os.path.abspath("stems"), source_audio,
    ]
    try:
        demucs_result = subprocess.run(demucs_command, cwd=os.getcwd(), capture_output=True, text=True)
        if demucs_result.stdout:
            CURRENT_LOG += demucs_result.stdout[-2000:] + "\n"
    except Exception as exc:
        demucs_result = None
        CURRENT_LOG += f"Demucs launch failed: {exc}\n"
    if demucs_result is None or demucs_result.returncode != 0 or not os.path.exists(vocal_path):
        warning = (getattr(demucs_result, "stderr", "") or "No vocal stem was produced.")[-1000:]
        CURRENT_LOG += f"Automatic vocal extraction failed: {warning}\n"
        return None, "ASR found no speech and vocal separation failed."

    CURRENT_PHASE = "Transcribing separated vocals"
    CURRENT_PROGRESS = 3.0
    vocal_command = [
        sys.executable, transcriber, vocal_path, "--source-audio", source_audio,
        "--output", output_path, "--model", "base",
    ]
    try:
        vocal_result = subprocess.run(vocal_command, cwd=os.getcwd(), capture_output=True, text=True)
    except Exception as exc:
        return None, f"Vocal ASR launch failed: {exc}"
    if vocal_result.stdout:
        CURRENT_LOG += vocal_result.stdout[-2000:] + "\n"
    if vocal_result.returncode == 0 and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as handle:
                vocal_segments = json.load(handle).get("segments", [])
        except Exception:
            vocal_segments = []
        if vocal_segments:
            CURRENT_LOG += f"Lyrics sidecar created from vocals: {output_path}\n"
            return output_path, None
    warning = (vocal_result.stderr or vocal_result.stdout or "Vocal ASR found no speech.")[-1000:]
    CURRENT_LOG += f"Automatic lyrics extraction failed: {warning}\n"
    return None, "No speech was recognized after vocal separation."

def run_syncnet_evaluation(video_path):
    """Measure SyncNet confidence and A/V offset without blocking video completion on failure."""
    global CURRENT_LOG
    helper = os.path.abspath("latentsync_syncnet_eval.py")
    model = os.path.join(LATENTSYNC_ROOT, "checkpoints", "auxiliary", "syncnet_v2.model")
    face_model = os.path.join(LATENTSYNC_ROOT, "checkpoints", "auxiliary", "sfd_face.pth")
    missing = [path for path in (helper, model, face_model) if not os.path.exists(path)]
    if missing:
        CURRENT_LOG += "SyncNet evaluation unavailable: " + ", ".join(missing) + "\n"
        return None

    scene_name = os.path.splitext(os.path.basename(video_path))[0]
    detect_dir = os.path.abspath(os.path.join(run_subdir("intermediate"), f"syncnet_{scene_name}_detect"))
    temp_dir = os.path.abspath(os.path.join(run_subdir("intermediate"), f"syncnet_{scene_name}_temp"))
    command = [
        LATENTSYNC_PYTHON, "-u", helper,
        "--latentsync-root", LATENTSYNC_ROOT,
        "--video-path", video_path,
        "--model-path", model,
        "--detect-dir", detect_dir,
        "--temp-dir", temp_dir,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("LATENTSYNC_CUDA_DEVICE", "1")
    env["PYTHONUNBUFFERED"] = "1"
    v2_status(f"SyncNet avaliando {os.path.basename(video_path)}", "Avaliação de sincronismo")
    try:
        result = subprocess.run(
            command,
            cwd=LATENTSYNC_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:
        CURRENT_LOG += f"SyncNet evaluation failed: {exc}\n"
        return None

    evaluation_log = (result.stdout or "") + (result.stderr or "")
    CURRENT_LOG += evaluation_log
    marker = "SYNCNET_RESULT_JSON="
    for line in reversed((result.stdout or "").splitlines()):
        if line.startswith(marker):
            try:
                score = json.loads(line[len(marker):])
                confidence = float(score["confidence"])
                av_offset = int(score["av_offset"])
                message = f"SyncNet: confiança {confidence:.2f}; deslocamento A/V {av_offset} frame(s)"
                CURRENT_LOG += message + "\n"
                v2_status(message, "Avaliação de sincronismo")
                return score
            except Exception as exc:
                CURRENT_LOG += f"Could not parse SyncNet result: {exc}\n"
                return None
    CURRENT_LOG += "SyncNet did not produce a score; video processing will continue.\n"
    return None


def run_latentsync(video_path, audio_path, output_path, inference_steps=20, guidance_scale=1.5):
    """Run LatentSync 1.6, automatically partitioning long scenes into 5-10 s pieces."""
    global CURRENT_LOG, CURRENT_PROCESS
    checkpoint = os.path.join(LATENTSYNC_ROOT, "checkpoints", "latentsync_unet.pt")
    whisper = os.path.join(LATENTSYNC_ROOT, "checkpoints", "whisper", "tiny.pt")
    required = [LATENTSYNC_PYTHON, checkpoint, whisper]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        v2_status("LatentSync indisponível; usando fallback Wav2Lip", "Fallback de lip-sync")
        CURRENT_LOG += "LatentSync missing: " + ", ".join(missing) + "\n"
        return None

    ffmpeg = os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
    normalized_video = os.path.join(run_subdir("intermediate"), f"latentsync_{os.path.basename(video_path)}")
    normalized_audio = os.path.join(
        run_subdir("audio"), f"latentsync_{os.path.splitext(os.path.basename(audio_path))[0]}.wav"
    )
    video_prep = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", video_path, "-an", "-vf", "fps=25",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", normalized_video],
        capture_output=True, text=True,
    )
    audio_prep = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", audio_path, "-ar", "16000", "-ac", "1", normalized_audio],
        capture_output=True, text=True,
    )
    if video_prep.returncode != 0 or audio_prep.returncode != 0:
        CURRENT_LOG += "LatentSync preprocessing failed; using Wav2Lip fallback.\n"
        return None

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("LATENTSYNC_CUDA_DEVICE", "1")
    env["PYTHONUNBUFFERED"] = "1"

    ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
    try:
        duration_probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", normalized_video],
            capture_output=True, text=True, timeout=30,
        )
        duration = float((duration_probe.stdout or "0").strip())
    except Exception:
        duration = 0.0

    # ceil(duration / 10) makes every piece 5-10 s whenever the source is >10 s.
    part_count = max(1, int(math.ceil(duration / 10.0))) if duration > 0 else 1
    part_duration = duration / part_count if part_count > 1 else duration
    if part_count > 1:
        v2_status(
            f"LatentSync: cena de {duration:.1f}s dividida em {part_count} partes de {part_duration:.1f}s",
            "Segmentação 5-10 s para LatentSync",
        )

    synced_parts = []
    scene_stem = os.path.splitext(os.path.basename(output_path))[0]
    for part_index in range(part_count):
        if STOP_GENERATION:
            return None
        if part_count == 1:
            part_video = normalized_video
            part_audio = normalized_audio
            part_output = output_path
        else:
            start = part_index * part_duration
            length = part_duration if part_index < part_count - 1 else max(0.1, duration - start)
            part_tag = f"{scene_stem}_part_{part_index + 1:02d}"
            part_video = os.path.abspath(os.path.join(run_subdir("intermediate"), f"{part_tag}_source.mp4"))
            part_audio = os.path.abspath(os.path.join(run_subdir("audio"), f"{part_tag}_audio.wav"))
            part_output = os.path.abspath(os.path.join(run_subdir("intermediate"), f"{part_tag}_synced.mp4"))
            cut_video = subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-ss", f"{start:.6f}", "-t", f"{length:.6f}",
                 "-i", normalized_video, "-an", "-c:v", "libx264", "-crf", "18",
                 "-pix_fmt", "yuv420p", part_video],
                capture_output=True, text=True,
            )
            cut_audio = subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-ss", f"{start:.6f}", "-t", f"{length:.6f}",
                 "-i", normalized_audio, "-ar", "16000", "-ac", "1", part_audio],
                capture_output=True, text=True,
            )
            if cut_video.returncode != 0 or cut_audio.returncode != 0:
                CURRENT_LOG += f"LatentSync segment preparation failed at part {part_index + 1}.\n"
                return None

        command = [
            LATENTSYNC_PYTHON, "-u", "-m", "scripts.inference",
            "--unet_config_path", "configs/unet/stage2_512.yaml",
            "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
            "--inference_steps", str(int(inference_steps)),
            "--guidance_scale", str(float(guidance_scale)),
            "--enable_deepcache",
            "--video_path", part_video,
            "--audio_path", part_audio,
            "--video_out_path", part_output,
        ]
        v2_status(
            f"LatentSync 1.6: {os.path.basename(video_path)} — parte {part_index + 1}/{part_count}",
            "LatentSync por cena",
        )
        CURRENT_PROCESS = subprocess.Popen(
            command, cwd=LATENTSYNC_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True,
        )
        for line in CURRENT_PROCESS.stdout:
            CURRENT_LOG += line
            print(f"[LatentSync] {line}", end="", flush=True)
            if STOP_GENERATION:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(CURRENT_PROCESS.pid)], capture_output=True)
                break
        CURRENT_PROCESS.wait()
        if CURRENT_PROCESS.returncode != 0 or not os.path.exists(part_output):
            CURRENT_LOG += f"LatentSync failed at part {part_index + 1}; using Wav2Lip fallback.\n"
            return None
        synced_parts.append(part_output)

    if len(synced_parts) > 1:
        part_list = os.path.abspath(os.path.join(run_subdir("intermediate"), f"{scene_stem}_parts.txt"))
        with open(part_list, "w", encoding="ascii") as list_file:
            for part_path in synced_parts:
                list_file.write(f"file '{part_path.replace(chr(92), '/')}'\n")
        join_result = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", part_list,
             "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", output_path],
            capture_output=True, text=True,
        )
        if join_result.returncode != 0 or not os.path.exists(output_path):
            CURRENT_LOG += "LatentSync part concatenation failed; using Wav2Lip fallback.\n"
            return None

    if os.path.exists(output_path):
        CURRENT_LOG += f"LatentSync complete: {output_path}\n"
        run_syncnet_evaluation(output_path)
        return output_path
    CURRENT_LOG += "LatentSync produced no output; using Wav2Lip fallback.\n"
    return None

def run_wav2lip(video_path, audio_path, output_path):
    """Run the optional Wav2Lip stage; return output_path only on success."""
    global CURRENT_LOG
    repo = os.path.abspath(os.path.join("tools", "Wav2Lip"))
    # Wav2Lip writes temp/result.avi relative to its repository. Create it
    # explicitly so a clean installation does not fail inside OpenCV's writer.
    os.makedirs(os.path.join(repo, "temp"), exist_ok=True)
    inference = os.path.join(repo, "inference.py")
    checkpoint_path = os.path.join(repo, "checkpoints", "wav2lip_gan.pth")
    face_checkpoint = os.path.join(repo, "face_detection", "detection", "sfd", "s3fd.pth")
    missing = [p for p in (inference, checkpoint_path, face_checkpoint) if not os.path.exists(p)]
    if missing:
        CURRENT_LOG += "Lip-sync requested but checkpoints are missing: " + ", ".join(missing) + "\n"
        return None
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    command = [
        sys.executable, inference,
        "--checkpoint_path", checkpoint_path,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_path,
        "--pads", "0", "10", "0", "0",
    ]
    v2_status(f"Wav2Lip: {os.path.basename(video_path)}", "Lip-sync por cena")
    result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True)
    if result.stdout:
        CURRENT_LOG += result.stdout[-4000:] + "\n"
        print(result.stdout[-4000:], flush=True)
    if result.returncode != 0 or not os.path.exists(output_path):
        CURRENT_LOG += "Wav2Lip failed; continuing without lip-sync.\n"
        if result.stderr:
            CURRENT_LOG += result.stderr[-4000:] + "\n"
        return None
    CURRENT_LOG += f"Lip-sync complete: {output_path}\n"
    return output_path

def get_or_create_vocal_stem(audio_path):
    """Return a cached Demucs vocal stem for ASR/Wav2Lip, or None on failure."""
    global CURRENT_LOG
    if not audio_path or not os.path.exists(audio_path):
        return None
    stem_name = vocal_cache_key(audio_path)
    demucs_root = os.path.abspath("stems")
    stem_dir = os.path.abspath(os.path.join(demucs_root, "htdemucs", stem_name))
    vocal_path = os.path.join(stem_dir, "vocals.wav")
    instrumental_path = os.path.join(stem_dir, "no_vocals.wav")
    cache_input = os.path.join(demucs_root, "_inputs", stem_name + ".wav")
    if os.path.exists(vocal_path) and os.path.exists(instrumental_path):
        v2_status("usando vocal Demucs em cache", "Preparando vocal", 85)
        return vocal_path
    os.makedirs(os.path.dirname(cache_input), exist_ok=True)
    if not os.path.exists(cache_input):
        shutil.copy2(audio_path, cache_input)
    for device in ("cuda", "cpu"):
        v2_status(
            "Demucs: isolando vocal para ASR/Wav2Lip" if device == "cuda" else
            "Demucs CUDA indisponível; repetindo na CPU",
            "Preparando vocal", 85,
        )
        command = [
            sys.executable, "-m", "demucs.separate", "-n", "htdemucs",
            "--two-stems=vocals", "-d", device, "-j", "1", "-o", demucs_root,
            cache_input,
        ]
        try:
            result = subprocess.run(command, cwd=os.getcwd(), capture_output=True, text=True)
        except Exception as exc:
            CURRENT_LOG += f"Demucs ({device}) não iniciou: {exc}\n"
            continue
        terminal_text = (result.stdout or "") + (result.stderr or "")
        if terminal_text:
            CURRENT_LOG += terminal_text[-5000:] + "\n"
            print(terminal_text[-5000:], flush=True)
        if result.returncode == 0 and os.path.exists(vocal_path):
            v2_status(f"Demucs concluído ({device}); vocal em cache para reutilização", "Preparando vocal", 86)
            return vocal_path
    CURRENT_LOG += "Demucs falhou em CUDA e CPU.\n"
    return None

def get_instrumental_stem_path(audio_path):
    """Path to the no_vocals.wav Demucs already writes alongside vocals.wav
    (see get_or_create_vocal_stem). Used to remix a reverb-processed vocal
    back with the untouched instrumental bed for room/environment spatialization."""
    if not audio_path or not os.path.exists(audio_path):
        return None
    stem_name = vocal_cache_key(audio_path)
    instrumental_path = os.path.abspath(os.path.join("stems", "htdemucs", stem_name, "no_vocals.wav"))
    return instrumental_path if os.path.exists(instrumental_path) else None

def slice_lipsync_audio(source_audio, scene_data, scene_number):
    """Cut the exact scene interval from the vocal stem for Wav2Lip."""
    ffmpeg = os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
    start_sec = max(0.0, float(scene_data.get("start_sec", 0.0)))
    duration = max(0.08, float(scene_data.get("end_sec", 0.0)) - start_sec)
    output = os.path.abspath(os.path.join(run_subdir("audio"), f"scene_{scene_number:02d}_vocal_lipsync.wav"))
    command = [
        ffmpeg, "-y", "-v", "error", "-ss", f"{start_sec:.6f}", "-t", f"{duration:.6f}",
        "-i", source_audio, "-ar", "16000", "-ac", "1", output,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return output if result.returncode == 0 and os.path.exists(output) else scene_data.get("audio_path")

def extract_frame(video_path, output_image_path, frame_idx=0):
    """Extracts a specific frame by index using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle negative indices (like -1 for last)
    if frame_idx < 0:
        frame_idx = frame_count + frame_idx
    
    # Bounds check
    if frame_idx >= frame_count or frame_idx < 0:
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if success:
        cv2.imwrite(output_image_path, frame)
        cap.release()
        return True
    cap.release()
    return False

def extract_first_frame(video_path, output_image_path):
    return extract_frame(video_path, output_image_path, 0)

def extract_last_frame(video_path, output_image_path):
    return extract_frame(video_path, output_image_path, -1)

def slice_audio(audio_path, prompt, fps, num_frames, lyrics_enabled=True):
    global SCENES_DATA, CURRENT_PHASE, CURRENT_PROGRESS, CURRENT_TOTAL_SCENES, LYRICS_ENABLED, MUSIC_AUDIO_PATH
    if not audio_path:
        return "Please upload an audio file.", []
    
    try:
        os.makedirs(AUDIO_CLIPS_DIR, exist_ok=True)
        # Load audio, with FFmpeg fallback for MP3/WAV on this Windows environment.
        waveform, sr, duration_sec = load_audio_compatible(audio_path)
        LYRICS_ENABLED = bool(lyrics_enabled)
        MUSIC_AUDIO_PATH = os.path.abspath(audio_path)
        lyrics_segments, lyrics_path, lyrics_warning = load_lyrics_segments(audio_path, duration_sec) if LYRICS_ENABLED else ([], None, None)
        vocal_analysis_path = None
        if LYRICS_ENABLED and (not lyrics_path or not lyrics_segments):
            try:
                vocal_analysis_path = get_or_create_vocal_stem(audio_path)
            except Exception as exc:
                CURRENT_LOG += f"Vocal isolation for timing failed: {exc}\n"
            generated_path, asr_warning = transcribe_lyrics_for_audio(
                vocal_analysis_path or audio_path, source_audio=audio_path
            )
            if generated_path:
                lyrics_segments, lyrics_path, lyrics_warning = load_lyrics_segments(audio_path, duration_sec)
                if not lyrics_segments:
                    lyrics_warning = "ASR completed but found no speech; prompts use visual style only."
            else:
                lyrics_warning = asr_warning or lyrics_warning
        sample_rate = sr
        total_frames = waveform.shape[1]
        
        # Calculate video scene duration
        scene_duration_sec = num_frames / fps
        
        target_scene_sec = min(max(scene_duration_sec, 5.0), 10.0)
        num_scenes = math.ceil(duration_sec / target_scene_sec)
        num_scenes = min(num_scenes, 20) # Limit to 20 scenes
        CURRENT_TOTAL_SCENES = num_scenes
        CURRENT_PHASE = f"Audio segmented into {num_scenes} scenes"
        CURRENT_PROGRESS = 0.0
        
        new_scenes_data = [None] * 20
        ui_updates = []
        
        boundaries, vocal_boundary_count = vocal_aware_boundaries(
            lyrics_segments, duration_sec, target_scene_sec, num_scenes
        )
        if boundaries is None:
            boundaries, downbeat_count = beat_aligned_boundaries(
                waveform, sr, duration_sec, target_scene_sec, num_scenes
            )
            boundary_note = f"BPM fallback: {downbeat_count} downbeats"
        else:
            boundary_note = f"vocal/ASR: {vocal_boundary_count} limites em fim de palavra/frase"

        for i in range(num_scenes):
            start_sample = int(boundaries[i] * sr)
            end_sample = min(int(boundaries[i + 1] * sr), total_frames)
            
            chunk_waveform = waveform[:, start_sample:end_sample]
            
            # Save chunk
            chunk_filename = f"scene_{i+1}_audio.wav"
            chunk_path = os.path.join(AUDIO_CLIPS_DIR, chunk_filename)
            save_audio_compatible(chunk_path, chunk_waveform, sr)
            
            scene_prompt = DEFAULT_SCENE_PROMPTS_EN[i]
            if prompt and prompt.strip():
                scene_prompt = f"{scene_prompt} Overall visual style: {prompt.strip()}"
            lyric_text = lyrics_for_window(lyrics_segments, boundaries[i], boundaries[i + 1])
            if LYRICS_ENABLED and lyric_text:
                scene_prompt = f'{scene_prompt} Lyric context for this performance: "{lyric_text}".'

            new_scenes_data[i] = {
                'prompt': scene_prompt,
                'video_path': None,
                'audio_path': os.path.abspath(chunk_path),
                'duration_sec': max(0.1, boundaries[i + 1] - boundaries[i]),
                'start_sec': float(boundaries[i]),
                'end_sec': float(boundaries[i + 1]),
                'lyrics_text': lyric_text,
                'first_frame': None,
                'last_frame': None
            }
            
        SCENES_DATA = new_scenes_data
        
        # Prepare UI updates
        for i in range(20):
            if i < num_scenes:
                # Row visible, Textbox updated
                ui_updates.append(gr.update(visible=True)) # Row
                ui_updates.append(gr.update(value=new_scenes_data[i]['prompt'], visible=True)) # Textbox
                ui_updates.append(gr.update(value=new_scenes_data[i]['audio_path'])) # Audio path display
                ui_updates.append(gr.update(value=None)) # Start Image
                ui_updates.append(gr.update(value=None)) # Last Image Override
            else:
                ui_updates.append(gr.update(visible=False)) # Row
                ui_updates.append(gr.update(visible=False)) # Textbox
                ui_updates.append(gr.update(value=None)) # Audio
                ui_updates.append(gr.update(value=None)) # Start Image
                ui_updates.append(gr.update(value=None)) # Last Image Override
        
        lyrics_note = " Lyrics loaded for this audio." if LYRICS_ENABLED and lyrics_path else (" Lyrics disabled." if not LYRICS_ENABLED else f" {lyrics_warning or 'No matching lyrics_timing.json; prompts use visual style only.'}")
        return tuple([f"Sliced into {num_scenes} cenas de 5–10 s ({boundary_note}).{lyrics_note}"] + ui_updates)
        
    except Exception as e:
        print(f"DEBUG Error in slice_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return tuple([f"Error: {str(e)}"] + [gr.update(visible=False)] * 20 * 5) # Update this count if UI structure changes

def slice_speech(audio_path, image_path, prompt, fps, num_frames):
    """Validate fast, then transcribe/slice speech in a background thread.

    Transcription (Whisper ASR, possibly preceded by Demucs vocal separation)
    can run for a long time with zero visible feedback if it blocks the
    Gradio click callback -- the Timer(2)/Worker Log below only updates
    between callbacks, and a synchronous click handler holds that slot for
    its entire duration. Returning immediately and doing the real work on a
    daemon thread lets v2_status()'s writes to CURRENT_LOG reach the browser
    live, the same way the main generation queue already does.
    """
    if not audio_path:
        return "Upload a speech audio file first."
    if not image_path:
        return "Upload a base speaker image first."
    threading.Thread(target=_slice_speech_worker, args=(audio_path, image_path, prompt, fps, num_frames), daemon=True).start()
    return "Transcrevendo e fatiando a fala... acompanhe o progresso no Worker Log abaixo."


def _slice_speech_worker(audio_path, image_path, prompt, fps, num_frames):
    """Background body of slice_speech; runs off the Gradio callback thread."""
    global SCENES_DATA, SCENES_JUST_SLICED, CURRENT_TOTAL_SCENES, MUSIC_AUDIO_PATH
    try:
        os.makedirs(AUDIO_CLIPS_DIR, exist_ok=True)
        waveform, sr, duration_sec = load_audio_compatible(audio_path)
        MUSIC_AUDIO_PATH = os.path.abspath(audio_path)
        v2_status("Transcrevendo fala (Whisper ASR)...", "Preparing speech transcription", 0.5)
        segments, timing_path, warning = load_lyrics_segments(audio_path, duration_sec)
        if not timing_path or not segments:
            _, asr_warning = transcribe_lyrics_for_audio(audio_path)
            segments, timing_path, warning = load_lyrics_segments(audio_path, duration_sec)
            warning = asr_warning if not timing_path else warning
        speech_segments = []
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            try:
                start = max(0.0, float(segment.get("start", 0.0)))
                end = min(duration_sec, float(segment.get("end", start)))
            except (TypeError, ValueError):
                continue
            if text and end > start:
                speech_segments.append((start, end, text))
        if not speech_segments:
            v2_status(warning or "No speech was detected in the audio.", "Error", 0.0)
            return

        # Group Whisper segments into video-sized dialogue blocks while preserving all audio.
        target_sec = max(1.0, float(num_frames) / float(fps))
        groups = []
        group_start = 0.0
        group_end = 0.0
        group_text = []
        for start, end, text in speech_segments:
            if group_text and end - group_start >= target_sec:
                groups.append((group_start, group_end, " ".join(group_text)))
                group_start = start
                group_text = []
            group_end = end
            group_text.append(text)
        if group_text:
            groups.append((group_start, group_end, " ".join(group_text)))
        if groups[0][0] > 0.25:
            groups[0] = (0.0, groups[0][1], groups[0][2])
        if groups[-1][1] < duration_sec - 0.1:
            start, end, text = groups[-1]
            groups[-1] = (start, duration_sec, text)
        if len(groups) > 20:
            # Merge the tail rather than silently dropping spoken content.
            groups = groups[:19] + [(groups[19][0], groups[-1][1], " ".join(g[2] for g in groups[19:]))]

        CURRENT_TOTAL_SCENES = len(groups)
        v2_status(f"Speech sliced into {len(groups)} dialogue scenes", "Slicing speech", 4.0)
        new_scenes_data = [None] * 20
        base_prompt = (prompt or "A natural talking-head speech performance").strip()
        for i, (start_sec, end_sec, speech_text) in enumerate(groups):
            start_sample = int(start_sec * sr)
            end_sample = min(int(end_sec * sr), waveform.shape[1])
            chunk_path = os.path.abspath(os.path.join(AUDIO_CLIPS_DIR, f"speech_scene_{i+1}_audio.wav"))
            save_audio_compatible(chunk_path, waveform[:, start_sample:end_sample], sr)
            scene_prompt = (
                f"{base_prompt}. The same speaker from the reference image is talking directly to camera, "
                "natural mouth articulation, realistic facial expressions, stable identity, subtle head movement, "
                f"clear spoken dialogue: \"{speech_text[:500]}\"."
            )
            new_scenes_data[i] = {
                "prompt": scene_prompt,
                "video_path": None,
                "audio_path": chunk_path,
                "start_image": image_path,
                "duration_sec": max(0.1, end_sec - start_sec),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "lyrics_text": speech_text,
                "first_frame": None,
                "last_frame": None,
            }
            v2_status(f"Cena de fala {i + 1}/{len(groups)} preparada", "Slicing speech", 4.0 + 90.0 * (i + 1) / len(groups))
        SCENES_DATA = new_scenes_data
        SCENES_JUST_SLICED = True
        v2_status(
            f"Speech transcribed and sliced into {len(groups)} scenes. Same image assigned to every scene; "
            "replace a later Start Image to change the speaker.",
            "Done", 100.0,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        v2_status(f"Speech slice error: {exc}", "Error", 0.0)

def _photo_files(folder):
    """Return supported photos in deterministic filename order."""
    if not folder or not os.path.isdir(folder):
        return []
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = [
        os.path.abspath(os.path.join(folder, name))
        for name in os.listdir(folder)
        if os.path.splitext(name)[1].lower() in extensions
    ]
    def natural_key(path):
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", os.path.basename(path))]
    return sorted(files, key=natural_key)

def interrogate_photo_clip(image_path):
    """Describe a photo with CLIP Interrogator, caching captions by file mtime."""
    global CLIP_INTERROGATOR_INSTANCE
    cache = {}
    try:
        if os.path.exists(PHOTO_CAPTION_CACHE_PATH):
            with open(PHOTO_CAPTION_CACHE_PATH, "r", encoding="utf-8") as handle:
                cache = json.load(handle)
    except Exception:
        cache = {}
    cache_key = f"{image_path}|{os.path.getmtime(image_path)}"
    if cache_key in cache:
        return cache[cache_key]
    v2_status(f"CLIP carregando/analisando: {os.path.basename(image_path)}", "CLIP Interrogator")
    with CLIP_INTERROGATOR_LOCK:
        if CLIP_INTERROGATOR_INSTANCE is None:
            from clip_interrogator import Config, Interrogator
            device = os.environ.get("CLIP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
            config = Config(
                device=device,
                caption_model_name="blip-base",
                clip_model_name="ViT-B-32/openai",
                cache_path=os.path.abspath("models/clip_interrogator"),
                caption_offload=True,
                clip_offload=True,
                quiet=True,
            )
            CLIP_INTERROGATOR_INSTANCE = Interrogator(config)
        from PIL import Image
        caption = CLIP_INTERROGATOR_INSTANCE.interrogate(Image.open(image_path).convert("RGB"))
        v2_status(f"CLIP descrição concluída: {os.path.basename(image_path)}", "CLIP Interrogator")
    cache[cache_key] = caption
    try:
        with open(PHOTO_CAPTION_CACHE_PATH, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return caption

def slice_photo_music(audio_path, photo_folder, prompt, fps, num_frames, analyze_clip=True):
    """Validate fast, then build photo/music scenes in a background thread.

    CLIP captioning ("first use downloads the BLIP/CLIP models and may take
    several minutes", per the UI's own warning) blocks the Gradio click
    callback for its entire duration if run inline -- see slice_speech's
    docstring for why that leaves the Worker Log frozen. Same fix: validate
    fast, then hand the slow work to a daemon thread.
    """
    if not audio_path:
        return "Upload the reference music first."
    photos = _photo_files(photo_folder)
    if not photos:
        return f"No supported photos found in: {photo_folder}"
    threading.Thread(
        target=_slice_photo_music_worker,
        args=(audio_path, photo_folder, prompt, fps, num_frames, analyze_clip),
        daemon=True,
    ).start()
    return f"{len(photos)} fotos encontradas. Analisando e fatiando... acompanhe o progresso no Worker Log abaixo."


def _slice_photo_music_worker(audio_path, photo_folder, prompt, fps, num_frames, analyze_clip=True):
    """Background body of slice_photo_music; runs off the Gradio callback thread."""
    global SCENES_DATA, SCENES_JUST_SLICED, CURRENT_TOTAL_SCENES, MUSIC_AUDIO_PATH
    v2_status("validando música e pasta de fotos", "Preparação", 1)
    photos = _photo_files(photo_folder)
    try:
        os.makedirs(AUDIO_CLIPS_DIR, exist_ok=True)
        waveform, sr, duration_sec = load_audio_compatible(audio_path)
        MUSIC_AUDIO_PATH = os.path.abspath(audio_path)
        photos = photos[:20]
        lyrics_segments, lyrics_path, _ = load_lyrics_segments(audio_path, duration_sec)
        if not lyrics_path or not lyrics_segments:
            try:
                vocal_analysis_path = get_or_create_vocal_stem(audio_path)
            except Exception:
                vocal_analysis_path = None
            generated_path, _ = transcribe_lyrics_for_audio(
                vocal_analysis_path or audio_path, source_audio=audio_path
            )
            if generated_path:
                lyrics_segments, lyrics_path, _ = load_lyrics_segments(audio_path, duration_sec)
        scene_duration_sec = duration_sec / max(1, len(photos))
        target_scene_sec = min(max(scene_duration_sec, 5.0), 10.0)
        boundaries, vocal_boundary_count = vocal_aware_boundaries(
            lyrics_segments, duration_sec, target_scene_sec, len(photos)
        )
        if boundaries is None:
            boundaries, downbeat_count = beat_aligned_boundaries(
                waveform, sr, duration_sec, target_scene_sec, len(photos)
            )
            boundary_note = f"BPM fallback: {downbeat_count} downbeats"
        else:
            boundary_note = f"vocal/ASR: {vocal_boundary_count} limites em fim de palavra/frase"
        CURRENT_TOTAL_SCENES = len(photos)
        v2_status("Analyzing photos and preparing scenes", "Analyzing photos", 2.0)
        new_scenes_data = [None] * 20
        base_prompt = (prompt or "Cinematic music video with coherent visual continuity").strip()
        captions_found = 0
        v2_status(f"{len(photos)} fotos encontradas; iniciando slice", "Slice", 5)
        for i, photo_path in enumerate(photos):
            if analyze_clip:
                caption = interrogate_photo_clip(photo_path)
                captions_found += bool(caption)
            else:
                caption = ""
            v2_status(f"cena {i+1}/{len(photos)} preparada", "Slice", 5 + 90*(i+1)/len(photos))
            start_sec, end_sec = boundaries[i], boundaries[i + 1]
            start_sample = int(start_sec * sr)
            end_sample = min(int(end_sec * sr), waveform.shape[1])
            chunk_path = os.path.abspath(os.path.join(AUDIO_CLIPS_DIR, f"photo_scene_{i+1}_audio.wav"))
            save_audio_compatible(chunk_path, waveform[:, start_sample:end_sample], sr)
            lyric_text = lyrics_for_window(lyrics_segments, start_sec, end_sec)
            scene_prompt = (
                f"Use the supplied reference photo as the exact visual identity for this scene. {base_prompt}. "
                "Preserve the same person, clothing and recognizable details from the reference image; cinematic motion, "
                "natural camera movement, music-video composition."
            )
            if caption:
                scene_prompt += f" CLIP visual description: {caption}."
            if lyric_text:
                scene_prompt += f' Lyric context: "{lyric_text}".'
            new_scenes_data[i] = {
                "prompt": scene_prompt,
                "video_path": None,
                "audio_path": chunk_path,
                "start_image": photo_path,
                "duration_sec": max(0.1, end_sec - start_sec),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "lyrics_text": lyric_text,
                "photo_path": photo_path,
                "clip_caption": caption,
                "first_frame": None,
                "last_frame": None,
            }
        SCENES_DATA = new_scenes_data
        SCENES_JUST_SLICED = True
        v2_status(
            f"Prepared {len(photos)} photo scenes; CLIP {captions_found}/{len(photos)}; {boundary_note}.",
            "Done", 100.0,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        v2_status(f"Photo/music slice error: {exc}", "Error", 0.0)

def process_chain_generation(scenes_list, audios_list, start_images_list, last_images_list, checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth, enable_lipsync=False, latentsync_steps=20, latentsync_guidance=1.5, room_preset="none", room_distance=0.0, start_index=None, mode="forward"):
    """
    mode: "forward" (chain from start_index up to end) or "single" (only start_index)
    """
    global CURRENT_LOG, LATEST_VIDEO_PATH, CURRENT_PROCESS, CURRENT_OUTPUT_PATH, IS_PROCESSING, STOP_GENERATION, SCENES_DATA
    global CURRENT_PHASE, CURRENT_PROGRESS, CURRENT_TOTAL_SCENES

    IS_PROCESSING = True
    STOP_GENERATION = False
    
    # scenes_list is a list of prompts (or None for empty slots)
    valid_indices = [i for i, p in enumerate(scenes_list) if p]
    if not valid_indices:
        IS_PROCESSING = False
        CURRENT_PHASE = "No scenes available"
        CURRENT_PROGRESS = 0.0
        return

    if mode == "single":
        indices_to_process = [start_index]
    else:
        # Forward chain from start_index (or the first valid index) up to the end
        current_start = start_index if start_index is not None else valid_indices[0]
        indices_to_process = [i for i in range(current_start, len(scenes_list)) if i in valid_indices]

    CURRENT_TOTAL_SCENES = len(indices_to_process)
    CURRENT_PHASE = "Preparing scene generation"
    CURRENT_PROGRESS = 2.0

    for i in indices_to_process:
        global CURRENT_SCENE_INDEX
        CURRENT_SCENE_INDEX = i
        scene_position = indices_to_process.index(i) + 1
        CURRENT_PHASE = f"Generating scene {scene_position}/{CURRENT_TOTAL_SCENES}"
        CURRENT_PROGRESS = 5.0 + 80.0 * (scene_position - 1) / max(1, CURRENT_TOTAL_SCENES)
        if STOP_GENERATION:
            CURRENT_LOG += "\n--- STOPPED BY USER ---\n"
            break
            
        prompt = scenes_list[i]
        scene_data = SCENES_DATA[i]
        
        if not scene_data:
             CURRENT_LOG += f"\nSkipping scene {i+1} : No Data\n"
             continue
             
        audio_path = audios_list[i] or scene_data.get('audio_path')
        scene_id = i + 1
        CURRENT_LOG += f"\n\n--- GENERATING SCENE {scene_id} ---\n"
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"scene_{scene_id:02d}_{timestamp}.mp4"
        output_path = os.path.abspath(os.path.join(run_subdir("scenes"), output_filename))
        CURRENT_OUTPUT_PATH = output_path
        
        # --- Context & Continuity Setup ---
        actual_prompt = prompt
        scene_duration = float(scene_data.get("duration_sec", int(num_frames) / fps))
        raw_frames = max(9, int(round(scene_duration * fps)))
        actual_num_frames = 8 * max(1, round((raw_frames - 1) / 8)) + 1
        conditioning_frames = []
        
        current_seed = seed
        if random_seed:
            current_seed = int(os.urandom(4).hex(), 16) % (2 ** 32)

        # Standard Continuity (Music Video likely doesn't need context compression as much as continuity? Let's keep Standard for now)
        def get_valid_path(img_data):
            if not img_data: return None
            if isinstance(img_data, str) and img_data.strip(): return img_data
            if isinstance(img_data, dict) and 'path' in img_data and img_data['path']: return img_data['path']
            return None

        custom_start = get_valid_path(start_images_list[i])
        
        if custom_start:
            conditioning_frames = [(custom_start, 0, 1.0)]
            CURRENT_LOG += f"Using custom Start Image for Scene {scene_id}\n"
        elif mode == "forward":
            if i > 0:
                custom_last = get_valid_path(last_images_list[i-1])
                if custom_last:
                    conditioning_frames = [(custom_last, 0, 1.0)]
                    CURRENT_LOG += f"Connecting Scene {scene_id} to Scene {i} (using custom Last Image Override)\n"
                elif SCENES_DATA[i-1] and SCENES_DATA[i-1]['video_path']:
                    prev_video = SCENES_DATA[i-1]['video_path']
                    
                    cap = cv2.VideoCapture(prev_video)
                    prev_frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    last_lat = (prev_frame_cnt - 1) // 8
                    
                    f_prev_l1 = os.path.join(run_subdir("frames"), f"scene_{scene_id:02d}_prev_last.jpg")
                    if extract_frame(prev_video, f_prev_l1, last_lat*8):
                         conditioning_frames = [(f_prev_l1, 0, 1.0)]
                         CURRENT_LOG += f"Connecting Scene {scene_id} to Scene {i} (last frames as latents)\n"

        # Build Command for music_to_video.py
        cmd = [
            sys.executable, "-m", "ltx_pipelines.music_to_video",
            "--distilled-checkpoint-path", checkpoint,
            "--gemma-root", gemma,
            "--spatial-upsampler-path", upsampler,
            "--prompt", actual_prompt,
            "--output-path", output_path,
            "--width", str(width),
            "--height", str(height),
            "--num-frames", str(int(actual_num_frames)),
            "--frame-rate", str(fps),
            "--num-inference-steps", str(int(steps)),
            "--seed", str(int(current_seed)),
            "--quantization", "fp8-cast"
        ]
        if os.environ.get("LTX_TORCH_COMPILE", "0").strip().lower() in {"1", "true", "yes", "on"}:
            cmd.append("--torch-compile")
            CURRENT_LOG += "torch.compile ativado para o transformer multimodal.\n"
        
        if audio_path:
            cmd.extend(["--audio-input-path", audio_path])
            
        for frame_path, latent_idx, guidance in conditioning_frames:
            cmd.extend(["--image", frame_path, str(latent_idx), str(guidance)])

        try:
            v2_status(
                f"LTX cena {scene_id}: iniciando checkpoint/pipeline ({actual_num_frames} frames, {width}x{height})",
                f"Gerando cena {scene_position}/{CURRENT_TOTAL_SCENES}",
                CURRENT_PROGRESS,
            )
            CURRENT_PROCESS = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )
            for line in CURRENT_PROCESS.stdout:
                if STOP_GENERATION:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(CURRENT_PROCESS.pid)], capture_output=True)
                    break
                CURRENT_LOG += line
                print(f"[LTX][cena {scene_id}] {line}", end="", flush=True)
            CURRENT_PROCESS.wait()
            if CURRENT_PROCESS.returncode != 0:
                CURRENT_LOG += (
                    f"LTX subprocess exited with code {CURRENT_PROCESS.returncode}. "
                    "Check GPU selection, FP8 policy, RAM/pagefile and the last model-dispatch line above.\n"
                )
            
            if CURRENT_PROCESS.returncode == 0 and os.path.exists(output_path):
                CURRENT_LOG += f"Scene {scene_id} Complete.\n"
                v2_status(f"LTX cena {scene_id} concluída: {output_path}", "Cena concluída", CURRENT_PROGRESS)
                LATEST_VIDEO_PATH = output_path
                
                # Update SCENES_DATA
                first_f = os.path.join(run_subdir("frames"), f"scene_{scene_id:02d}_first.jpg")
                last_f = os.path.join(run_subdir("frames"), f"scene_{scene_id:02d}_last.jpg")
                extract_first_frame(output_path, first_f)
                extract_last_frame(output_path, last_f)
                
                SCENES_DATA[i]['video_path'] = output_path
                SCENES_DATA[i]['first_frame'] = first_f
                SCENES_DATA[i]['last_frame'] = last_f
                CURRENT_PROGRESS = 5.0 + 80.0 * scene_position / max(1, CURRENT_TOTAL_SCENES)
            else:
                CURRENT_LOG += f"Scene {scene_id} Failed or Canceled.\n"
                break
        except Exception as e:
            CURRENT_LOG += f"Exception: {str(e)}\n"
            break
            
    CURRENT_LOG += "\n--- GENERATION CYCLE FINISHED ---\n"

    # After a complete forward chain, lip-sync eligible scenes, concatenate, remux master audio, then upscale.
    if mode == "forward" and not STOP_GENERATION:
        completed_paths = [SCENES_DATA[i].get("video_path") for i in valid_indices if SCENES_DATA[i]]
        paths_ready = (
            completed_paths
            and len(completed_paths) == len(valid_indices)
            and all(p is not None and os.path.exists(p) for p in completed_paths)
        )
        if paths_ready:
            try:
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                paths_for_concat = list(completed_paths)
                spatialized_master_path = None
                room_active = room_preset and room_preset != "none"
                vocal_source = None
                if enable_lipsync or room_active:
                    vocal_source = get_or_create_vocal_stem(MUSIC_AUDIO_PATH) if MUSIC_AUDIO_PATH else None
                    if not vocal_source:
                        v2_status("vocal isolado indisponível; usando áudio original das cenas", CURRENT_PHASE, 86)
                if enable_lipsync:
                    CURRENT_PHASE = "Lip-sync per scene (LatentSync 1.6)"
                    CURRENT_PROGRESS = 85.0
                    for position, scene_index in enumerate(valid_indices):
                        scene_data = SCENES_DATA[scene_index] or {}
                        lyric_text = str(scene_data.get("lyrics_text") or "").strip()
                        if not lyric_text:
                            v2_status(
                                f"cena {scene_index + 1}: instrumental/sem letra; lip-sync ignorado",
                                CURRENT_PHASE,
                                86.0 + 5.0 * (position + 1) / max(1, len(valid_indices)),
                            )
                            continue
                        source_audio = vocal_source or scene_data.get("audio_path")
                        lip_audio = slice_lipsync_audio(source_audio, scene_data, scene_index + 1) if vocal_source else source_audio
                        lip_output = os.path.abspath(os.path.join(
                            run_subdir("lipsync"), f"scene_{scene_index + 1:02d}_{stamp}_latentsync.mp4"
                        ))
                        synced = run_latentsync(
                            paths_for_concat[position], lip_audio, lip_output,
                            inference_steps=latentsync_steps,
                            guidance_scale=latentsync_guidance,
                        )
                        if not synced:
                            fallback_output = os.path.abspath(os.path.join(
                                run_subdir("lipsync"), f"scene_{scene_index + 1:02d}_{stamp}_wav2lip.mp4"
                            ))
                            synced = run_wav2lip(paths_for_concat[position], lip_audio, fallback_output)
                        if synced:
                            paths_for_concat[position] = synced
                            scene_data["lipsync_video_path"] = synced
                        else:
                            v2_status(f"cena {scene_index + 1}: rosto não detectado; mantendo clipe original", CURRENT_PHASE)

                if room_active and vocal_source:
                    CURRENT_PHASE = "Ambiente/reverb da voz"
                    CURRENT_PROGRESS = 89.0
                    label = audio_fx.ROOM_PRESET_LABELS.get(room_preset, room_preset)
                    v2_status(f"aplicando ambiente '{label}' (distancia={float(room_distance):.2f}) ao vocal isolado", CURRENT_PHASE, CURRENT_PROGRESS)
                    instrumental_source = get_instrumental_stem_path(MUSIC_AUDIO_PATH)
                    spatialized_master_path = audio_fx.build_spatialized_master(
                        vocal_source, instrumental_source, run_subdir("intermediate"),
                        preset=room_preset, distance=room_distance, stamp=stamp,
                        ffmpeg=os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe"),
                        log=lambda msg: v2_status(msg, CURRENT_PHASE),
                    )
                    if spatialized_master_path:
                        v2_status("faixa mestre espacializada pronta para o remux final", CURRENT_PHASE, 90.0)
                    else:
                        v2_status("reverb de ambiente falhou; usando a faixa original no remux final", CURRENT_PHASE, 90.0)

                CURRENT_PHASE = "Concatenating scene videos"
                CURRENT_PROGRESS = 90.0
                v2_status(f"concatenando {len(paths_for_concat)} clipe(s)", CURRENT_PHASE, CURRENT_PROGRESS)
                concat_list = os.path.abspath(os.path.join(run_subdir("intermediate"), "concat_list.txt"))
                concat_video = os.path.abspath(os.path.join(run_subdir("intermediate"), "music_video_concat.mp4"))
                final_video = os.path.abspath(os.path.join(run_subdir("final"), "music_video_final.mp4"))
                upscaled_video = os.path.abspath(os.path.join(run_subdir("final"), "music_video_final_x3.mp4"))
                with open(concat_list, "w", encoding="ascii") as list_file:
                    for path in paths_for_concat:
                        list_file.write(f"file '{path.replace(chr(92), '/')}'\n")

                ffmpeg = os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
                CURRENT_LOG += f"\nConcatenating {len(paths_for_concat)} scenes...\n"
                concat_proc = subprocess.run(
                    [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                     "-an", "-c", "copy", concat_video], capture_output=True, text=True
                )
                if concat_proc.returncode != 0:
                    CURRENT_LOG += "Concatenação sem recompressão incompatível; reencodificando H.264.\n"
                    concat_proc = subprocess.run(
                        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                         "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                         "-pix_fmt", "yuv420p", concat_video], capture_output=True, text=True
                    )
                if concat_proc.returncode != 0:
                    raise RuntimeError(concat_proc.stderr[-2000:])

                ambient_enabled = os.environ.get("LTX_AMBIENT_AUDIO", "0").lower() in {"1", "true", "yes", "on"}
                ambient_volume = max(0.0, min(1.0, float(os.environ.get("LTX_AMBIENT_VOLUME", "0.35"))))
                ambient_audio = os.path.abspath(os.path.join(run_subdir("intermediate"), "ltx_ambient_audio.wav"))
                ambient_ok = False
                if ambient_enabled:
                    v2_status("extraindo camada de som gerada pelo LTX", "Áudio ambiente", 92.0)
                    ambient_proc = subprocess.run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-vn", "-ac", "2", "-ar", "48000", ambient_audio], capture_output=True, text=True)
                    ambient_ok = ambient_proc.returncode == 0 and os.path.exists(ambient_audio)
                    if not ambient_ok:
                        CURRENT_LOG += "Áudio ambiente LTX indisponível nos clipes; usando apenas a trilha original.\n"
                final_audio_source = spatialized_master_path if (spatialized_master_path and os.path.exists(spatialized_master_path)) else MUSIC_AUDIO_PATH
                if final_audio_source and os.path.exists(final_audio_source):
                    if spatialized_master_path and final_audio_source == spatialized_master_path:
                        v2_status("recolocando a faixa com ambiente/reverb aplicado", "Áudio final", 94)
                    else:
                        v2_status("recolocando a música original sem cortes", "Áudio final", 94)
                    if ambient_ok:
                        remux_args = [ffmpeg, "-y", "-i", concat_video, "-i", final_audio_source, "-i", ambient_audio,
                            "-filter_complex", f"[1:a]volume=1[music];[2:a]volume={ambient_volume}[amb];[music][amb]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                            "-map", "0:v:0", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", final_video]
                    else:
                        remux_args = [ffmpeg, "-y", "-i", concat_video, "-i", final_audio_source,
                            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", final_video]
                    remux_proc = subprocess.run(remux_args, capture_output=True, text=True)
                    if remux_proc.returncode != 0:
                        raise RuntimeError(remux_proc.stderr[-2000:])
                else:
                    final_video = concat_video

                video_for_upscale = final_video

                CURRENT_PHASE = "Upscaling final video 3x"
                CURRENT_PROGRESS = 96.0
                v2_status("iniciando upscale Real-ESRGAN 3x", CURRENT_PHASE, CURRENT_PROGRESS)
                CURRENT_LOG += "Applying Real-ESRGAN upscale 3x...\n"
                upscale_script = os.path.abspath("upscale_video.ps1")
                upscale_proc = subprocess.run(
                    ["C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                     "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", upscale_script,
                     "-InFile", video_for_upscale, "-Output", upscaled_video,
                     "-Model", "realesr-animevideov3", "-Scale", "3",
                     "-Gpu", os.environ.get("LTX_UPSCALE_GPU", "1")],
                    capture_output=True, text=True
                )
                if upscale_proc.returncode != 0:
                    raise RuntimeError(upscale_proc.stderr[-2000:] or upscale_proc.stdout[-2000:])

                LATEST_VIDEO_PATH = upscaled_video
                CURRENT_PHASE = "Complete"
                CURRENT_PROGRESS = 100.0
                v2_status(f"vídeo final pronto: {upscaled_video}", CURRENT_PHASE, CURRENT_PROGRESS)
                CURRENT_LOG += f"Final video ready: {upscaled_video}\n"
            except Exception as post_error:
                CURRENT_PHASE = "Post-processing error"
                CURRENT_LOG += f"Post-processing error: {post_error}\n"
            finally:
                try:
                    os.remove(concat_list)
                except Exception:
                    pass
        else:
            missing_scenes = [
                str(i + 1) for i in valid_indices
                if not SCENES_DATA[i] or not SCENES_DATA[i].get("video_path")
            ]
            CURRENT_PHASE = "Generation incomplete"
            CURRENT_LOG += (
                "Post-processing skipped: no valid video path for scene(s) "
                + ", ".join(missing_scenes or ["unknown"])
                + ". Fix the failed scene and run the chain again.\n"
            )

    if CURRENT_PHASE not in ("Complete", "Post-processing error", "Generation incomplete"):
        CURRENT_PHASE = "Complete"
        CURRENT_PROGRESS = 100.0
    if CURRENT_RUN_DIR:
        try:
            with open(os.path.join(run_subdir("logs"), "process.log"), "w", encoding="utf-8") as handle:
                handle.write(CURRENT_LOG)
        except Exception as log_error:
            print(f"[V3] Could not save run log: {log_error}", flush=True)
    IS_PROCESSING = False

def start_generation_thread(prompts, audios, start_images, last_images, checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth, enable_lipsync=False, latentsync_steps=20, latentsync_guidance=1.5, torch_compile_enabled=False, ambient_enabled=False, ambient_volume=0.35, room_preset="none", room_distance=0.0, start_index=None, mode="forward"):
    global CURRENT_PHASE, CURRENT_PROGRESS
    CURRENT_PHASE = "Queued"
    CURRENT_PROGRESS = 0.0
    os.environ["LTX_TORCH_COMPILE"] = "1" if torch_compile_enabled else "0"
    os.environ["LTX_AMBIENT_AUDIO"] = "1" if ambient_enabled else "0"
    os.environ["LTX_AMBIENT_VOLUME"] = str(float(ambient_volume or 0.35))
    audios, start_images = create_generation_folder(prompts, audios, start_images)
    # Clear subsequent scenes in data if starting a chain or single regeneration logic?
    # For music video, if we regenerate, we keep the audio_path!
    # So we should only clear video_path.
    
    if mode == "forward":
        begin_idx = start_index if start_index is not None else 0
        for i in range(begin_idx, 20):
            if SCENES_DATA[i]:
                SCENES_DATA[i]['video_path'] = None
                SCENES_DATA[i]['first_frame'] = None
                SCENES_DATA[i]['last_frame'] = None

    threading.Thread(target=process_chain_generation, args=(prompts, audios, start_images, last_images, checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth, enable_lipsync, latentsync_steps, latentsync_guidance, room_preset, room_distance, start_index, mode), daemon=True).start()
    return f"Generation started. Output folder: {CURRENT_RUN_DIR}"

def stop_generation():
    global STOP_GENERATION
    STOP_GENERATION = True
    return "Stopping..."

def update_ui():
    global LATEST_VIDEO_PATH, CURRENT_LOG, SCENES_DATA, CURRENT_OUTPUT_PATH, CURRENT_SCENE_INDEX
    global CURRENT_PHASE, CURRENT_PROGRESS, SCENES_JUST_SLICED
    status = f"{CURRENT_PHASE} | {CURRENT_PROGRESS:.1f}%"

    # Prepare updates for all scene boxes
    updates = []
    for i in range(20):
        data = SCENES_DATA[i]

        display_video = data.get('video_path') if data else None
        display_preview = data.get('last_frame') if data else None

        # Intermediate preview logic
        if IS_PROCESSING and CURRENT_SCENE_INDEX == i and CURRENT_OUTPUT_PATH:
            preview_file = CURRENT_OUTPUT_PATH.replace('.mp4', '_.mp4')
            if os.path.exists(preview_file):
                display_video = preview_file

        if data or (IS_PROCESSING and CURRENT_SCENE_INDEX == i):
            v_val = display_video
            p_val = display_preview
            updates.append(gr.update(value=v_val, visible=True))
            updates.append(gr.update(value=p_val, visible=True))
        else:
            updates.append(gr.update(value=None)) # Video
            updates.append(gr.update(value=None)) # Image

    # Scene-editor fields (prompt/audio/start image/last image) only refresh
    # right when a background slice (speech or photo) has just finished --
    # not on every tick -- so a user's manual edit to a scene between slices
    # is never overwritten by this same timer.
    if SCENES_JUST_SLICED:
        SCENES_JUST_SLICED = False
        editor_updates = []
        for i in range(20):
            data = SCENES_DATA[i]
            if data:
                editor_updates.extend([
                    gr.update(visible=True),
                    gr.update(value=data.get("prompt"), visible=True),
                    gr.update(value=data.get("audio_path")),
                    gr.update(value=data.get("start_image")),
                    gr.update(value=None),
                ])
            else:
                editor_updates.extend([
                    gr.update(visible=False), gr.update(visible=False), gr.update(value=None),
                    gr.update(value=None), gr.update(value=None),
                ])
    else:
        editor_updates = [gr.update()] * (20 * 5)

    return tuple([LATEST_VIDEO_PATH, CURRENT_LOG, status] + updates + editor_updates)

def cancel_job():
    global CURRENT_PROCESS, CURRENT_LOG
    if CURRENT_PROCESS:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(CURRENT_PROCESS.pid)], capture_output=True)
            CURRENT_LOG += "\n--- CANCELED ---\n"
            return "Canceled."
        except:
            return "Error canceling."
    return "No active process."

# --- Guided prompt composer ---
def compose_guided_prompt(category, shot, lighting, palette, atmosphere, camera, rhythm, vfx,
                          character, dialogue_style, dialogue_volume, language, ambient, end_motion, free_text):
    parts = []
    if category and category != "None": parts.append(str(category))
    if shot: parts.append(f"{shot} shot")
    if free_text: parts.append(str(free_text).strip())
    if character: parts.append(f"Character: {character.strip()}")
    if lighting: parts.append(f"Lighting: {', '.join(lighting) if isinstance(lighting, list) else lighting}")
    if palette and palette != "None": parts.append(f"{palette} color palette")
    if atmosphere: parts.append(f"Atmosphere: {', '.join(atmosphere) if isinstance(atmosphere, list) else atmosphere}")
    if camera: parts.append(f"Camera movement: {', '.join(camera) if isinstance(camera, list) else camera}")
    if end_motion: parts.append(f"At the end of the movement, visibly show: {end_motion.strip()}")
    if rhythm and rhythm != "None": parts.append(f"Timing: {rhythm}")
    if vfx: parts.append(f"VFX: {', '.join(vfx) if isinstance(vfx, list) else vfx}")
    if dialogue_style: parts.append(f"Delivery: {dialogue_style}")
    if dialogue_volume: parts.append(f"Voice volume: {dialogue_volume}")
    if language: parts.append(f"Language/accent: {language}")
    if ambient: parts.append(f"Ambient sound: {ambient}")
    return ". ".join(p.strip(". ") for p in parts if p and p.strip()) + "."

def composer_preview(*values):
    prompt = compose_guided_prompt(*values)
    count = len([s for s in prompt.replace("!", ".").replace("?", ".").split(".") if s.strip()])
    return prompt, f"Sentences: {count} (recommended 4–8)"

PROMPT_EXAMPLES = {
    "Monster truck documentary": "Cinematic documentary, wide epic shot. A monster truck crosses a dusty arena, dust particles in backlight, slow crane movement.",
    "Dramatic comedy scene": "Cinematic film noir, medium shot. Two friends argue in a small kitchen, expressive gestures, warm practical light, slow push in.",
    "Sci-fi talk show": "Cinematic space opera, medium shot. A host interviews a guest on a futuristic talk show, neon glow, over-the-shoulder camera, shallow depth of field.",
    "Claymation frogs doing yoga": "Animation: claymation, wide playful shot. Three frogs practice yoga in a forest clearing, soft rim light, gentle dolly in.",
}

def example_prompt(name):
    return PROMPT_EXAMPLES.get(name, "")

def lint_prompt(prompt):
    text = (prompt or "").lower()
    warnings = []
    if any(w in text for w in ("text", "logo", "sign", "lettering")): warnings.append("⚠️ Texto/logo legível pode sair deformado no LTX-2.")
    if any(w in text for w in ("jump", "juggle", "flip", "somersault")): warnings.append("⚠️ Física caótica detectada; prefira descrever dança ou movimento controlado.")
    if text.count(" and ") > 3: warnings.append("⚠️ Muitas ações simultâneas; tente manter até 3 ações por cena.")
    if "warm sunset" in text and "cold fluorescent" in text: warnings.append("⚠️ Fontes de luz conflitantes detectadas.")
    return "\n".join(warnings) if warnings else "✅ Prompt sem alertas principais."

# --- UI Layout ---

theme = gr.themes.Soft(primary_hue="purple").set(
    body_background_fill="*neutral_50",
    block_background_fill="*neutral_100",
)

with gr.Blocks(title="LTX-2 Music Video Maker V3 - LatentSync 1.6") as demo:
    gr.Markdown("# 🎵 LTX-2 Music Video Maker")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_file = gr.Audio(label="Upload Music File", type="filepath")
            master_prompt = gr.Textbox(label="Visual Style / Prompt", placeholder="Cyberpunk city, neon lights, rain...", lines=2)
            gr.Markdown("### Prompt Composer (English)")
            with gr.Row():
                prompt_category = gr.Dropdown(["None", "Animation: stop-motion", "Animation: 2D", "Animation: 3D", "Animation: claymation", "Stylized: comic book", "Stylized: cyberpunk", "Stylized: painterly", "Cinematic: documentary", "Cinematic: film noir", "Cinematic: fantasy", "Cinematic: thriller"], value="None", label="Category / genre")
                prompt_shot = gr.Dropdown(["wide expansive", "wide epic", "medium", "close-up intimate", "extreme close-up claustrophobic"], value="wide expansive", label="Shot scale")
            with gr.Row():
                prompt_lighting = gr.CheckboxGroup(["golden hour", "backlighting", "neon glow", "dramatic shadows", "flickering candles", "soft rim light"], label="Lighting")
                prompt_palette = gr.Dropdown(["None", "vibrant", "muted", "monochromatic", "high contrast"], value="None", label="Color palette")
            with gr.Row():
                prompt_atmosphere = gr.CheckboxGroup(["fog", "rain", "dust", "mist", "smoke", "reflections"], label="Atmosphere")
                prompt_camera = gr.CheckboxGroup(["pan left", "pan right", "dolly in", "dolly out", "handheld", "static", "crane up", "tilt up", "push in", "pull back", "wide establishing"], label="Camera movement")
            with gr.Row():
                prompt_rhythm = gr.Dropdown(["None", "slow motion", "time-lapse", "rapid cuts", "lingering shot", "freeze-frame", "fade-in/out"], value="None", label="Rhythm / timing")
                prompt_vfx = gr.CheckboxGroup(["motion blur", "shallow depth of field", "particle systems"], label="VFX")
            with gr.Row():
                prompt_character = gr.Textbox(label="Character (age, hair, clothes, distinctive trait)", lines=1)
                prompt_end_motion = gr.Textbox(label="Visible at end of camera movement", lines=1)
            with gr.Row():
                prompt_dialogue_style = gr.Dropdown(["", "energetic announcer", "resonant with gravitas", "distorted radio", "robotic monotone", "childlike curiosity"], label="Delivery")
                prompt_dialogue_volume = gr.Dropdown(["", "whisper", "mutters", "shouts", "screams"], label="Voice volume")
                prompt_language = gr.Textbox(label="Language / accent", lines=1)
            prompt_ambient = gr.Textbox(label="Ambient sound description", placeholder="rain and wind, forest birds, coffeeshop", lines=1)
            prompt_preview = gr.Textbox(label="Prompt preview (editable)", value="Cinematic music video with coherent visual continuity.", lines=4, interactive=True)
            prompt_sentence_badge = gr.Markdown("Sentences: 0 (recommended 4–8)")
            prompt_example = gr.Dropdown(["", *PROMPT_EXAMPLES.keys()], label="Load example")
            prompt_lint = gr.Markdown("✅ Prompt sem alertas principais.")
            lyrics_enabled = gr.Checkbox(
                label="Use lyrics_timing.json in scene prompts",
                info="Adds ASR lyric text for each beat-aligned scene. Re-slice after changing.",
                value=True,
            )
            enable_lipsync = gr.Checkbox(
                label="Enable advanced lip-sync (LatentSync 1.6)",
                info="Runs LatentSync 1.6 per singing scene on the RTX 3090, with Wav2Lip fallback, then restores the original music.",
                value=False,
            )
            torch_compile_enabled = gr.Checkbox(label="Enable torch.compile", value=False, info="May recompile when LTX integer step attributes change; disable for maximum stability.")
            ambient_enabled = gr.Checkbox(label="Mix LTX-generated ambient audio", value=False)
            ambient_volume = gr.Slider(label="Generated ambient volume", minimum=0.0, maximum=1.0, value=0.35, step=0.05)
            room_preset = gr.Dropdown(
                choices=audio_fx.ROOM_PRESET_CHOICES,
                value="none",
                label="Ambiente da cena (reverb da voz)",
                info="Aplicado no vocal isolado DEPOIS do lip-sync (a sincronia labial usa sempre a voz seca); dá naturalidade/espaço à fala do personagem sem afetar o timing.",
            )
            room_distance = gr.Slider(
                label="Distância do personagem à câmera",
                minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                info="0 = close-up/perto; 1 = longe (mais reverb, mais abafado, mais baixo).",
            )
            with gr.Accordion("LatentSync 1.6 Settings", open=False):
                latentsync_steps = gr.Slider(
                    label="LatentSync inference steps",
                    minimum=20,
                    maximum=50,
                    step=1,
                    value=20,
                    info="20 is faster; higher values can improve visual quality.",
                )
                latentsync_guidance = gr.Slider(
                    label="LatentSync guidance scale",
                    minimum=1.0,
                    maximum=3.0,
                    step=0.1,
                    value=1.5,
                    info="Higher values can improve lip-sync but may add distortion or jitter.",
                )
            
            with gr.Accordion("LTX-2 Settings", open=True):
                checkpoint = gr.Textbox(label="Checkpoint", value=DEFAULT_CHECKPOINT)
                gemma = gr.Textbox(label="Gemma Root", value=DEFAULT_GEMMA)
                upsampler = gr.Textbox(label="Upsampler", value=DEFAULT_UPSAMPLER)
                with gr.Row():
                    steps = gr.Slider(
                        label="Distilled steps (fixed at 8)", minimum=1, maximum=50, value=8,
                        interactive=False, info="The native distilled schedule uses eight fixed sigma values.",
                    )
                    fps = gr.Number(label="FPS", value=24)
                with gr.Row():
                    width = gr.Number(label="Width", value=1280)
                    height = gr.Number(label="Height", value=704)
                num_frames = gr.Slider(label="Frames per Scene", minimum=9, maximum=257, step=8, value=225)
                quick_profile_btn = gr.Button("⚡ Aplicar teste mínimo 320×128", size="sm")
                gr.Markdown(
                    "Perfil de teste: 320×128, 24 FPS, 9 frames e 4 steps. "
                    "A resolução atende ao múltiplo de 64 exigido pelo pipeline de duas etapas."
                )
                quick_profile_btn.click(
                    fn=apply_low_resolution_test_profile,
                    outputs=[steps, fps, width, height, num_frames],
                )
                
                slice_btn = gr.Button("🔪 Slice Music & Prepare Scenes", variant="primary")
                
                with gr.Row():
                    seed = gr.Number(label="Seed", value=10, precision=0)
                    random_seed = gr.Checkbox(label="Random Seed", value=True)
                with gr.Row():
                    # context compression not prioritized for now, keeping args for compatibility
                    use_context_compression = gr.Checkbox(label="Use Context Compression", value=False, visible=False) 
                    latent_reuse_count = gr.Slider(label="Latent Reuse", minimum=1, maximum=8, step=1, value=2, visible=False)
                    context_depth = gr.Slider(label="Context Depth", minimum=1, maximum=5, step=1, value=2, visible=False)

        with gr.Column(scale=3):
            gr.Markdown("### 🎞️ Video Scenes")
            scene_rows = []
            scene_prompts = []
            scene_audios = [] 
            scene_start_images = []
            scene_last_images = []
            scene_videos = []
            scene_previews = []
            scene_reg_chain_btns = []
            scene_reg_single_btns = []
            
            for i in range(1, 21):
                with gr.Row(visible=False) as row: # Hidden until decomposed
                    scene_rows.append(row)
                    with gr.Column(scale=3):
                        prompt_box = gr.Textbox(label=f"Scene {i} Prompt", lines=2)
                        scene_prompts.append(prompt_box)
                        audio_comp = gr.Audio(label=f"Scene {i} Audio", type="filepath", interactive=True)
                        scene_audios.append(audio_comp)
                        
                        with gr.Row():
                            start_img = gr.Image(label="Start Image (optional)", type="filepath")
                            last_img = gr.Image(label="Last Image Override (optional)", type="filepath")
                            scene_start_images.append(start_img)
                            scene_last_images.append(last_img)
                        
                        with gr.Row():
                            chain_btn = gr.Button(f"🔗 Chain From {i}", size="sm")
                            single_btn = gr.Button(f"🎯 Only {i}", size="sm")
                            scene_reg_chain_btns.append(chain_btn)
                            scene_reg_single_btns.append(single_btn)
                    
                    video_comp = gr.Video(label="Clip", scale=2)
                    preview_comp = gr.Image(label="Last Frame", scale=1) 
                    
                    scene_videos.append(video_comp)
                    scene_previews.append(preview_comp)
            
            with gr.Row():
                generate_btn = gr.Button("🚀 Start Full Forward Chain", variant="primary", size="lg")
                stop_btn = gr.Button("🛑 Stop", variant="secondary", size="lg")
                cancel_btn = gr.Button("🗑️ Kill Process", variant="stop")
            
            latest_video = gr.Video(label="Latest Generated Scene (Global View)")
            status_box = gr.Textbox(label="Status", interactive=False)

    with gr.Tabs():
        with gr.Tab("🗣️ Speech / Talking Head"):
            gr.Markdown("### Discurso / áudio falado")
            gr.Markdown("Envie o discurso e uma imagem-base. O ASR cria as cenas automaticamente; a mesma imagem é mantida até você substituí-la no campo **Start Image** de uma cena posterior.")
            with gr.Row():
                speech_audio = gr.Audio(label="Speech audio", type="filepath")
                speech_image = gr.Image(label="Base speaker image", type="filepath")
            speech_prompt = gr.Textbox(
                label="Speech visual prompt",
                value="A realistic presenter speaking to camera in a cinematic setting",
                lines=2,
            )
            with gr.Row():
                speech_slice_btn = gr.Button("🗣️ Transcribe & Slice Speech", variant="primary")
                speech_generate_btn = gr.Button("🚀 Generate Talking-Head Video", variant="primary")
            speech_status = gr.Textbox(label="Speech status", interactive=False)
            gr.Markdown("Após a divisão, ajuste qualquer **Start Image** nas cenas para trocar o personagem a partir daquele ponto. A geração usa a concatenação e o upscale 3x do fluxo principal.")

        with gr.Tab("📷 Photos + Music (V3 + LatentSync)"):
            gr.Markdown("### Videoclipe a partir de uma pasta de fotos")
            gr.Markdown("As fotos são associadas sequencialmente às cenas. O Interrogador CLIP descreve cada imagem e acrescenta a descrição ao prompt; as imagens entram como referência visual inicial de cada clipe.")
            with gr.Row():
                photo_music_audio = gr.Audio(label="Reference music", type="filepath")
                photo_folder = gr.Textbox(
                    label="Photo folder",
                    value=r"E:\Users\home\Documents\captura",
                    info="Folder scanned for JPG, PNG, WEBP or BMP files.",
                )
            photo_master_prompt = gr.Textbox(
                label="Global visual style / prompt",
                value="Cinematic music video, coherent visual identity, expressive camera movement",
                lines=2,
            )
            photo_clip_analysis = gr.Checkbox(
                label="Analyze photos with CLIP Interrogator",
                value=True,
                info="First use downloads the BLIP/CLIP models and may take several minutes.",
            )
            with gr.Row():
                photo_slice_btn = gr.Button("📷 Analyze Photos + Slice Music", variant="primary")
                photo_generate_btn = gr.Button("🚀 Generate Photo Music Video", variant="primary")
            photo_status = gr.Textbox(label="Photo/music status", interactive=False)
            gr.Markdown("After slicing, you can replace any scene's **Start Image** manually. The normal chain, concatenation and 3x upscale are reused.")
    
    with gr.Accordion("Worker Log", open=False):
        log_box = gr.Textbox(label=None, lines=10, interactive=False)

    # --- Events ---
    composer_inputs = [prompt_category, prompt_shot, prompt_lighting, prompt_palette, prompt_atmosphere, prompt_camera, prompt_rhythm, prompt_vfx, prompt_character, prompt_dialogue_style, prompt_dialogue_volume, prompt_language, prompt_ambient, prompt_end_motion, master_prompt]
    for _component in composer_inputs:
        _component.change(fn=composer_preview, inputs=composer_inputs, outputs=[prompt_preview, prompt_sentence_badge])
    prompt_example.change(fn=example_prompt, inputs=[prompt_example], outputs=[prompt_preview])
    prompt_preview.change(fn=lint_prompt, inputs=[prompt_preview], outputs=[prompt_lint])
    slice_btn.click(
        fn=slice_audio,
        inputs=[audio_file, prompt_preview, fps, num_frames, lyrics_enabled],
        outputs=[status_box] + [comp for tuple_5 in zip(scene_rows, scene_prompts, scene_audios, scene_start_images, scene_last_images) for comp in tuple_5]
    )
    
    # Fica FORA do fluxo de geração de propósito: é ferramenta de
    # pós-produção, aplicada ao vídeo final já concatenado e upscalado,
    # e só quando o usuário quiser.
    video_doctor_ui.build_doctor_tab(
        get_default_video=lambda: LATEST_VIDEO_PATH,
        label="🩺 Diagnóstico e correção")

    def collect_prompts_and_start(*args):
        # Args structure: [prompts_list..., audios..., start_img..., last_img..., checkpoint, ..., button_args]
        # We need to slice args.
        num_scenes = 20
        prompts = args[0:20]
        audios = args[20:40]
        start_images = args[40:60]
        last_images = args[60:80]
        rest = args[80:]
        return start_generation_thread(prompts, audios, start_images, last_images, *rest)

    all_inputs = scene_prompts + scene_audios + scene_start_images + scene_last_images + [checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth, enable_lipsync, latentsync_steps, latentsync_guidance, torch_compile_enabled, ambient_enabled, ambient_volume, room_preset, room_distance]

    # Only speech_status is a direct return value now: slice_speech kicks off
    # a background thread and returns immediately. The scene editor fields
    # (prompt/audio/start image rows) are populated by the Timer below, once
    # SCENES_JUST_SLICED signals the thread has finished.
    speech_slice_btn.click(
        fn=slice_speech,
        inputs=[speech_audio, speech_image, speech_prompt, fps, num_frames],
        outputs=[speech_status]
    )

    speech_generate_btn.click(
        fn=collect_prompts_and_start,
        inputs=all_inputs,
        outputs=[status_box]
    )

    # Same background-thread pattern as speech_slice_btn above.
    photo_slice_btn.click(
        fn=slice_photo_music,
        inputs=[photo_music_audio, photo_folder, photo_master_prompt, fps, num_frames, photo_clip_analysis],
        outputs=[photo_status]
    )

    photo_generate_btn.click(
        fn=collect_prompts_and_start,
        inputs=all_inputs,
        outputs=[status_box]
    )
    
    generate_btn.click(
        fn=collect_prompts_and_start,
        inputs=all_inputs,
        outputs=[status_box]
    )

    stop_btn.click(fn=stop_generation, outputs=[status_box])
    cancel_btn.click(fn=cancel_job, outputs=[status_box])

    # Per-scene buttons
    for i in range(20):
        def make_chain_fn(index):
            def chain_fn(*args):
                num_scenes = 20
                prompts = args[0:20]
                audios = args[20:40]
                start_images = args[40:60]
                last_images = args[60:80]
                rest = args[80:]
                return start_generation_thread(prompts, audios, start_images, last_images, *rest, start_index=index, mode="forward")
            return chain_fn
            
        def make_single_fn(index):
            def single_fn(*args):
                num_scenes = 20
                prompts = args[0:20]
                audios = args[20:40]
                start_images = args[40:60]
                last_images = args[60:80]
                rest = args[80:]
                return start_generation_thread(prompts, audios, start_images, last_images, *rest, start_index=index, mode="single")
            return single_fn

        scene_reg_chain_btns[i].click(
            fn=make_chain_fn(i),
            inputs=all_inputs,
            outputs=[status_box]
        )
        
        scene_reg_single_btns[i].click(
            fn=make_single_fn(i),
            inputs=all_inputs,
            outputs=[status_box]
        )
    
    timer = gr.Timer(2)
    timer.tick(
        fn=update_ui,
        outputs=[latest_video, log_box, status_box]
        + [comp for zip_list in zip(scene_videos, scene_previews) for comp in zip_list]
        + [comp for tuple_5 in zip(scene_rows, scene_prompts, scene_audios, scene_start_images, scene_last_images) for comp in tuple_5]
    )

if __name__ == "__main__":
    # Gradio must be allowed to serve reference photos outside the project
    # directory when scene rows return their absolute paths.
    demo.launch(
        server_name=os.environ.get("LTX_UI_HOST", "127.0.0.1"),
        server_port=7804,
        theme=theme,
        allowed_paths=[r"E:\Users\home\Documents\captura"],
    )
