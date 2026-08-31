import gradio as gr
import video_doctor_ui  # aba de diagnostico temporal pos-geracao
import subprocess
import os
import datetime
import threading
import sys
import math
import json
import warnings
import numpy as np
import torch
import torchaudio
import wave
import librosa
from collections import deque
import cv2

# --- Configuration & Defaults ---
DEFAULT_CHECKPOINT = "./models/ltx-2.3-22b-distilled-fp8.safetensors"
DEFAULT_GEMMA = "./models/gemma3"
DEFAULT_UPSAMPLER = "./models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
AUDIO_CLIPS_DIR = "./audio_clips"

# --- Global State ---
JOB_QUEUE = deque()
QUEUE_LOCK = threading.Lock()
CURRENT_LOG = "System Ready. Waiting for jobs..."
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
# SCENES_DATA will store: {'prompt': str, 'video_path': str, 'audio_path': str, 'first_frame': str, 'last_frame': str}
SCENES_DATA = [None] * 20  # Increased to 20 scenes support

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

def load_lyrics_segments(audio_path=None, duration_sec=None):
    """Load timestamps only when their metadata belongs to the selected audio."""
    audio_abs = os.path.normcase(os.path.normpath(os.path.abspath(audio_path))) if audio_path else None
    stem = os.path.splitext(os.path.basename(audio_path))[0] if audio_path else ""
    candidates = [
        os.path.join(os.getcwd(), f"lyrics_timing_{stem}.json") if stem else None,
        os.path.join(os.getcwd(), "lyrics_timing.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f"lyrics_timing_{stem}.json") if stem else None,
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

def transcribe_lyrics_for_audio(audio_path):
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
    source_audio = os.path.abspath(audio_path)
    command = [sys.executable, transcriber, source_audio, "--output", output_path, "--model", "base"]
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

def run_wav2lip(video_path, audio_path, output_path):
    """Run the optional Wav2Lip stage; return output_path only on success."""
    global CURRENT_LOG
    repo = os.path.abspath(os.path.join("tools", "Wav2Lip"))
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
        "--pads", "0", "20", "0", "0",
        "--nosmooth",
    ]
    CURRENT_LOG += "Running Wav2Lip on the concatenated video...\n"
    result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True)
    if result.stdout:
        CURRENT_LOG += result.stdout[-4000:] + "\n"
    if result.returncode != 0 or not os.path.exists(output_path):
        CURRENT_LOG += "Wav2Lip failed; continuing without lip-sync.\n"
        if result.stderr:
            CURRENT_LOG += result.stderr[-4000:] + "\n"
        return None
    CURRENT_LOG += f"Lip-sync complete: {output_path}\n"
    return output_path

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
        if LYRICS_ENABLED and (not lyrics_path or not lyrics_segments):
            generated_path, asr_warning = transcribe_lyrics_for_audio(audio_path)
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
        
        num_scenes = math.ceil(duration_sec / scene_duration_sec)
        num_scenes = min(num_scenes, 20) # Limit to 20 scenes
        CURRENT_TOTAL_SCENES = num_scenes
        CURRENT_PHASE = f"Audio segmented into {num_scenes} scenes"
        CURRENT_PROGRESS = 0.0
        
        new_scenes_data = [None] * 20
        ui_updates = []
        
        boundaries, downbeat_count = beat_aligned_boundaries(
            waveform, sr, duration_sec, scene_duration_sec, num_scenes
        )

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
        return tuple([f"Sliced into {num_scenes} beat-aligned scenes (BPM analysis: {downbeat_count} downbeats).{lyrics_note}"] + ui_updates)
        
    except Exception as e:
        print(f"DEBUG Error in slice_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return tuple([f"Error: {str(e)}"] + [gr.update(visible=False)] * 20 * 5) # Update this count if UI structure changes

def slice_speech(audio_path, image_path, prompt, fps, num_frames):
    """Transcribe a speech track and prepare talking-head scenes from one reference image."""
    global SCENES_DATA, CURRENT_PHASE, CURRENT_PROGRESS, CURRENT_TOTAL_SCENES, MUSIC_AUDIO_PATH
    if not audio_path:
        return tuple(["Upload a speech audio file first."] + [gr.update(visible=False)] * 20 * 5)
    if not image_path:
        return tuple(["Upload a base speaker image first."] + [gr.update(visible=False)] * 20 * 5)
    try:
        os.makedirs(AUDIO_CLIPS_DIR, exist_ok=True)
        waveform, sr, duration_sec = load_audio_compatible(audio_path)
        MUSIC_AUDIO_PATH = os.path.abspath(audio_path)
        CURRENT_PHASE = "Preparing speech transcription"
        CURRENT_PROGRESS = 0.5
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
            message = warning or "No speech was detected in the audio."
            return tuple([message] + [gr.update(visible=False)] * 20 * 5)

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
        CURRENT_PHASE = f"Speech sliced into {len(groups)} dialogue scenes"
        CURRENT_PROGRESS = 4.0
        new_scenes_data = [None] * 20
        ui_updates = []
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
                "duration_sec": max(0.1, end_sec - start_sec),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "lyrics_text": speech_text,
                "first_frame": None,
                "last_frame": None,
            }
        SCENES_DATA = new_scenes_data
        for i in range(20):
            if i < len(groups):
                ui_updates.extend([
                    gr.update(visible=True),
                    gr.update(value=new_scenes_data[i]["prompt"], visible=True),
                    gr.update(value=new_scenes_data[i]["audio_path"]),
                    gr.update(value=image_path),
                    gr.update(value=None),
                ])
            else:
                ui_updates.extend([
                    gr.update(visible=False), gr.update(visible=False), gr.update(value=None),
                    gr.update(value=None), gr.update(value=None),
                ])
        return tuple([f"Speech transcribed and sliced into {len(groups)} scenes. Same image assigned to every scene; replace a later Start Image to change the speaker."] + ui_updates)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return tuple([f"Speech slice error: {exc}"] + [gr.update(visible=False)] * 20 * 5)

def process_chain_generation(scenes_list, audios_list, start_images_list, last_images_list, checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth, enable_lipsync=False, start_index=None, mode="forward"):
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
        output_filename = f"scene_{scene_id}_{timestamp}.mp4"
        output_path = os.path.abspath(output_filename)
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
                    
                    f_prev_l1 = f"scene_{scene_id}_f_prev_last1.jpg"
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
        
        if audio_path:
            cmd.extend(["--audio-input-path", audio_path])
            
        for frame_path, latent_idx, guidance in conditioning_frames:
            cmd.extend(["--image", frame_path, str(latent_idx), str(guidance)])

        try:
            CURRENT_PROCESS = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )
            for line in CURRENT_PROCESS.stdout:
                if STOP_GENERATION:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(CURRENT_PROCESS.pid)], capture_output=True)
                    break
                CURRENT_LOG += line
            CURRENT_PROCESS.wait()
            
            if CURRENT_PROCESS.returncode == 0 and os.path.exists(output_path):
                CURRENT_LOG += f"Scene {scene_id} Complete.\n"
                LATEST_VIDEO_PATH = output_path
                
                # Update SCENES_DATA
                first_f = f"scene_{scene_id}_first.jpg"
                last_f = f"scene_{scene_id}_last.jpg"
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

    # After a complete forward chain, concatenate, optionally lip-sync, then upscale 3x.
    if mode == "forward" and not STOP_GENERATION:
        completed_paths = [SCENES_DATA[i].get("video_path") for i in valid_indices if SCENES_DATA[i]]
        paths_ready = (
            completed_paths
            and len(completed_paths) == len(valid_indices)
            and all(p is not None and os.path.exists(p) for p in completed_paths)
        )
        if paths_ready:
            try:
                CURRENT_PHASE = "Concatenating scene videos"
                CURRENT_PROGRESS = 90.0
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                concat_list = os.path.abspath(f"music_video_concat_{stamp}.txt")
                final_video = os.path.abspath(f"music_video_final_{stamp}.mp4")
                upscaled_video = os.path.abspath(f"music_video_final_{stamp}_x3.mp4")
                with open(concat_list, "w", encoding="ascii") as list_file:
                    for path in completed_paths:
                        list_file.write(f"file '{path.replace(chr(92), '/')}'\n")

                ffmpeg = os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
                CURRENT_LOG += f"\nConcatenating {len(completed_paths)} scenes...\n"
                concat_proc = subprocess.run(
                    [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                     "-c", "copy", final_video], capture_output=True, text=True
                )
                if concat_proc.returncode != 0:
                    raise RuntimeError(concat_proc.stderr[-2000:])

                video_for_upscale = final_video
                if enable_lipsync:
                    CURRENT_PHASE = "Lip-sync (Wav2Lip)"
                    CURRENT_PROGRESS = 93.0
                    lip_audio = MUSIC_AUDIO_PATH or completed_paths[0]
                    lipsync_video = os.path.abspath(f"music_video_final_{stamp}_lipsync.mp4")
                    synced = run_wav2lip(final_video, lip_audio, lipsync_video)
                    if synced:
                        video_for_upscale = synced
                    else:
                        CURRENT_LOG += "Lip-sync unavailable or no face detected; using original video.\n"

                CURRENT_PHASE = "Upscaling final video 3x"
                CURRENT_PROGRESS = 96.0
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
    IS_PROCESSING = False

def start_generation_thread(prompts, audios, start_images, last_images, checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth, enable_lipsync=False, start_index=None, mode="forward"):
    global CURRENT_PHASE, CURRENT_PROGRESS
    CURRENT_PHASE = "Queued"
    CURRENT_PROGRESS = 0.0
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

    threading.Thread(target=process_chain_generation, args=(prompts, audios, start_images, last_images, checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth, enable_lipsync, start_index, mode), daemon=True).start()
    return "Generation started..."

def stop_generation():
    global STOP_GENERATION
    STOP_GENERATION = True
    return "Stopping..."

def update_ui():
    global LATEST_VIDEO_PATH, CURRENT_LOG, SCENES_DATA, CURRENT_OUTPUT_PATH, CURRENT_SCENE_INDEX
    global CURRENT_PHASE, CURRENT_PROGRESS
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
            
    return tuple([LATEST_VIDEO_PATH, CURRENT_LOG, status] + updates)

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

# --- UI Layout ---

theme = gr.themes.Soft(primary_hue="purple").set(
    body_background_fill="*neutral_50",
    block_background_fill="*neutral_100",
)

with gr.Blocks(title="LTX-2 Music Video Maker") as demo:
    gr.Markdown("# 🎵 LTX-2 Music Video Maker")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_file = gr.Audio(label="Upload Music File", type="filepath")
            master_prompt = gr.Textbox(label="Visual Style / Prompt", placeholder="Cyberpunk city, neon lights, rain...", lines=2)
            lyrics_enabled = gr.Checkbox(
                label="Use lyrics_timing.json in scene prompts",
                info="Adds ASR lyric text for each beat-aligned scene. Re-slice after changing.",
                value=True,
            )
            enable_lipsync = gr.Checkbox(
                label="Enable real lip-sync (Wav2Lip)",
                info="Runs Wav2Lip after concatenation; requires wav2lip_gan.pth and s3fd.pth.",
                value=False,
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
    
    with gr.Accordion("Worker Log", open=False):
        log_box = gr.Textbox(label=None, lines=10, interactive=False)

    # --- Events ---
    slice_btn.click(
        fn=slice_audio,
        inputs=[audio_file, master_prompt, fps, num_frames, lyrics_enabled],
        outputs=[status_box] + [comp for tuple_5 in zip(scene_rows, scene_prompts, scene_audios, scene_start_images, scene_last_images) for comp in tuple_5]
    )
    
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

    all_inputs = scene_prompts + scene_audios + scene_start_images + scene_last_images + [checkpoint, gemma, upsampler, steps, fps, width, height, num_frames, seed, random_seed, use_context_compression, latent_reuse_count, context_depth, enable_lipsync]

    speech_slice_btn.click(
        fn=slice_speech,
        inputs=[speech_audio, speech_image, speech_prompt, fps, num_frames],
        outputs=[speech_status] + [comp for tuple_5 in zip(scene_rows, scene_prompts, scene_audios, scene_start_images, scene_last_images) for comp in tuple_5]
    )

    speech_generate_btn.click(
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
        outputs=[latest_video, log_box, status_box] + [comp for zip_list in zip(scene_videos, scene_previews) for comp in zip_list]
    )

    # Pos-producao: diagnostico e correcao temporal do video ja gerado.
    # Fica FORA do fluxo de geracao -- ferramenta aplicada ao resultado.
    video_doctor_ui.build_doctor_tab(label="Diagnostico e correcao (video doctor)",
                                     container="accordion")

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("LTX_UI_HOST", "127.0.0.1"), server_port=7801, theme=theme)
