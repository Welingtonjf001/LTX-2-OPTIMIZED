"""Stage [5]: render one LTX video clip per dialogue line (shot-per-speaker; v1 does
not attempt multi-character-speaking-at-once shots -- see plan), or one clip for the
whole scene when it has no dialogue (establishing/action-only beats).

Uses the plain native pipeline (``python -m ltx_pipelines.music_to_video``), the same
call ``process_chain_generation`` in music_maker_ui_v2.py already makes -- NOT the
TensorRT two-stage route built for tensorxx_ge/webui_v2/v3.py. v1 deliberately keeps
this simpler: no TensorRT engine file is required as a dependency for the screenplay
pipeline to run end to end. Swapping in the TensorRT route later (reusing
tensorxx_ge.run_ltx_with_tensorrt) is a straightforward v2 upgrade, not a rewrite.

Each clip's prompt = the scene's storyboard visual_prompt, elaborated for dialogue
clips with a close-up framing note + the line's text + the speaking character's
descriptor. The storyboard PNG is passed as image conditioning (continuity/style
reference, not an identity lock). Duration comes from the line's synthesized audio
(dialogue clips) or a configurable default (action-only scenes), snapped to LTX's
8k+1 frame grid -- the exact formula already used in process_chain_generation.

MEASURED (2026-08-09 smoke test): a single continuous 401-frame (16.8s) clip stalls
the spatial upsampler stage for 10+ minutes with the RTX 3090 pinned at ~24.3/24.5GB
and 0% incremental progress (stage 1 low-res denoise itself is fine, ~177s even for
this worst case -- the upsampler is what can't handle a long sequence in that VRAM
budget). --max-clip-seconds splits any line's audio longer than that cap into several
sub-clips (ffmpeg -ss/-t on a scratch copy) rendered as separate consecutive jobs;
assemble_final.py already concatenates every clip in script order, so no extra
merge step is needed -- the split segments just play back-to-back. Visual continuity
across a split isn't guaranteed frame-exact (each segment is a fresh generation
anchored to the same storyboard image, no IP-Adapter/face-lock in v1 -- see plan),
same tradeoff already accepted for cross-scene identity.

Runs on GPU 1 (RTX 3090), matching every other native-pipeline launcher in this
project. Frees ComfyUI's VRAM first (POST /free) since ComfyUI stays resident on the
same GPU after generate_storyboards.py for fast reuse -- it would otherwise compete
with LTX for VRAM here.

MEASURED (2026-08-09, same session): the two-stage pipeline (base gen + spatial
upsampler) requires width/height to both be multiples of 64 -- assert_resolution()
in ltx_pipelines raises otherwise. 896x512 keeps the project's usual 1.75:1 aspect
ratio (same as 1344x768) while satisfying that constraint (896/64=14, 512/64=8).

MEASURED (2026-08-09, same session, part 2): at 448x256, lip-sync failed 15/15
("Face not detected") -- clips open on a WIDE shot (the storyboard image conditions
the whole clip, not just the close-up dialogue framing the prompt asks for) and only
drift into close-up over several seconds; at 448x256 that opening wide-shot face is
too few pixels for LatentSync's detector. 896x512 (4x the pixels of 448x256, still
far cheaper than the original 1344x768) gives the same wide-shot face enough
resolution to detect. Root cause (mismatched wide-shot image anchor vs close-up
prompt) isn't fixed here -- see plan's per-shot-type storyboard as the real v2 fix --
this is the resolution-side mitigation.

CLI: python -m script_pipeline.render_scenes --run-dir DIR
     [--checkpoint ...] [--gemma-root ...] [--upsampler ...]
     [--width 448] [--height 256] [--fps 24] [--steps 8]
     [--max-clip-seconds 5.0] [--default-scene-seconds 4.0] [--seed 1234]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CHECKPOINT = "./models/ltx-2.3-22b-distilled-fp8.safetensors"
DEFAULT_GEMMA = "./models/gemma3"
DEFAULT_UPSAMPLER = "./models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

# Which module renders a clip. Default keeps the 2.3 native pipeline exactly as
# before; setting LTX_PIPELINE_MODULE=ltx_pipelines_25 swaps in the LTX-2.5 shim
# (ComfyUI route -- ltx_pipelines has no 2.5 support, see MEMORIAL.md). The shim
# accepts this same argv, ignoring the flags that don't apply to 2.5, so the whole
# screenplay chain (screenplay_ui -> screenplay_to_video -> here) runs 2.5 without
# duplicating any of it. start_screenplay_25.bat sets this.
LTX_PIPELINE_MODULE = os.environ.get("LTX_PIPELINE_MODULE", "ltx_pipelines.music_to_video")

# Wan is not distilled the way the LTX checkpoint here is, so it runs with real CFG
# and therefore actually uses a negative prompt (the LTX path has none).
WAN_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, extra limbs, extra fingers, "
    "text, watermark, logo, subtitles, static image, jpeg artifacts"
)

# LTX 2.3's prompt guide asks for sound in three layers -- ambience/room tone, voice,
# and effects. We previously described none, so the model had nothing to aim at for the
# audio it generates alongside the picture.
AMBIENT_AUDIO_HINT = "natural ambient room tone and environmental sound of the location"
DIALOGUE_AUDIO_HINT = "clear intelligible speech in the foreground, quiet ambience behind it"


def scene_setting(scene: dict) -> str:
    """A short, clean environment phrase for the prompt's environment slot.

    Deliberately NOT scene["heading_raw"]: for a freeform screenplay that field held
    the raw paragraph truncated mid-word, which is exactly the debris that used to
    open every prompt (see compose_prompt). Prefer the structured location/time, and
    fall back to a real slugline only when one was actually parsed.
    """
    parts = [p for p in (scene.get("location"), scene.get("time_of_day")) if p and p.strip()]
    if parts:
        return ", ".join(parts)
    heading = (scene.get("heading_raw") or "").strip()
    return heading if heading and not heading.endswith("...") else ""


def _load_scenes(run_dir: Path) -> list[dict]:
    parse_dir = run_dir / "parse"
    enriched = parse_dir / "scenes_enriched.json"
    structural = parse_dir / "scenes.json"
    path = enriched if enriched.exists() else structural
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cast(run_dir: Path) -> dict:
    cast_path = run_dir / "characters" / "cast.json"
    return json.loads(cast_path.read_text(encoding="utf-8")) if cast_path.exists() else {}


def _load_lines(run_dir: Path) -> dict:
    """Return {(scene_index, line_index): line_entry} from dialogue/lines.json."""
    lines_path = run_dir / "dialogue" / "lines.json"
    if not lines_path.exists():
        return {}
    entries = json.loads(lines_path.read_text(encoding="utf-8"))
    return {(e["scene_index"], e["line_index"]): e for e in entries}


def normalize_wan_frames(value) -> int:
    """Wan's latent grid is 4k+1 (WanImageToVideo computes ((length-1)//4)+1 latent
    frames), not LTX's 8k+1 -- see normalize_ltx_frames for the LTX counterpart."""
    raw = max(5, int(round(value)))
    return 4 * max(1, round((raw - 1) / 4)) + 1


def render_job_wan(
    job: dict, *, output_path: Path, log, width: int, height: int, fps: int,
    steps: int, cfg: float, seed: int, default_scene_seconds: float,
    checkpoint: str, clip_name: str, vae_name: str, weight_dtype: str,
    sampler: str, scheduler: str, server: str,
    chain_image: str | None = None, end_image: str | None = None,
) -> bool:
    """Render one clip through ComfyUI's Wan nodes instead of ltx_pipelines.

    Everything upstream (shot list, action beats, chain continuity, prompts) and
    downstream (lip-sync, mix, assemble) is model-agnostic and shared -- only the
    generation call differs, which is why this lives beside the LTX path rather than
    in a forked pipeline.

    Two differences from the LTX path that callers should know about:
      * Wan generates SILENT video (no audio branch at all, unlike LTX 2.3 which
        conditions on and emits audio). The line's TTS audio is muxed in afterwards
        with ffmpeg, so downstream stages see the same "clip with audio" they expect.
      * With both a start and an end image, this uses WanFirstLastFrameToVideo, which
        pins both ends of the clip -- a stronger continuity guarantee than the LTX
        path's start-frame conditioning plus optional end keyframe.
    """
    from script_pipeline.generate_storyboards import (
        _fill_template, COMFYUI_OUTPUT_DIR, submit_and_wait,
    )

    duration_sec = job["duration_sec"] if job["duration_sec"] is not None else default_scene_seconds
    length = normalize_wan_frames(duration_sec * fps)

    comfy_input = ROOT / "ComfyUI" / "input"
    comfy_input.mkdir(parents=True, exist_ok=True)

    def _stage_image(path: str, tag: str) -> str:
        """LoadImage reads by filename from ComfyUI/input, so stage a copy there."""
        staged_name = f"_screenplay_{job['id']}_{tag}.png"
        shutil.copy2(path, comfy_input / staged_name)
        return staged_name

    start_image = chain_image or job["storyboard_path"]
    if not start_image:
        log(f"{job['id']}: Wan exige uma imagem inicial (storyboard ou frame anterior); pulando.")
        return False

    # MEASURED (2026-08-10) by reading ComfyUI's own node schemas: Wan 2.2's 5B and
    # 14B lines are DIFFERENT architectures, not variants of one graph.
    #   5B  -> Wan22ImageToVideoLatent: 48 latent channels, /16 spatial, wan2.2_vae,
    #          start_image only (no end_image), and it emits ONLY a latent -- the text
    #          conditioning goes straight from CLIPTextEncode into KSampler.
    #   14B -> WanImageToVideo / WanFirstLastFrameToVideo: 16 latent channels, /8
    #          spatial, wan_2.1_vae, and they pass positive/negative through.
    # Picking the wrong graph silently produces a shape mismatch, so route on the
    # model family rather than on which options the caller asked for.
    is_5b = "5b" in Path(checkpoint).stem.lower()
    use_flf = bool(end_image) and not is_5b
    if is_5b:
        if end_image:
            log(f"{job['id']}: Wan 5B nao aceita frame final (Wan22ImageToVideoLatent so tem start_image); ignorando end_image.")
        template_name = "wan22_ti2v_5b.json"
    else:
        template_name = "wan_flf2v.json" if use_flf else "wan_i2v.json"
    values = {
        "CHECKPOINT": checkpoint, "CLIP_NAME": clip_name, "VAE_NAME": vae_name,
        "WEIGHT_DTYPE": weight_dtype, "WIDTH": width, "HEIGHT": height,
        "LENGTH": length, "FPS": float(fps), "SEED": seed, "STEPS": steps,
        "CFG": cfg, "SAMPLER": sampler, "SCHEDULER": scheduler,
        "POSITIVE_PROMPT": job["prompt"], "NEGATIVE_PROMPT": WAN_NEGATIVE_PROMPT,
        "START_IMAGE": _stage_image(start_image, "start"),
        "FILENAME_PREFIX": f"wan_{job['id']}",
    }
    if use_flf:
        values["END_IMAGE"] = _stage_image(end_image, "end")

    template = json.loads((ROOT / "comfyui_workflows" / template_name).read_text(encoding="utf-8"))
    workflow = _fill_template(template, values)

    log(f"{job['id']}: [Wan] {length} frames ({duration_sec:.1f}s @ {fps}fps), "
        f"{'primeiro+ultimo frame' if use_flf else 'frame inicial'}, audio={'sim (muxado)' if job['audio_path'] else 'nao'}")
    entry = submit_and_wait(server, workflow, log=log, timeout=1800)
    if entry is None:
        log(f"{job['id']}: FALHOU (ComfyUI/Wan).")
        return False

    produced = None
    for _node_id, out in (entry.get("outputs") or {}).items():
        for key in ("videos", "images", "gifs"):
            for item in out.get(key, []) or []:
                if item.get("filename"):
                    produced = COMFYUI_OUTPUT_DIR / item.get("subfolder", "") / item["filename"]
                    break
            if produced:
                break
        if produced:
            break
    if produced is None or not Path(produced).exists():
        log(f"{job['id']}: Wan concluiu mas nenhum arquivo de video foi encontrado.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if job["audio_path"]:
        # Wan output is silent -- mux the line's TTS audio so downstream stages
        # (lipsync/mix/assemble) get the same shape of clip the LTX path produces.
        ffmpeg = os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
        result = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", str(produced), "-i", job["audio_path"],
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
             "-shortest", str(output_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0 or not output_path.exists():
            log(f"{job['id']}: falha ao muxar audio ({result.stderr[-300:]}); mantendo video mudo.")
            shutil.copy2(produced, output_path)
    else:
        shutil.copy2(produced, output_path)

    log(f"{job['id']}: ok -> {output_path}")
    return True


def normalize_ltx_frames(value) -> int:
    """Snap to LTX's 8k+1 latent grid -- same formula as process_chain_generation."""
    raw = max(9, int(round(value)))
    return 8 * max(1, round((raw - 1) / 8)) + 1


def _ensure_stereo(audio_path: str, work_dir: Path) -> str:
    """ltx_pipelines.music_to_video's audio muxing requires 2-channel input (MEASURED:
    it raises "Expected samples with 2 channels" on our mono dialogue WAVs). The
    dialogue WAVs must stay mono for lip-sync's sake (LatentSync/Wav2Lip want mono),
    so convert a scratch COPY here rather than touching the original."""
    import soundfile as sf

    info = sf.info(audio_path)
    if info.channels >= 2:
        return audio_path
    work_dir.mkdir(parents=True, exist_ok=True)
    stereo_path = work_dir / (Path(audio_path).stem + "_stereo.wav")
    ffmpeg = os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
    result = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", audio_path, "-ac", "2", str(stereo_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0 or not stereo_path.exists():
        raise RuntimeError(f"Failed to convert {audio_path} to stereo: {result.stderr}")
    return str(stereo_path)


def _silence_gaps(audio_path: str, *, top_db: float = 35.0) -> list[tuple[float, float]]:
    """Trechos de silencio dentro do audio, em segundos: [(inicio, fim), ...].

    MEDIDO 2026-08-29: `_split_audio_into_segments` cortava em pontos
    ARITMETICOS (duracao_total / n), sem nenhuma nocao de onde a fala para --
    cortava no meio de palavra em toda fronteira interna, e cada pedaco virava
    um clipe + lip-sync INDEPENDENTE. E a causa de "truncam-se as falas, perde
    o sincronismo" reportada numa fala de 22s que virou 5 segmentos de
    4,52s cada. `librosa.effects.split` acha os intervalos NAO-silenciosos;
    os gaps entre eles sao onde cortar sem partir uma silaba."""
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    intervals = librosa.effects.split(y, top_db=top_db)  # [[start_sample, end_sample], ...]
    if len(intervals) == 0:
        return []
    gaps = []
    total_samples = len(y)
    if intervals[0][0] > 0:
        gaps.append((0.0, intervals[0][0] / sr))
    for (_, end_prev), (start_next, _) in zip(intervals[:-1], intervals[1:]):
        gaps.append((end_prev / sr, start_next / sr))
    if intervals[-1][1] < total_samples:
        gaps.append((intervals[-1][1] / sr, total_samples / sr))
    return gaps


def _boundaries_from_gaps(gaps: list[tuple[float, float]], total_duration: float,
                          max_seconds: float) -> list[float]:
    """Fronteiras de corte que respeitam o teto `max_seconds` por segmento E
    caem em silencio real sempre que existe um dentro do orcamento.

    Percorre os silencios em ordem; sempre que ESPERAR pelo proximo faria o
    segmento atual estourar o teto, corta no ultimo silencio que ainda coube
    -- greedy, mas o teto de duracao e a garantia dura (o motivo de
    `max_clip_seconds` existir: um clipe longo demais trava o upsampler,
    MEDIDO no docstring do modulo). So cai no corte aritmetico quando um
    trecho de fala corrida excede o teto sem NENHUM silencio dentro dele."""
    centros = sorted((a + b) / 2.0 for a, b in gaps)
    boundaries = [0.0]
    cursor = 0.0
    while total_duration - cursor > max_seconds:
        candidatos = [c for c in centros if cursor < c <= cursor + max_seconds]
        if candidatos:
            proximo = candidatos[-1]  # o mais tarde ainda dentro do teto -- segmento maior, corte melhor
        else:
            proximo = cursor + max_seconds  # sem silencio no trecho: cai no corte aritmetico
        boundaries.append(proximo)
        cursor = proximo
    boundaries.append(total_duration)
    return boundaries


def _split_audio_into_segments(
    audio_path: str, total_duration: float, max_seconds: float, work_dir: Path,
) -> list[tuple[str, float]]:
    """Cut a dialogue-line WAV into <= max_seconds chunks (ffmpeg -ss/-t on a scratch
    copy, original untouched) so each chunk renders as its own short LTX clip instead
    of one long stall-prone generation. Returns [(chunk_path, chunk_duration), ...]
    in playback order.

    Os cortes caem em silencio real sempre que existe um dentro do orcamento
    de `max_seconds` -- ver `_boundaries_from_gaps`. Sem silencio nenhum perto
    (fala corrida), cai no corte aritmetico de antes: nunca pior que o
    comportamento original, so melhor quando ha onde cortar certo."""
    work_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
    stem = Path(audio_path).stem

    if total_duration <= max_seconds:
        boundaries = [0.0, total_duration]
    else:
        try:
            gaps = _silence_gaps(audio_path)
            boundaries = _boundaries_from_gaps(gaps, total_duration, max_seconds)
        except Exception:
            n_segments = max(1, math.ceil(total_duration / max_seconds))
            seg_len = total_duration / n_segments
            boundaries = [i * seg_len for i in range(n_segments + 1)]

    segments = []
    for i in range(len(boundaries) - 1):
        start, stop = boundaries[i], boundaries[i + 1]
        dur = stop - start
        seg_path = work_dir / f"{stem}_seg{i:02d}.wav"
        result = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", audio_path, "-ss", f"{start:.3f}",
             "-t", f"{dur:.3f}", str(seg_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0 or not seg_path.exists():
            raise RuntimeError(f"Falha ao dividir {audio_path} (segmento {i}): {result.stderr}")
        segments.append((str(seg_path), dur))
    return segments


def extract_last_frame(video_path: str, output_path: Path) -> bool:
    """Same OpenCV frame-grab already validated in music_maker_ui_v2.py's chain
    generation (extract_frame/extract_last_frame there). Used to feed each clip's
    final frame as the NEXT clip's starting image, instead of always resetting to
    the scene's static storyboard -- gives genuine action continuity between
    consecutive clips (a hand mid-reach doesn't snap back to its storyboard pose)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    success, frame = cap.read()
    cap.release()
    if not success:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), frame))


def free_comfyui_vram(server: str = "http://127.0.0.1:8188", *, log=print) -> None:
    try:
        data = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
        req = urllib.request.Request(f"{server}/free", data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        log("ComfyUI: memoria liberada (SDXL descarregado) antes do render LTX.")
    except urllib.error.URLError:
        pass  # ComfyUI not running -- nothing to free.
    except Exception as exc:  # noqa: BLE001
        log(f"ComfyUI: falha ao liberar memoria (seguindo mesmo assim): {exc}")


def build_render_plan(
    scenes: list[dict], cast: dict, lines_by_key: dict, storyboard_dir: Path, *,
    max_clip_seconds: float | None = None, split_dir: Path | None = None,
    camera_movement: str | None = None, action_beat_seconds: float = 3.0,
) -> list[dict]:
    """One render job per dialogue line (or several, if it's longer than
    max_clip_seconds and gets split), or one per scene when it has no dialogue.

    camera_movement: free-text camera direction (e.g. "slow push in", "handheld"),
    applied to every clip in this run -- same preset vocabulary as the Prompt
    Composer's camera-movement checkboxes in music_maker_ui_v2.py, but as a single
    global choice per render rather than per-scene, since v1 has no reliable way to
    ask the LLM enrichment for a different movement per scene (that model's JSON
    output is often unavailable -- see parse_screenplay.py's enrichment fallback).
    None/empty leaves clips exactly as before (no camera-direction clause added)."""
    camera_clause = f" Camera movement: {camera_movement}." if camera_movement else ""
    jobs = []

    def compose_prompt(*, action: str, characters: str = "", setting: str = "",
                       speech: str = "", audio: str = "") -> str:
        """Assemble one clip prompt in the order LTX 2.3's own prompt guide asks for:
        main action first, then motion/character detail, then environment, with camera
        and audio LAST -- and as a single present-tense paragraph.

        MEASURED (2026-08-11) against a real run: prompts used to OPEN with
        scene["heading_raw"], which for a freeform screenplay was the raw paragraph cut
        at 80 characters mid-word ("cena - no ponto de onibus a jovem - Park Min - ...
        aguarda o ..."). Every clip therefore began with a truncated Portuguese
        fragment, in a different language from the rest of the prompt, and dialogue
        clips repeated that same debris at the end via the character descriptor. That
        is a direct cause of incoherent shots -- the model spent the most heavily
        weighted part of the prompt parsing garbage. Empty parts are dropped rather
        than emitted as stray punctuation."""
        parts = [p.strip().rstrip(".") for p in (action, speech, characters, setting) if p and p.strip()]
        prompt = ". ".join(parts)
        if prompt:
            prompt += "."
        prompt += camera_clause
        if audio:
            prompt += f" Audio: {audio.strip().rstrip('.')}."
        return prompt.strip()
    for scene in scenes:
        storyboard_path = storyboard_dir / f"scene_{scene['index']:02d}.png"
        storyboard_str = str(storyboard_path) if storyboard_path.exists() else None
        dialogue = scene.get("dialogue", [])
        if not dialogue:
            duration = None
            if max_clip_seconds:
                duration = max_clip_seconds  # clamps the default-scene-seconds fallback too
            jobs.append({
                "id": f"scene{scene['index']:02d}",
                "scene_index": scene["index"], "line_index": None,
                "prompt": compose_prompt(
                    action=scene.get("visual_prompt") or scene.get("action_text", ""),
                    setting=scene_setting(scene),
                    audio=AMBIENT_AUDIO_HINT,
                ),
                "storyboard_path": storyboard_str,
                "audio_path": None, "duration_sec": duration,
            })
            continue

        setting_anchor = scene_setting(scene)

        # Walk the scene's shot list IN ORDER (see Scene.shot_list in
        # parse_screenplay.py). Falls back to dialogue-only, in script order, for
        # scenes parsed before this field existed or when enrichment produced nothing.
        shot_list = scene.get("shot_list") or [
            {"type": "dialogue", "line_index": i} for i in range(len(dialogue))
        ]
        shot_position = 0

        for shot in shot_list:
            shot_position += 1
            # Per-shot storyboard when generate_storyboards.py produced one (its
            # --per-shot mode, the default); otherwise the scene-wide image.
            per_shot_path = storyboard_dir / f"scene_{scene['index']:02d}_s{shot_position:03d}.png"
            shot_storyboard = str(per_shot_path) if per_shot_path.exists() else storyboard_str
            if shot.get("type") == "action":
                visual = str(shot.get("visual") or "").strip()
                if not visual:
                    continue
                jobs.append({
                    # Zero-padded position keeps clips.json / the scenes folder sorting
                    # in the same order the film actually plays.
                    "id": f"scene{scene['index']:02d}_s{shot_position:03d}_act",
                    "scene_index": scene["index"], "line_index": None,
                    "prompt": compose_prompt(
                        action=visual, setting=setting_anchor, audio=AMBIENT_AUDIO_HINT,
                    ),
                    "storyboard_path": shot_storyboard,
                    # No dialogue -> no TTS audio; duration is the caller's
                    # action_beat_seconds, capped by max_clip_seconds like any clip.
                    "audio_path": None,
                    "duration_sec": min(action_beat_seconds, max_clip_seconds) if max_clip_seconds else action_beat_seconds,
                })
                continue

            line_index = shot.get("line_index")
            if line_index is None or not (0 <= line_index < len(dialogue)):
                continue
            line = dialogue[line_index]
            key = (scene["index"], line_index)
            line_entry = lines_by_key.get(key)
            character = line["character"]
            descriptor = cast.get(character, {}).get("descriptor", "")
            beat_visual = line.get("beat_visual")
            # MEASURED (2026-08-10): every clip in a scene used to get the SAME flat
            # action_text, so nothing told the model what changed at THIS line.
            # beat_visual is the per-line "what's happening right now" description.
            action = beat_visual or scene.get("visual_prompt") or scene.get("action_text", "")
            # The guide asks for emotion as a VISIBLE physical cue rather than an
            # abstract label, and for dialogue in quotes with a short acting direction
            # attached to the line -- not a bare quoted string.
            delivery = line.get("parenthetical") or line.get("emotion") or ""
            speech = (
                f'Close-up on {character}, speaking directly to camera with natural mouth '
                f'articulation, saying: "{line["text"]}"'
            )
            if delivery:
                speech += f", delivered {delivery}"
            prompt = compose_prompt(
                action=action, speech=speech, characters=descriptor,
                setting=setting_anchor, audio=DIALOGUE_AUDIO_HINT,
            )
            audio_path = line_entry["audio_path"] if (line_entry and line_entry.get("ok")) else None
            duration = line_entry["duration_sec"] if (line_entry and line_entry.get("ok")) else None
            base_id = f"scene{scene['index']:02d}_s{shot_position:03d}_line{line_index:02d}"

            if audio_path and duration and max_clip_seconds and duration > max_clip_seconds:
                segments = _split_audio_into_segments(audio_path, duration, max_clip_seconds, split_dir)
                for seg_i, (seg_path, seg_dur) in enumerate(segments):
                    jobs.append({
                        "id": f"{base_id}_seg{seg_i:02d}", "scene_index": scene["index"],
                        "line_index": line_index, "character": character, "prompt": prompt,
                        "storyboard_path": shot_storyboard,
                        "audio_path": seg_path, "duration_sec": seg_dur,
                    })
            else:
                jobs.append({
                    "id": base_id, "scene_index": scene["index"], "line_index": line_index,
                    "character": character, "prompt": prompt, "storyboard_path": shot_storyboard,
                    "audio_path": audio_path, "duration_sec": duration,
                })
    return jobs


def _run_streamed(command: list[str], *, env: dict, log) -> int:
    log(f"Comando: {' '.join(command)}")
    process = subprocess.Popen(
        command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
        encoding="utf-8", errors="replace",
    )
    for line in process.stdout:
        log(line.rstrip("\n"))
    process.wait()
    return process.returncode


def render_job(
    job: dict, *, output_path: Path, checkpoint: str, gemma_root: str, upsampler: str,
    width: int, height: int, fps: int, steps: int, seed: int, default_scene_seconds: float,
    log, chain_image: str | None = None, end_image: str | None = None,
    end_image_strength: float = 0.5, generate_ambient_audio: bool = False,
) -> bool:
    duration_sec = job["duration_sec"] if job["duration_sec"] is not None else default_scene_seconds
    num_frames = normalize_ltx_frames(duration_sec * fps)

    command = [
        sys.executable, "-u", "-m", LTX_PIPELINE_MODULE,
        "--distilled-checkpoint-path", checkpoint,
        "--gemma-root", gemma_root,
        "--spatial-upsampler-path", upsampler,
        "--prompt", job["prompt"],
        "--output-path", str(output_path),
        "--width", str(width), "--height", str(height),
        "--num-frames", str(num_frames), "--frame-rate", str(fps),
        "--num-inference-steps", str(steps), "--seed", str(seed),
        "--quantization", "fp8-cast",
    ]
    # chain_image (the previous clip's last frame, when chaining is on) takes
    # priority over the scene's static storyboard -- see extract_last_frame()'s
    # docstring. Same "0 0.8" slot/strength as the storyboard-conditioned case.
    image_path = chain_image or job["storyboard_path"]
    if image_path:
        command += ["--image", image_path, "0", "0.8"]
    # --image takes PATH FRAME_IDX STRENGTH and may be repeated: frame_idx 0 becomes
    # VideoConditionByLatentIndex (replaces the opening latent) while any other index
    # becomes VideoConditionByKeyframeIndex (see combined_image_conditionings in
    # ltx_pipelines/utils/helpers.py). Pinning a keyframe near the END as well as the
    # start gives the clip a destination instead of letting it drift wherever it
    # likes, which is what makes consecutive chained clips read as one continuous
    # action. Kept at a LOWER strength than the opening frame on purpose: the end
    # frame is a target to move toward, not a still to freeze on.
    if end_image:
        command += ["--image", end_image, str(max(1, num_frames - 1)), str(end_image_strength)]
    if job["audio_path"]:
        stereo_audio = _ensure_stereo(job["audio_path"], output_path.parent / "_stereo_audio")
        command += ["--audio-input-path", stereo_audio]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"  # RTX 3090, same convention as start_music_video_*.bat
    env.setdefault("LTX_TRANSFORMER_GPU_MEMORY", "18GiB")
    env.setdefault("LTX_TRANSFORMER_CPU_MEMORY", "32GiB")
    env.setdefault("LTX_TEXT_ENCODER_GPU_MEMORY", "4GiB")
    env.setdefault("LTX_UPSAMPLER_GPU_MEMORY", "18GiB")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    # Ambient audio for wordless shots: LTX denoises an audio latent for every clip,
    # but ltx_pipelines only vocodes it when an input audio file was supplied -- so
    # action shots came out with no audio track at all. Setting this decodes that
    # state into the model's own scene ambience. Only affects clips WITHOUT dialogue
    # (a clip with --audio-input-path takes the other branch).
    if generate_ambient_audio and not job["audio_path"]:
        env["LTX_GENERATE_AMBIENT_AUDIO"] = "1"

    image_kind = "cadeia (frame anterior)" if chain_image else ("storyboard" if job["storyboard_path"] else "nao")
    end_note = ", keyframe final=sim" if end_image else ""
    log(f"{job['id']}: {num_frames} frames ({duration_sec:.1f}s @ {fps}fps), imagem={image_kind}{end_note}, audio={'sim' if job['audio_path'] else 'nao'}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    code = _run_streamed(command, env=env, log=log)
    if code != 0:
        log(f"{job['id']}: FALHOU (codigo {code}).")
        return False
    if not output_path.exists():
        log(f"{job['id']}: subprocesso terminou OK mas nao criou {output_path}.")
        return False
    log(f"{job['id']}: ok -> {output_path}")
    return True


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gemma-root", default=DEFAULT_GEMMA)
    parser.add_argument("--upsampler", default=DEFAULT_UPSAMPLER)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-clip-seconds", type=float, default=5.0,
                         help="Lines/scenes longer than this get split into several sub-clips (see module docstring: long single clips stall the upsampler). 0 disables splitting.")
    parser.add_argument("--default-scene-seconds", type=float, default=4.0)
    parser.add_argument("--camera-movement", default=None,
                         help='Free-text camera direction applied to every clip, e.g. "slow push in", "handheld", "pan left". Omit for no explicit camera direction (LTX picks its own).')
    # --- engine selection -----------------------------------------------------
    parser.add_argument("--engine", default="ltx", choices=["ltx", "wan"],
                         help="ltx: native ltx_pipelines subprocess (audio-aware). "
                              "wan: ComfyUI Wan 2.1/2.2 nodes (silent video; TTS audio is muxed afterwards).")
    parser.add_argument("--wan-checkpoint", default="", help="Wan diffusion model in ComfyUI/models/diffusion_models/.")
    parser.add_argument("--wan-clip", default="", help="Wan text encoder (umt5) in ComfyUI/models/text_encoders/.")
    parser.add_argument("--wan-vae", default="", help="Wan VAE in ComfyUI/models/vae/.")
    parser.add_argument("--wan-weight-dtype", default="default", choices=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"])
    parser.add_argument("--wan-cfg", type=float, default=5.0, help="Wan runs real CFG (unlike the distilled LTX checkpoint).")
    parser.add_argument("--wan-sampler", default="euler")
    parser.add_argument("--wan-scheduler", default="simple")
    parser.add_argument("--wan-first-last", action="store_true",
                         help="Use WanFirstLastFrameToVideo: pin the clip's END to the scene storyboard as well as its start.")
    parser.add_argument("--comfy-server", default="http://127.0.0.1:8188")
    parser.add_argument("--end-keyframe-strength", type=float, default=0.0,
                         help="Anchor each chained clip's LAST frame to the scene storyboard at this strength "
                              "(LTX VideoConditionByKeyframeIndex; 0 disables). Counters drift accumulating "
                              "along a chain of clips, at the cost of pulling every shot back toward one image.")
    parser.add_argument("--ambient-audio", action="store_true",
                         help="Let LTX vocode its own generated audio for wordless shots (rain, engines, doors) "
                              "instead of leaving them silent. Costs a vocoder pass per action clip.")
    parser.add_argument("--action-beat-seconds", type=float, default=3.0,
                         help="Duration of each wordless action shot (see Scene.action_beats in parse_screenplay.py). Capped by --max-clip-seconds.")
    parser.add_argument("--chain-continuity", dest="chain_continuity", action="store_true", default=True,
                         help="Each clip after the first in a scene starts from the PREVIOUS clip's last frame instead of resetting to the static storyboard image -- default on.")
    parser.add_argument("--no-chain-continuity", dest="chain_continuity", action="store_false")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    # Each engine has its own grid: LTX's two-stage pipeline asserts multiples of 64,
    # while Wan's nodes step by 16 (Wan's own default 832x480 is NOT a multiple of 64,
    # so applying the LTX rule to it would reject a perfectly valid resolution).
    if args.engine == "wan":
        if args.width % 16 or args.height % 16:
            parser.error(f"--width/--height must both be multiples of 16 for Wan (got {args.width}x{args.height}).")
        missing = [name for name, value in (("--wan-checkpoint", args.wan_checkpoint),
                                            ("--wan-clip", args.wan_clip),
                                            ("--wan-vae", args.wan_vae)) if not value]
        if missing:
            parser.error(f"--engine wan requires: {', '.join(missing)}. "
                         "Wan needs a separate diffusion model, umt5 text encoder and VAE in ComfyUI/models/.")
    elif args.width % 64 or args.height % 64:
        parser.error(f"--width/--height must both be multiples of 64 for the two-stage LTX pipeline (got {args.width}x{args.height}).")

    from script_pipeline import run_folder

    run_dir = Path(args.run_dir).resolve()
    scenes = _load_scenes(run_dir)
    cast = _load_cast(run_dir)
    lines_by_key = _load_lines(run_dir)
    storyboard_dir = run_dir / "storyboard"
    scenes_dir = run_folder.subdir(run_dir, "scenes")

    log = lambda msg: run_folder.append_log(run_dir, msg)  # noqa: E731
    free_comfyui_vram(log=log)

    max_clip_seconds = args.max_clip_seconds if args.max_clip_seconds > 0 else None
    jobs = build_render_plan(
        scenes, cast, lines_by_key, storyboard_dir,
        max_clip_seconds=max_clip_seconds, split_dir=scenes_dir / "_line_segments",
        camera_movement=args.camera_movement, action_beat_seconds=args.action_beat_seconds,
    )
    log(f"render_scenes: {len(jobs)} clipe(s) a renderizar.")

    clips_manifest = []
    failures = 0
    last_frame_by_scene: dict[int, str] = {}
    chain_frames_dir = scenes_dir / "_chain_frames"
    for i, job in enumerate(jobs):
        output_path = scenes_dir / f"{job['id']}.mp4"
        chain_image = last_frame_by_scene.get(job["scene_index"]) if args.chain_continuity else None
        # MEASURED (2026-08-10), and this cost a whole render: anchoring every clip's
        # END to the SAME scene storyboard makes that image a fixed point. Clip N ends
        # at the storyboard, so clip N+1 (which starts from clip N's last frame) also
        # starts there -- and ends there too. The chain collapses and every shot comes
        # out visually identical regardless of its description. Verified by comparing
        # shots eight beats apart: same framing, same blocking, same background.
        # The end keyframe is only safe when it points somewhere DIFFERENT from where
        # this clip began -- i.e. at the NEXT shot's own storyboard, which turns it
        # into a transition target instead of an attractor.
        next_job = jobs[i + 1] if i + 1 < len(jobs) else None
        next_storyboard = next_job.get("storyboard_path") if next_job else None
        end_image = (
            next_storyboard
            if (args.end_keyframe_strength > 0 and chain_image and next_storyboard
                and next_storyboard != job["storyboard_path"])
            else None
        )
        if args.engine == "wan":
            ok = render_job_wan(
                job, output_path=output_path, log=log, width=args.width, height=args.height,
                fps=args.fps, steps=args.steps, cfg=args.wan_cfg, seed=args.seed + i,
                default_scene_seconds=args.default_scene_seconds,
                checkpoint=args.wan_checkpoint, clip_name=args.wan_clip, vae_name=args.wan_vae,
                weight_dtype=args.wan_weight_dtype, sampler=args.wan_sampler,
                scheduler=args.wan_scheduler, server=args.comfy_server,
                chain_image=chain_image,
                # Wan pins both ends natively via WanFirstLastFrameToVideo -- no
                # strength knob, so this is an on/off choice rather than the LTX
                # path's --end-keyframe-strength.
                end_image=(job["storyboard_path"] if (args.wan_first_last and chain_image) else None),
            )
        else:
            ok = render_job(
                job, output_path=output_path, checkpoint=args.checkpoint, gemma_root=args.gemma_root,
                upsampler=args.upsampler, width=args.width, height=args.height, fps=args.fps,
                steps=args.steps, seed=args.seed + i, default_scene_seconds=args.default_scene_seconds,
                log=log, chain_image=chain_image, end_image=end_image,
                end_image_strength=args.end_keyframe_strength,
                generate_ambient_audio=args.ambient_audio,
            )
        clips_manifest.append({
            "id": job["id"], "scene_index": job["scene_index"], "line_index": job["line_index"],
            "character": job.get("character"), "video_path": str(output_path) if ok else None,
            "audio_path": job["audio_path"], "ok": ok,
        })
        if not ok:
            failures += 1
        elif args.chain_continuity:
            frame_path = chain_frames_dir / f"{job['id']}_last.png"
            if extract_last_frame(str(output_path), frame_path):
                last_frame_by_scene[job["scene_index"]] = str(frame_path)
            else:
                log(f"{job['id']}: falha ao extrair ultimo frame; proximo clipe da cena volta ao storyboard.")
                last_frame_by_scene.pop(job["scene_index"], None)

    (scenes_dir / "clips.json").write_text(
        json.dumps(clips_manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log(f"render_scenes: {len(jobs) - failures}/{len(jobs)} clipe(s) renderizado(s) com sucesso.")

    if failures == 0:
        run_folder.mark_stage_complete(run_dir, "render")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
