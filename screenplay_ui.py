"""Gradio UI for the screenplay-to-video pipeline (script_pipeline/), built on the
same structural patterns already established in music_maker_ui_v2.py / webui_v2.py /
webui_v3.py:

  - Global mutable state (CURRENT_RUN_DIR, IS_PROCESSING, CURRENT_PHASE, ...) updated
    from background worker threads, never from the Gradio request thread itself.
  - Every "start X" button launches a daemon threading.Thread and returns immediately
    (`start_generation_thread` in webui_v2.py); the actual work streams into the run's
    own log file (script_pipeline/run_folder.append_log -- every stage script already
    writes through this) rather than a separate in-memory log buffer.
  - A gr.Timer(2) drives update_ui(), which re-reads the run folder from disk on every
    tick (log tail, storyboard PNGs, scenes.json/cast.json, final movie) -- the same
    "poll, don't push" model as the music UI's SCENES_DATA refresh, but simpler here
    since the screenplay pipeline's stage outputs are already just files on disk with
    no extra bookkeeping needed.
  - A Worker Log accordion + a Stop button that kills the active per-stage subprocess
    (CURRENT_PROCESS, mirroring cancel_job() in webui_v2.py).

Unlike the music UI's fixed 20-scene slot grid (which exists because every slot needs
simultaneous inline editing for the forward-chain workflow), scene count here is
data-driven from the parsed screenplay -- so storyboards are browsed as a gr.Gallery
and edited one at a time via a scene-index selector, not one fixed row per scene.

CLI: python -u screenplay_ui.py [--port 7810]
"""

from __future__ import annotations

import os
import datetime
import subprocess
import sys
import threading
import types
from pathlib import Path

import gradio as gr
import video_doctor_ui  # aba de diagnostico temporal pos-geracao

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from script_pipeline import run_folder  # noqa: E402
import screenplay_to_video as spv  # noqa: E402

# --- Configuration & Defaults (mirrors screenplay_to_video.py's CLI defaults exactly,
# so a run started from the UI behaves identically to one started from the CLI) ---
DEFAULTS = dict(
    checkpoint="./models/ltx-2.3-22b-distilled-fp8.safetensors",
    gemma_root="./models/gemma3",
    upsampler="./models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    width=896, height=512, fps=24, steps=8, max_clip_seconds=5.0,
    default_scene_seconds=4.0, seed=1234, camera_movement=None, chain_continuity=True,
    storyboard_checkpoint="flux-2-klein-9b-fp8.safetensors",
    storyboard_width=1792, storyboard_height=1024, storyboard_steps=8,
    storyboard_clip="Qwen3-8B-FP8-native-bf16.safetensors",
    storyboard_vae="flux2-vae.safetensors", storyboard_guidance=3.5,
    tts_engine="auto", language="pt", translate_scenes_en=True,
    no_llm=False, llm=False, enrich_engine="gemma4",  # see screenplay_to_video.py for why not gemma3
    lipsync_engine="auto", room_preset="none", room_distance=0.0,
    ambient_track=None, ambient_volume=0.25, output="movie.mp4",
    action_beat_seconds=3.0, end_keyframe_strength=0.0, ambient_audio=True,
    engine="ltx", wan_checkpoint="wan2.2_ti2v_5B_fp16.safetensors",
    wan_clip="umt5_xxl_fp8_e4m3fn_scaled.safetensors", wan_vae="wan2.2_vae.safetensors",
    wan_weight_dtype="default", wan_cfg=5.0, wan_first_last=False,
    storyboard_per_shot=True,
    min_audio_db=-60.0, strict_verify=False,
)

LANGUAGE_CHOICES = [
    ("Portugues (Brasil)", "pt"), ("English", "en"), ("Espanol", "es"),
    ("Francais", "fr"), ("Deutsch", "de"), ("Japanese (nihongo)", "ja"), ("Korean (hangugeo)", "ko"),
]

MAX_SCENES_PREVIEW = 40  # scenes-overview markdown cap; a real screenplay rarely exceeds this

# --- Global State (written only from worker threads; read by update_ui()) ---
CURRENT_RUN_DIR: str | None = None
IS_PROCESSING = False
CURRENT_PHASE = "Ocioso"
CURRENT_PROCESS: subprocess.Popen | None = None
STOP_REQUESTED = False


def _status(message: str, *, phase: str | None = None) -> None:
    global CURRENT_PHASE
    if CURRENT_RUN_DIR:
        run_folder.append_log(CURRENT_RUN_DIR, message)
    else:
        print(message, flush=True)
    if phase is not None:
        CURRENT_PHASE = phase


def build_args(**overrides) -> types.SimpleNamespace:
    values = dict(DEFAULTS)
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _run_stage_ui(stage: str, run_dir: str, args: types.SimpleNamespace) -> bool:
    """Same argv-building as screenplay_to_video.run_stage(), duplicated (not called
    directly) so this UI can keep a handle on the live subprocess for the Stop button
    -- run_stage() itself doesn't expose its Popen object to the caller."""
    global CURRENT_PROCESS
    argv = [sys.executable, "-u", "-m", spv.STAGE_MODULES[stage], "--run-dir", str(run_dir)]

    if stage == "parse":
        argv += ["--script", spv.find_saved_script(Path(run_dir))]
        if args.no_llm:
            argv.append("--no-llm")
        argv += ["--language", args.language,
                 "--translate-scenes-en" if args.translate_scenes_en else "--no-translate-scenes-en",
                 "--enrich-engine", args.enrich_engine]
    elif stage == "cast" and args.llm:
        argv.append("--llm")
    elif stage == "storyboard":
        argv += ["--checkpoint", args.storyboard_checkpoint, "--width", str(args.storyboard_width),
                 "--height", str(args.storyboard_height), "--steps", str(args.storyboard_steps),
                 "--clip", args.storyboard_clip, "--vae", args.storyboard_vae,
                 "--guidance", str(args.storyboard_guidance),
                 "--per-shot" if args.storyboard_per_shot else "--per-scene"]
    elif stage == "dialogue":
        argv += ["--engine", args.tts_engine, "--language", args.language]
    elif stage == "render":
        argv += ["--checkpoint", args.checkpoint, "--gemma-root", args.gemma_root,
                 "--upsampler", args.upsampler, "--width", str(args.width), "--height", str(args.height),
                 "--fps", str(args.fps), "--steps", str(args.steps),
                 "--max-clip-seconds", str(args.max_clip_seconds),
                 "--default-scene-seconds", str(args.default_scene_seconds), "--seed", str(args.seed)]
        if args.camera_movement:
            argv += ["--camera-movement", args.camera_movement]
        argv += ["--action-beat-seconds", str(args.action_beat_seconds),
                 "--end-keyframe-strength", str(args.end_keyframe_strength),
                 "--engine", args.engine]
        if args.ambient_audio:
            argv.append("--ambient-audio")
        if args.engine == "wan":
            argv += ["--wan-checkpoint", args.wan_checkpoint, "--wan-clip", args.wan_clip,
                     "--wan-vae", args.wan_vae, "--wan-weight-dtype", args.wan_weight_dtype,
                     "--wan-cfg", str(args.wan_cfg)]
            if args.wan_first_last:
                argv.append("--wan-first-last")
        argv.append("--chain-continuity" if args.chain_continuity else "--no-chain-continuity")
    elif stage == "lipsync":
        argv += ["--engine", args.lipsync_engine]
    elif stage == "mix":
        argv += ["--room-preset", args.room_preset, "--room-distance", str(args.room_distance),
                 "--ambient-volume", str(args.ambient_volume)]
        if args.ambient_track:
            argv += ["--ambient-track", args.ambient_track]
    elif stage == "assemble":
        argv += ["--output", args.output]
    elif stage == "verify":
        argv += ["--min-audio-db", str(args.min_audio_db)]
        if args.strict_verify:
            argv.append("--strict")

    _status(f"\n=== ETAPA: {stage} ===")
    process = subprocess.Popen(
        argv, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
        encoding="utf-8", errors="replace",
    )
    CURRENT_PROCESS = process
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    CURRENT_PROCESS = None
    return process.returncode == 0


def _run_stages_worker(stages: list[str], args: types.SimpleNamespace) -> None:
    global IS_PROCESSING, CURRENT_PHASE, STOP_REQUESTED
    IS_PROCESSING = True
    try:
        for stage in stages:
            if STOP_REQUESTED:
                _status(f"Execucao interrompida antes de '{stage}'.", phase="Interrompido")
                return
            CURRENT_PHASE = f"Rodando: {stage}"
            ok = _run_stage_ui(stage, CURRENT_RUN_DIR, args)
            if not ok:
                _status(f"Etapa '{stage}' falhou.", phase=f"Falhou em: {stage}")
                return
        CURRENT_PHASE = "Concluido"
        _status("Pipeline concluido.")
    finally:
        IS_PROCESSING = False


def _launch(stages: list[str], args: types.SimpleNamespace) -> str:
    global STOP_REQUESTED
    if CURRENT_RUN_DIR is None:
        return "Nenhuma execucao ativa -- analise um roteiro primeiro."
    if IS_PROCESSING:
        return "Ja existe uma etapa em andamento."
    STOP_REQUESTED = False
    threading.Thread(target=_run_stages_worker, args=(stages, args), daemon=True).start()
    return f"Iniciado: {' -> '.join(stages)}"


def stop_processing() -> str:
    global STOP_REQUESTED, CURRENT_PROCESS
    STOP_REQUESTED = True
    if CURRENT_PROCESS is not None:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(CURRENT_PROCESS.pid)], capture_output=True)
        except Exception:
            pass
    return "Parando (a etapa atual sera encerrada; etapas ja concluidas ficam salvas)..."


# --- New run / stage triggers ---------------------------------------------------

def new_run(script_text: str, script_file, language: str, translate_en: bool, use_llm: bool, enrich_engine: str) -> str:
    global CURRENT_RUN_DIR, IS_PROCESSING, STOP_REQUESTED
    if IS_PROCESSING:
        return "Ja existe uma etapa em andamento."
    source_path = script_file if script_file else None
    if not source_path:
        text = (script_text or "").strip()
        if not text:
            return "Cole um roteiro ou envie um arquivo .txt antes de analisar."
        scratch = ROOT / "outputs" / "screenplay" / "_scratch_input.txt"
        scratch.parent.mkdir(parents=True, exist_ok=True)
        scratch.write_text(text, encoding="utf-8")
        source_path = str(scratch)

    run_dir = run_folder.create_run(source_path)
    CURRENT_RUN_DIR = str(run_dir)
    STOP_REQUESTED = False
    args = build_args(language=language, translate_scenes_en=translate_en, no_llm=not use_llm, enrich_engine=enrich_engine)
    threading.Thread(target=_run_stages_worker, args=(["parse", "cast"], args), daemon=True).start()
    return f"Nova execucao: {run_dir}\nAnalisando roteiro (parse + elenco)..."


def run_storyboards(sb_checkpoint, sb_width, sb_height, sb_steps, sb_clip, sb_vae, sb_guidance) -> str:
    args = build_args(
        storyboard_checkpoint=sb_checkpoint, storyboard_width=int(sb_width), storyboard_height=int(sb_height),
        storyboard_steps=int(sb_steps), storyboard_clip=sb_clip, storyboard_vae=sb_vae,
        storyboard_guidance=float(sb_guidance),
    )
    return _launch(["storyboard"], args)


def run_dialogue(engine: str, language: str) -> str:
    args = build_args(tts_engine=engine, language=language)
    return _launch(["dialogue"], args)


def run_render(width, height, steps, fps, max_clip_seconds, seed, camera_movement, chain_continuity,
               engine, wan_checkpoint, wan_clip, wan_vae, wan_cfg, ambient_audio,
               action_beat_seconds, end_keyframe_strength) -> str:
    args = build_args(
        width=int(width), height=int(height), steps=int(steps), fps=int(fps),
        max_clip_seconds=float(max_clip_seconds), seed=int(seed),
        camera_movement=camera_movement or None, chain_continuity=bool(chain_continuity),
        engine=engine, wan_checkpoint=wan_checkpoint, wan_clip=wan_clip, wan_vae=wan_vae,
        wan_cfg=float(wan_cfg), ambient_audio=bool(ambient_audio),
        action_beat_seconds=float(action_beat_seconds),
        end_keyframe_strength=float(end_keyframe_strength),
    )
    return _launch(["render"], args)


def run_lipsync(engine: str) -> str:
    return _launch(["lipsync"], build_args(lipsync_engine=engine))


def run_mix(room_preset, room_distance, ambient_track, ambient_volume) -> str:
    args = build_args(
        room_preset=room_preset, room_distance=float(room_distance),
        ambient_track=ambient_track or None, ambient_volume=float(ambient_volume),
    )
    return _launch(["mix"], args)


def run_assemble(output_name: str) -> str:
    # verify runs right behind assemble (never on its own button): the review is only
    # meaningful against a freshly built movie, and making it automatic is the whole
    # point -- the two silent films that shipped were both waved through by a human
    # who had no reason to suspect the audio track was there but empty.
    return _launch(["assemble", "verify"], build_args(output=output_name or "movie.mp4"))


def run_full_pipeline(
    language, translate_en, use_llm, enrich_engine, sb_checkpoint, sb_width, sb_height, sb_steps, sb_clip, sb_vae, sb_guidance,
    tts_engine, width, height, steps, fps, max_clip_seconds, seed, camera_movement, chain_continuity, lipsync_engine,
    room_preset, room_distance, ambient_track, ambient_volume, output_name,
) -> str:
    args = build_args(
        language=language, translate_scenes_en=translate_en, no_llm=not use_llm, enrich_engine=enrich_engine,
        storyboard_checkpoint=sb_checkpoint, storyboard_width=int(sb_width), storyboard_height=int(sb_height),
        storyboard_steps=int(sb_steps), storyboard_clip=sb_clip, storyboard_vae=sb_vae,
        storyboard_guidance=float(sb_guidance), tts_engine=tts_engine,
        width=int(width), height=int(height), steps=int(steps), fps=int(fps),
        max_clip_seconds=float(max_clip_seconds), seed=int(seed), camera_movement=camera_movement or None,
        chain_continuity=bool(chain_continuity), lipsync_engine=lipsync_engine,
        room_preset=room_preset, room_distance=float(room_distance),
        ambient_track=ambient_track or None, ambient_volume=float(ambient_volume), output=output_name or "movie.mp4",
    )
    return _launch(spv.STAGE_ORDER, args)


# --- Scene storyboard editor (regenerate one scene / replace with an uploaded image) --

# --- Decoupage editing (item 3) + character reference photos (item 1) -------------

def _scenes_path() -> Path | None:
    """The enriched file is what every later stage reads, so edits must land there."""
    if not CURRENT_RUN_DIR:
        return None
    parse_dir = Path(CURRENT_RUN_DIR) / "parse"
    enriched = parse_dir / "scenes_enriched.json"
    return enriched if enriched.exists() else (parse_dir / "scenes.json")


def _flat_shots() -> list[tuple[int, int, dict, dict]]:
    """Flatten every scene's shot_list into film order.
    Returns (global_position, scene_index, shot, scene)."""
    out = []
    for scene in _load_scenes_json():
        for shot in scene.get("shot_list") or []:
            out.append((len(out) + 1, scene["index"], shot, scene))
    return out


def _shots_overview_markdown() -> str:
    shots = _flat_shots()
    if not shots:
        return "(analise um roteiro primeiro; ou o parse nao produziu decupagem)"
    lines = []
    for pos, scene_index, shot, scene in shots:
        if shot.get("type") == "action":
            lines.append(f"**{pos}.** `acao` (cena {scene_index}) — {shot.get('visual', '')}")
        else:
            line = (scene.get("dialogue") or [])[shot.get("line_index", 0)]
            beat = line.get("beat_visual") or ""
            lines.append(
                f"**{pos}.** `fala` (cena {scene_index}) — **{line.get('character')}**: "
                f"\"{line.get('text', '')}\"  \n_{beat}_"
            )
    return "\n\n".join(lines)


def load_shot(position: int):
    shots = _flat_shots()
    pos = int(position)
    if not (1 <= pos <= len(shots)):
        return "", f"Plano {pos} nao existe (a decupagem tem {len(shots)})."
    _, scene_index, shot, scene = shots[pos - 1]
    if shot.get("type") == "action":
        return shot.get("visual", ""), f"Plano {pos}: acao da cena {scene_index}."
    line = (scene.get("dialogue") or [])[shot.get("line_index", 0)]
    return (line.get("beat_visual") or "",
            f"Plano {pos}: fala de {line.get('character')} — a FALA em si nao muda aqui, "
            "so a descricao visual do momento.")


def save_shot(position: int, text: str) -> str:
    """Write the edited description back into the run's scenes file."""
    path = _scenes_path()
    if path is None:
        return "Nenhuma execucao ativa."
    import json as _json
    scenes = _json.loads(path.read_text(encoding="utf-8"))
    pos, target = 0, None
    for scene in scenes:
        for shot in scene.get("shot_list") or []:
            pos += 1
            if pos == int(position):
                target = (scene, shot)
                break
        if target:
            break
    if target is None:
        return f"Plano {position} nao encontrado."
    scene, shot = target
    if shot.get("type") == "action":
        shot["visual"] = text.strip()
    else:
        scene["dialogue"][shot.get("line_index", 0)]["beat_visual"] = text.strip()
    path.write_text(_json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"Plano {position} salvo. Regere o storyboard e/ou o render para aplicar."


def list_characters() -> "gr.Dropdown":
    import json as _json
    if not CURRENT_RUN_DIR:
        return gr.update(choices=[], value=None)
    cast_path = Path(CURRENT_RUN_DIR) / "characters" / "cast.json"
    if not cast_path.exists():
        return gr.update(choices=[], value=None)
    names = list(_json.loads(cast_path.read_text(encoding="utf-8")).keys())
    return gr.update(choices=names, value=names[0] if names else None)


def list_archetypes() -> "gr.Dropdown":
    """Populate the archetype dropdown from whatever is on disk.

    Read live rather than hardcoded: the speaker library is maintained outside this
    repo, and a list baked in here would silently go stale the next time voices are
    added -- which is exactly how 24 character clips and two whole folders ended up
    invisible to the pipeline before.
    """
    from script_pipeline.voice_library import list_archetype_voices
    names = sorted(list_archetype_voices())
    return gr.update(choices=names, value=names[0] if names else None)


def set_character_archetype(name: str, archetype: str | None) -> str:
    """Set (or clear) one character's archetype voice in cast.json."""
    import json as _json
    if not CURRENT_RUN_DIR or not name:
        return "Selecione um personagem (e analise um roteiro antes)."
    cast_path = Path(CURRENT_RUN_DIR) / "characters" / "cast.json"
    if not cast_path.exists():
        return "cast.json ainda nao existe."
    cast = _json.loads(cast_path.read_text(encoding="utf-8"))
    if name not in cast:
        return f"Personagem '{name}' nao esta no elenco."
    voice = cast[name].setdefault("voice", {})
    if archetype:
        from script_pipeline.voice_library import list_archetype_voices
        if archetype not in list_archetype_voices():
            return f"Arquetipo '{archetype}' nao existe na biblioteca."
        voice["archetype_voice"] = archetype
        msg = (f"{name}: voz de arquetipo '{archetype}'. Falas com emocao usam a tomada "
               "atuada quando ela corresponde; as demais usam a identidade base.")
    else:
        voice["archetype_voice"] = None
        msg = f"{name}: de volta a voz atribuida automaticamente."
    cast_path.write_text(_json.dumps(cast, ensure_ascii=False, indent=2), encoding="utf-8")
    return msg


def set_character_reference(name: str, photo: str | None) -> str:
    """Attach (or clear) a reference photo for one character in cast.json."""
    import json as _json
    if not CURRENT_RUN_DIR or not name:
        return "Selecione um personagem (e analise um roteiro antes)."
    cast_path = Path(CURRENT_RUN_DIR) / "characters" / "cast.json"
    if not cast_path.exists():
        return "cast.json ainda nao existe."
    cast = _json.loads(cast_path.read_text(encoding="utf-8"))
    if name not in cast:
        return f"Personagem '{name}' nao esta no elenco."
    if photo:
        # Copy into the run folder so the reference survives independently of wherever
        # the user picked the file from.
        dest_dir = Path(CURRENT_RUN_DIR) / "characters" / "references"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{name.replace(' ', '_')}.png"
        from PIL import Image
        Image.open(photo).convert("RGB").save(dest)
        cast[name]["reference_image"] = str(dest)
        msg = f"{name}: foto de referencia definida -> {dest.name}"
    else:
        cast[name]["reference_image"] = None
        msg = f"{name}: voltou ao modo automatico (identidade so pelo descritor de texto)."
    cast_path.write_text(_json.dumps(cast, ensure_ascii=False, indent=2), encoding="utf-8")
    return msg + " Regere os storyboards para aplicar."


def regenerate_shot_storyboard(
    position: int, sb_checkpoint, sb_width, sb_height, sb_steps, sb_clip, sb_vae, sb_guidance,
) -> str:
    """Regenerate the storyboard for ONE shot of the decoupage (item 1: 'regenerate
    scenes'). Runs the same generate_storyboards machinery in-process rather than as a
    stage, because a single image is fast and the user is waiting on it."""
    if CURRENT_RUN_DIR is None:
        return "Nenhuma execucao ativa."
    if IS_PROCESSING:
        return "Ja existe uma etapa em andamento."
    shots = _flat_shots()
    pos = int(position)
    if not (1 <= pos <= len(shots)):
        return f"Plano {pos} nao existe (a decupagem tem {len(shots)})."

    def _worker():
        global IS_PROCESSING, CURRENT_PHASE
        IS_PROCESSING = True
        CURRENT_PHASE = f"Regerando storyboard do plano {pos}"
        try:
            from script_pipeline.generate_storyboards import (
                build_prompt, ensure_comfyui_running, generate_scene_storyboard,
            )
            _, scene_index, shot, scene = shots[pos - 1]
            cast_path = Path(CURRENT_RUN_DIR) / "characters" / "cast.json"
            import json as _json
            cast = _json.loads(cast_path.read_text(encoding="utf-8")) if cast_path.exists() else {}

            reference = None
            if shot.get("type") == "action":
                prompt = build_prompt({**scene, "visual_prompt": shot.get("visual")}, cast)
            else:
                line = (scene.get("dialogue") or [])[shot.get("line_index", 0)]
                prompt = build_prompt({**scene, "visual_prompt": line.get("beat_visual")}, cast)
                prompt += f" Close-up on {line.get('character', '')}."
                reference = (cast.get(line.get("character", "")) or {}).get("reference_image")

            # The per-shot filename must match what render_scenes.py looks for, which
            # is keyed on the shot's position WITHIN ITS SCENE, not the global one.
            local_pos = sum(1 for p, si, _, _ in shots[:pos] if si == scene_index)
            out_path = Path(CURRENT_RUN_DIR) / "storyboard" / f"scene_{scene_index:02d}_s{local_pos:03d}.png"

            log = lambda m: run_folder.append_log(CURRENT_RUN_DIR, m)  # noqa: E731
            if not ensure_comfyui_running("http://127.0.0.1:8188", log=log):
                CURRENT_PHASE = "Falhou (ComfyUI)"
                return
            generate_scene_storyboard(
                scene, cast, server="http://127.0.0.1:8188", checkpoint=sb_checkpoint,
                width=int(sb_width), height=int(sb_height), steps=int(sb_steps), cfg=7.0,
                seed=1234 + pos, out_path=out_path, log=log,
                clip=sb_clip, vae=sb_vae, guidance=float(sb_guidance),
                prompt_override=prompt, reference_image=reference,
            )
            CURRENT_PHASE = "Concluido"
        finally:
            IS_PROCESSING = False

    threading.Thread(target=_worker, daemon=True).start()
    return f"Regerando storyboard do plano {pos}..."


def _load_scenes_json() -> list[dict]:
    if not CURRENT_RUN_DIR:
        return []
    import json
    parse_dir = Path(CURRENT_RUN_DIR) / "parse"
    path = parse_dir / "scenes_enriched.json"
    if not path.exists():
        path = parse_dir / "scenes.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_scene_for_edit(scene_index: int):
    scenes = _load_scenes_json()
    scene = next((s for s in scenes if s["index"] == int(scene_index)), None)
    if scene is None:
        return "(cena nao encontrada)", None, "(cena nao encontrada)"
    prompt = scene.get("visual_prompt") or scene.get("action_text") or scene.get("heading_raw", "")
    img_path = Path(CURRENT_RUN_DIR) / "storyboard" / f"scene_{int(scene_index):02d}.png"
    summary = (
        f"**Cena {scene['index']}** -- {scene.get('heading_raw', '')}\n\n"
        f"Personagens: {', '.join(scene.get('characters', [])) or '(nenhum)'}\n\n"
        f"Falas: {len(scene.get('dialogue', []))}"
    )
    return prompt, (str(img_path) if img_path.exists() else None), summary


def regenerate_scene(
    scene_index: int, prompt_text: str, sb_checkpoint, sb_width, sb_height, sb_steps, sb_clip, sb_vae, sb_guidance,
) -> str:
    if CURRENT_RUN_DIR is None:
        return "Nenhuma execucao ativa."
    if IS_PROCESSING:
        return "Ja existe uma etapa em andamento."
    args = build_args(
        storyboard_checkpoint=sb_checkpoint, storyboard_width=int(sb_width), storyboard_height=int(sb_height),
        storyboard_steps=int(sb_steps), storyboard_clip=sb_clip, storyboard_vae=sb_vae,
        storyboard_guidance=float(sb_guidance),
    )

    def _worker():
        global IS_PROCESSING, CURRENT_PHASE
        IS_PROCESSING = True
        CURRENT_PHASE = f"Regenerando storyboard da cena {scene_index}"
        try:
            argv = [
                sys.executable, "-u", "-m", "script_pipeline.generate_storyboards",
                "--run-dir", CURRENT_RUN_DIR, "--checkpoint", args.storyboard_checkpoint,
                "--width", str(args.storyboard_width), "--height", str(args.storyboard_height),
                "--steps", str(args.storyboard_steps), "--clip", args.storyboard_clip,
                "--vae", args.storyboard_vae, "--guidance", str(args.storyboard_guidance),
                "--only-scene", str(int(scene_index)),
            ]
            if (prompt_text or "").strip():
                argv += ["--prompt-override", prompt_text.strip()]
            _status(f"\n=== Regenerando storyboard: cena {scene_index} ===")
            global CURRENT_PROCESS
            process = subprocess.Popen(argv, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, universal_newlines=True,
                                        encoding="utf-8", errors="replace")
            CURRENT_PROCESS = process
            for line in process.stdout:
                print(line, end="", flush=True)
            process.wait()
            CURRENT_PROCESS = None
            CURRENT_PHASE = "Concluido" if process.returncode == 0 else "Falhou"
        finally:
            IS_PROCESSING = False

    threading.Thread(target=_worker, daemon=True).start()
    return f"Regenerando storyboard da cena {scene_index}..."


def replace_scene_image(scene_index: int, uploaded_path: str | None) -> str:
    if CURRENT_RUN_DIR is None:
        return "Nenhuma execucao ativa."
    if not uploaded_path:
        return "Nenhuma imagem enviada."
    import shutil
    dest = Path(CURRENT_RUN_DIR) / "storyboard" / f"scene_{int(scene_index):02d}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.open(uploaded_path).convert("RGB").save(dest)
    _status(f"Storyboard da cena {scene_index} substituido manualmente por {Path(uploaded_path).name}.")
    return f"Cena {scene_index}: imagem substituida -> {dest}"


# --- Timer-driven UI refresh (poll the run folder on disk, same idea as SCENES_DATA
# refresh in webui_v2.py, but sourced from files instead of an in-memory mirror) -----

def _tail_log(n_chars: int = 6000) -> str:
    if not CURRENT_RUN_DIR:
        return "Nenhuma execucao ativa ainda."
    path = run_folder.log_path(CURRENT_RUN_DIR)
    if not path.exists():
        return "(sem log ainda)"
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-n_chars:]


def _scenes_overview_markdown() -> str:
    scenes = _load_scenes_json()
    if not scenes:
        return "(nenhuma cena analisada ainda)"
    lines = []
    for scene in scenes[:MAX_SCENES_PREVIEW]:
        chars = ", ".join(scene.get("characters", [])) or "(nenhum)"
        lines.append(
            f"**Cena {scene['index']}** -- {scene.get('heading_raw', '')}  \n"
            f"Personagens: {chars} | Falas: {len(scene.get('dialogue', []))}  \n"
            f"_{(scene.get('visual_prompt') or scene.get('action_text') or '').strip()[:200]}_"
        )
    if len(scenes) > MAX_SCENES_PREVIEW:
        lines.append(f"... e mais {len(scenes) - MAX_SCENES_PREVIEW} cena(s).")
    return "\n\n---\n\n".join(lines)


def _cast_overview_markdown() -> str:
    if not CURRENT_RUN_DIR:
        return "(nenhum elenco ainda)"
    import json
    cast_path = Path(CURRENT_RUN_DIR) / "characters" / "cast.json"
    if not cast_path.exists():
        return "(nenhum elenco ainda)"
    cast = json.loads(cast_path.read_text(encoding="utf-8"))
    if not cast:
        return "(roteiro sem personagens com fala)"
    lines = []
    for name, entry in cast.items():
        voice = entry.get("voice", {})
        gender_label = voice.get("gender", "?")
        if voice.get("gender_guessed"):
            gender_label += " -- ⚠️ palpite, sem pista textual, confira"
        # Say WHERE the voice came from: a hand-picked archetype, the curated CSV, the
        # 17-take emotive library, or a plain clip with no emotional range at all. Those
        # four differ in what the delivery can do, so the label has to distinguish them.
        if voice.get("archetype_voice"):
            origem = f"arquetipo escolhido: {voice['archetype_voice']}"
        elif voice.get("voice_map_source"):
            origem = f"mapa_vozes.csv ({voice['voice_map_source']})"
        elif voice.get("emotive_voice"):
            origem = "biblioteca emotiva (17 tomadas por emocao)"
        else:
            origem = "clipe simples -- sem variacao emocional"
        lines.append(
            f"**{name}** ({gender_label}) -- {entry.get('descriptor', '')}  \n"
            f"Voz: xtts={voice.get('xtts_speaker_wav', '-')}, qwen={voice.get('qwen_speaker', '-')}  \n"
            f"Origem: {origem}"
        )
    return "\n\n".join(lines)


def _storyboard_gallery() -> list[str]:
    if not CURRENT_RUN_DIR:
        return []
    storyboard_dir = Path(CURRENT_RUN_DIR) / "storyboard"
    if not storyboard_dir.is_dir():
        return []
    return sorted(str(p) for p in storyboard_dir.glob("scene_*.png"))


def _final_video_path() -> str | None:
    if not CURRENT_RUN_DIR:
        return None
    path = Path(CURRENT_RUN_DIR) / "final" / "movie.mp4"
    return str(path) if path.exists() else None


def _verification_markdown() -> str:
    """Render verification.json. Defects are shown ABOVE the video player, because the
    failure mode this guards against is a movie that looks completely normal in the
    player -- a silent track and a black clip both play back without complaint."""
    import json  # module-local, matching how every other reader in this file imports it

    if not CURRENT_RUN_DIR:
        return "_Sem execucao ativa._"
    report_path = Path(CURRENT_RUN_DIR) / "verification.json"
    if not report_path.exists():
        return "_Ainda sem revisao. Ela roda automaticamente apos a montagem._"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"_Nao consegui ler verification.json: {exc}_"

    findings = report.get("findings") or []
    errors = [f for f in findings if f.get("severity") == "error"]
    warnings = [f for f in findings if f.get("severity") != "error"]

    lines: list[str] = []
    if not findings:
        lines.append("### Nenhum defeito detectado")
    else:
        lines.append(f"### {len(errors)} erro(s), {len(warnings)} aviso(s)")
    for item in errors + warnings:
        icon = "**ERRO**" if item.get("severity") == "error" else "aviso"
        where = f" (`{item['clip']}`)" if item.get("clip") else ""
        lines.append(f"- {icon} `{item.get('code', '?')}`{where}: {item.get('detail', '')}")

    lines.append("")
    lines.append("#### Medicoes")
    for check in report.get("checks") or []:
        if not isinstance(check, dict):
            continue
        name = check.get("check", "?")
        detail = ", ".join(f"{k}={v}" for k, v in check.items() if k != "check" and v is not None)
        lines.append(f"- `{name}`: {detail}")
    return "\n".join(lines)


def update_ui():
    status_line = f"{CURRENT_PHASE}" + (f" | pasta: {CURRENT_RUN_DIR}" if CURRENT_RUN_DIR else "")
    return (
        status_line,
        _tail_log(),
        _scenes_overview_markdown(),
        _shots_overview_markdown(),
        _cast_overview_markdown(),
        _storyboard_gallery(),
        _final_video_path(),
        _verification_markdown(),
    )


# --- UI Layout -------------------------------------------------------------------

theme = gr.themes.Soft(primary_hue="indigo").set(
    body_background_fill="*neutral_50",
    block_background_fill="*neutral_100",
)

with gr.Blocks(title="LTX-2 Screenplay to Video") as demo:
    gr.Markdown("# 🎬 Roteiro -> Video (script_pipeline)")
    gr.Markdown(
        "> Fluxo: analisar roteiro (parse + elenco) -> storyboards (FLUX) -> vozes (TTS) -> "
        "renderizar cenas (LTX) -> lip-sync -> mixar audio -> montar filme final. "
        "Cada etapa roda isolada em subprocesso (nenhum modelo pesado fica residente entre etapas)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Roteiro")
            script_text = gr.Textbox(label="Cole o roteiro aqui", lines=16, placeholder="INT. LOCAL - NOITE\n...")
            script_file = gr.File(label="ou envie um arquivo .txt", file_types=[".txt"], type="filepath")

            with gr.Accordion("Idioma", open=True):
                language = gr.Dropdown(LANGUAGE_CHOICES, value="pt", label="Idioma das vozes e das falas")
                translate_en = gr.Checkbox(
                    value=True,
                    label="Traduzir a DESCRICAO VISUAL das cenas para ingles (recomendado -- FLUX/LTX leem melhor prompts em ingles)",
                    info="As falas dos personagens NUNCA sao traduzidas -- ficam sempre no idioma selecionado, na voz e no texto incorporado ao prompt de video.",
                )
                use_llm = gr.Checkbox(value=True, label="Enriquecer com LLM: prompt visual por cena + emocao das falas")
                enrich_engine = gr.Dropdown(
                    [("Gemma4 E2B (venv isolado) -- recomendado", "gemma4"), ("Gemma3 12B (venv principal)", "gemma3")],
                    value="gemma4", label="Modelo de enriquecimento",
                    info="Gemma4 roda em bf16 nativo (sem quantizacao) e precisa do venv gemma4_env. Gemma3 foi MEDIDO devolvendo token-salad multilingue em vez de JSON: sem shot list, storyboard unico por cena e prompt de render feito do roteiro cru.",
                )

            parse_btn = gr.Button("🔎 1. Analisar roteiro (parse + elenco)", variant="primary")

            with gr.Accordion("Opcoes de storyboard (FLUX)", open=False):
                sb_checkpoint = gr.Textbox(value=DEFAULTS["storyboard_checkpoint"], label="Checkpoint FLUX")
                with gr.Row():
                    sb_width = gr.Number(value=DEFAULTS["storyboard_width"], label="Largura", precision=0)
                    sb_height = gr.Number(value=DEFAULTS["storyboard_height"], label="Altura", precision=0)
                sb_steps = gr.Slider(1, 30, value=DEFAULTS["storyboard_steps"], step=1, label="Steps")
                sb_guidance = gr.Slider(0.0, 10.0, value=DEFAULTS["storyboard_guidance"], step=0.5, label="Guidance")
                sb_clip = gr.Textbox(value=DEFAULTS["storyboard_clip"], label="Text encoder (CLIP)")
                sb_vae = gr.Textbox(value=DEFAULTS["storyboard_vae"], label="VAE")
            storyboard_btn = gr.Button("🖼️ 2. Gerar storyboards")

            with gr.Accordion("Opcoes de voz (TTS)", open=False):
                tts_engine = gr.Dropdown(["auto", "xtts", "qwen"], value="auto", label="Motor de TTS")
            dialogue_btn = gr.Button("🗣️ 3. Sintetizar vozes")

            with gr.Accordion("Opcoes de render (LTX)", open=False):
                with gr.Row():
                    render_width = gr.Number(value=DEFAULTS["width"], label="Largura", precision=0)
                    render_height = gr.Number(value=DEFAULTS["height"], label="Altura", precision=0)
                with gr.Row():
                    render_steps = gr.Slider(1, 30, value=DEFAULTS["steps"], step=1, label="Steps")
                    render_fps = gr.Number(value=DEFAULTS["fps"], label="FPS", precision=0)
                max_clip_seconds = gr.Slider(
                    1.0, 15.0, value=DEFAULTS["max_clip_seconds"], step=0.5,
                    label="Duracao maxima por clipe (s)",
                    info="Falas mais longas sao divididas em varios sub-clipes -- clipes longos travam o upsampler.",
                )
                render_seed = gr.Number(value=DEFAULTS["seed"], label="Seed", precision=0)
                engine = gr.Radio(
                    [("LTX 2.3 (audio nativo)", "ltx"), ("Wan 2.2 TI2V 5B (mudo + audio muxado)", "wan")],
                    value="ltx", label="Engine de video",
                )
                with gr.Accordion("Modelos do Wan (so quando engine=wan)", open=False):
                    wan_checkpoint = gr.Textbox(value=DEFAULTS["wan_checkpoint"], label="Modelo de difusao")
                    wan_clip = gr.Textbox(value=DEFAULTS["wan_clip"], label="Text encoder (umt5)")
                    wan_vae = gr.Textbox(value=DEFAULTS["wan_vae"], label="VAE")
                    wan_cfg = gr.Slider(1.0, 12.0, value=DEFAULTS["wan_cfg"], step=0.5, label="CFG (Wan nao e destilado)")
                ambient_audio = gr.Checkbox(
                    value=True, label="Som ambiente gerado pelo LTX nos planos sem fala",
                    info="Sem isso os planos de acao saem completamente mudos.",
                )
                action_beat_seconds = gr.Slider(
                    1.0, 8.0, value=DEFAULTS["action_beat_seconds"], step=0.5,
                    label="Duracao dos planos de acao (s)",
                )
                end_keyframe_strength = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.05, label="Ancorar fim do clipe no proximo plano",
                    info="0 = desligado. Ancorar sempre na MESMA imagem colapsa a cadeia (ja medido); aqui aponta para o plano seguinte.",
                )
                camera_movement = gr.Dropdown(
                    ["", "pan left", "pan right", "dolly in", "dolly out", "handheld", "static",
                     "crane up", "tilt up", "push in", "pull back", "wide establishing"],
                    value="", allow_custom_value=True, label="Movimento de camera (opcional, aplicado a todos os clipes)",
                    info="Vazio = LTX escolhe por conta propria. Pode digitar um texto livre tambem (ex: 'slow zoom out').",
                )
                chain_continuity = gr.Checkbox(
                    value=True, label="Continuidade entre clipes da mesma cena",
                    info="Cada clipe (depois do primeiro) comeca do ultimo frame do clipe anterior, em vez de sempre voltar ao storyboard estatico -- mantem pose/acao continuas.",
                )
            render_btn = gr.Button("🎥 4. Renderizar cenas")

            with gr.Accordion("Opcoes de lip-sync e mixagem", open=False):
                lipsync_engine = gr.Dropdown(["auto", "latentsync", "wav2lip"], value="auto", label="Motor de lip-sync")
                room_preset = gr.Dropdown(["none", "small_room", "hall", "cathedral"], value="none", label="Reverb (preset de sala)")
                room_distance = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Distancia (reverb)")
                ambient_track = gr.Audio(label="Trilha ambiente (opcional)", type="filepath")
                ambient_volume = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="Volume do ambiente")
            lipsync_btn = gr.Button("👄 5. Lip-sync")
            mix_btn = gr.Button("🔊 6. Mixar audio")

            output_name = gr.Textbox(value="movie.mp4", label="Nome do arquivo final")
            assemble_btn = gr.Button("🎬 7. Montar filme final")

            gr.Markdown("---")
            full_pipeline_btn = gr.Button("🚀 Rodar tudo (do parse ate o filme final)", variant="primary", size="lg")
            with gr.Row():
                stop_btn = gr.Button("🛑 Parar", variant="stop")
            status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📄 Cenas"):
                    scenes_md = gr.Markdown("(nenhuma cena analisada ainda)")
                with gr.Tab("🎞️ Decupagem"):
                    gr.Markdown(
                        "Ordem exata dos planos do filme, como saiu do parse. "
                        "Edite o texto de qualquer plano e salve para mudar o que sera gerado."
                    )
                    shots_md = gr.Markdown("(analise um roteiro primeiro)")
                    with gr.Row():
                        shot_index = gr.Number(value=1, label="Plano nº", precision=0)
                        load_shot_btn = gr.Button("Carregar plano")
                    shot_text = gr.Textbox(label="Descricao visual deste plano", lines=3)
                    with gr.Row():
                        save_shot_btn = gr.Button("💾 Salvar plano", variant="primary")
                        regen_shot_sb_btn = gr.Button("🖼️ Regerar storyboard deste plano")
                    shot_status = gr.Textbox(label=None, interactive=False)
                with gr.Tab("🎭 Elenco"):
                    cast_md = gr.Markdown("(nenhum elenco ainda)")
                    gr.Markdown(
                        "#### Personagem por foto de referencia\n"
                        "Sem foto, a identidade vem so do descritor de texto (modo automatico) e o "
                        "personagem tende a mudar de rosto entre planos. Com foto, o FLUX ancora o "
                        "rosto -- verificado em teste controlado neste projeto."
                    )
                    with gr.Row():
                        char_name = gr.Dropdown([], label="Personagem", allow_custom_value=True)
                        reload_chars_btn = gr.Button("↻ Atualizar lista")
                    char_photo = gr.Image(label="Foto de referencia", type="filepath")
                    with gr.Row():
                        set_ref_btn = gr.Button("📌 Usar esta foto", variant="primary")
                        clear_ref_btn = gr.Button("🗑️ Voltar ao automatico")
                    char_status = gr.Textbox(label=None, interactive=False)

                    gr.Markdown(
                        "#### Voz de arquetipo (opcional)\n"
                        "A atribuicao automatica escolhe uma voz por genero, com 17 tomadas "
                        "emocionais. Aqui voce pode substituir por um arquetipo (heroi, bruxa, "
                        "fantasma, androide, narrador...). Nao ha escolha automatica de arquetipo "
                        "de proposito: casar 'um mago idoso' com C05_mago e julgamento semantico, "
                        "e errar o arquetipo da ao personagem a PERSONA errada, nao so o timbre."
                    )
                    with gr.Row():
                        archetype_dd = gr.Dropdown([], label="Arquetipo", allow_custom_value=False)
                        reload_arch_btn = gr.Button("↻ Listar arquetipos")
                    with gr.Row():
                        set_arch_btn = gr.Button("🎭 Aplicar arquetipo", variant="primary")
                        clear_arch_btn = gr.Button("🗑️ Voltar a voz automatica")
                    arch_status = gr.Textbox(label=None, interactive=False)
                with gr.Tab("🖼️ Storyboards"):
                    gallery = gr.Gallery(label="Storyboards gerados", columns=3, height=420)
                    gr.Markdown("#### Editar / regenerar uma cena")
                    with gr.Row():
                        edit_scene_index = gr.Number(value=1, label="Numero da cena", precision=0)
                        load_scene_btn = gr.Button("Carregar")
                    scene_summary_md = gr.Markdown("")
                    edit_prompt = gr.Textbox(label="Prompt visual da cena (editavel)", lines=3)
                    edit_preview = gr.Image(label="Storyboard atual", interactive=False)
                    with gr.Row():
                        regen_btn = gr.Button("🔁 Regenerar esta cena com o prompt acima")
                    with gr.Row():
                        replace_upload = gr.Image(label="Ou substitua por uma imagem sua", type="filepath")
                        replace_btn = gr.Button("📤 Substituir")
                    scene_edit_status = gr.Textbox(label=None, interactive=False)
                with gr.Tab("🎬 Video final"):
                    verification_md = gr.Markdown("_Ainda sem revisao._")
                    final_video = gr.Video(label="Filme final")

                # Pos-producao: diagnostico e correcao temporal do filme ja
                # montado. Fica FORA do fluxo de geracao de proposito.
                video_doctor_ui.build_doctor_tab(label="🩺 Diagnostico e correcao")

    with gr.Accordion("Log do processo", open=True):
        log_box = gr.Textbox(label=None, lines=14, interactive=False)

    # --- Events ---
    parse_btn.click(
        fn=new_run, inputs=[script_text, script_file, language, translate_en, use_llm, enrich_engine], outputs=[status_box],
    )
    storyboard_btn.click(
        fn=run_storyboards, inputs=[sb_checkpoint, sb_width, sb_height, sb_steps, sb_clip, sb_vae, sb_guidance],
        outputs=[status_box],
    )
    dialogue_btn.click(fn=run_dialogue, inputs=[tts_engine, language], outputs=[status_box])
    render_btn.click(
        fn=run_render, inputs=[render_width, render_height, render_steps, render_fps, max_clip_seconds, render_seed, camera_movement, chain_continuity,
                               engine, wan_checkpoint, wan_clip, wan_vae, wan_cfg, ambient_audio,
                               action_beat_seconds, end_keyframe_strength],
        outputs=[status_box],
    )
    lipsync_btn.click(fn=run_lipsync, inputs=[lipsync_engine], outputs=[status_box])
    mix_btn.click(fn=run_mix, inputs=[room_preset, room_distance, ambient_track, ambient_volume], outputs=[status_box])
    assemble_btn.click(fn=run_assemble, inputs=[output_name], outputs=[status_box])
    full_pipeline_btn.click(
        fn=run_full_pipeline,
        inputs=[
            language, translate_en, use_llm, enrich_engine, sb_checkpoint, sb_width, sb_height, sb_steps, sb_clip, sb_vae, sb_guidance,
            tts_engine, render_width, render_height, render_steps, render_fps, max_clip_seconds, render_seed, camera_movement, chain_continuity,
            lipsync_engine, room_preset, room_distance, ambient_track, ambient_volume, output_name,
        ],
        outputs=[status_box],
    )
    stop_btn.click(fn=stop_processing, outputs=[status_box])

    load_shot_btn.click(fn=load_shot, inputs=[shot_index], outputs=[shot_text, shot_status])
    save_shot_btn.click(fn=save_shot, inputs=[shot_index, shot_text], outputs=[shot_status])
    regen_shot_sb_btn.click(
        fn=regenerate_shot_storyboard,
        inputs=[shot_index, sb_checkpoint, sb_width, sb_height, sb_steps, sb_clip, sb_vae, sb_guidance],
        outputs=[shot_status],
    )
    reload_chars_btn.click(fn=list_characters, outputs=[char_name])
    set_ref_btn.click(fn=set_character_reference, inputs=[char_name, char_photo], outputs=[char_status])
    clear_ref_btn.click(fn=lambda n: set_character_reference(n, None), inputs=[char_name], outputs=[char_status])
    reload_arch_btn.click(fn=list_archetypes, outputs=[archetype_dd])
    set_arch_btn.click(fn=set_character_archetype, inputs=[char_name, archetype_dd], outputs=[arch_status])
    clear_arch_btn.click(fn=lambda n: set_character_archetype(n, None), inputs=[char_name], outputs=[arch_status])

    load_scene_btn.click(fn=load_scene_for_edit, inputs=[edit_scene_index], outputs=[edit_prompt, edit_preview, scene_summary_md])
    regen_btn.click(
        fn=regenerate_scene,
        inputs=[edit_scene_index, edit_prompt, sb_checkpoint, sb_width, sb_height, sb_steps, sb_clip, sb_vae, sb_guidance],
        outputs=[scene_edit_status],
    )
    replace_btn.click(fn=replace_scene_image, inputs=[edit_scene_index, replace_upload], outputs=[scene_edit_status])

    timer = gr.Timer(2)
    timer.tick(fn=update_ui, outputs=[status_box, log_box, scenes_md, shots_md, cast_md, gallery,
                                      final_video, verification_md])


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7810)
    args = parser.parse_args()
    demo.launch(server_name=os.environ.get("LTX_UI_HOST", "127.0.0.1"), server_port=args.port, theme=theme, inbrowser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
