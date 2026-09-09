# -*- coding: utf-8 -*-
"""StoryPlay 2.5 -- storyboard com QUADROS INTERMEDIARIOS visiveis, por cena.

Diferenca em relacao ao pipeline de screenplay existente: la, o estagio
`storyboard` gera UMA imagem por cena, usada apenas como primeiro frame do
clipe. Aqui, cada cena e decomposta nos seus planos (o `shot_list` que o
enriquecimento do parse ja produz: falas + acoes) e UMA IMAGEM E GERADA POR
PLANO. As imagens aparecem na interface para revisao antes de virar video.

Essas imagens nao sao so preview: entram na geracao como KEYFRAMES em posicoes
escolhidas do clipe, via LTXVAddGuideAdvanced (frame_idx + strength por guia),
que o LTX 2.5 suporta nativamente. Ou seja, o storyboard passa a dirigir o
video inteiro, e nao apenas o seu primeiro frame.

Fluxo:
  1) Texto (roteiro OU prosa/prompt LTX -- prosa e reestruturada automaticamente)
  2) Parse -> cenas + planos
  3) Storyboard -> uma imagem por plano, visivel na galeria
  4) Video -> por cena, com os quadros como keyframes

CLI: python -u storyplay25.py [--port 7911]
"""
from __future__ import annotations

import os
import argparse
import datetime
import json
import re
import sys
import traceback
from pathlib import Path

import gradio as gr
import video_doctor_ui  # aba de diagnostico temporal pos-geracao

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import ltx25_backend  # noqa: E402
from script_pipeline import generate_storyboards as sb  # noqa: E402
from script_pipeline import run_folder  # noqa: E402

# Style anchors. generate_storyboards.build_prompt() ends every prompt with a
# photoreal anchor, and its own comment explains why: without one, FLUX drifts
# into illustration style and "then set the look of the whole clip". So a style
# choice has to REPLACE that sentence, not be appended after it -- and the same
# anchor must also go into the LTX prompt, or the storyboard and the video pull
# in opposite directions.
PHOTOREAL_ANCHOR = "Photorealistic cinematic film still, live action footage, natural lighting."
STYLES = {
    "fotorrealista": PHOTOREAL_ANCHOR,
    "disney": (
        "3D animated feature film still in the Disney/Pixar house style, stylized "
        "appealing character design with large expressive eyes and soft rounded features, "
        "smooth subsurface-scattering skin, polished cinematic lighting with warm rim light, "
        "rich saturated colors, family animated film look."
    ),
    "anime": (
        "Hand-drawn Japanese anime film still, clean ink linework, cel shading, "
        "expressive eyes, painted background art, cinematic composition."
    ),
}

COMFY_SERVER = "http://127.0.0.1:8188"
DEFAULT_SB_CHECKPOINT = "flux-2-klein-9b-fp8.safetensors"
DEFAULT_SB_CLIP = "Qwen3-8B-FP8-native-bf16.safetensors"
DEFAULT_SB_VAE = "flux2-vae.safetensors"

STATE: dict = {"run_dir": None, "scenes": [], "cast": {}, "shots": {}}
LOG_LINES: list[str] = []


def log(msg: str) -> None:
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    LOG_LINES.append(line)
    print(line, flush=True)


def current_log() -> str:
    return "\n".join(LOG_LINES[-400:])


# --------------------------------------------------------------------------
# Etapa 1/2: parse
# --------------------------------------------------------------------------

def do_parse(script_text: str, engine: str):
    LOG_LINES.clear()
    if not script_text.strip():
        return "Cole um roteiro ou um texto/prompt descritivo primeiro.", current_log(), gr.update()

    # create_run() snapshots an existing script file into a fresh run folder, so
    # the text has to land on disk first (same order screenplay_ui.py uses).
    staging = ROOT / "outputs" / "storyplay25_input"
    staging.mkdir(parents=True, exist_ok=True)
    source_path = staging / f"storyplay_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
    source_path.write_text(script_text, encoding="utf-8")

    run_dir = Path(run_folder.create_run(str(source_path)))
    STATE["run_dir"] = run_dir
    snapshot = next((p for p in (run_dir / "input").glob("*.txt")), None)
    script_path = snapshot or source_path

    import subprocess

    cmd = [sys.executable, "-u", "-m", "script_pipeline.parse_screenplay",
           "--script", str(script_path), "--run-dir", str(run_dir),
           "--enrich-engine", engine]
    log(f"parse: {' '.join(cmd[-6:])}")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    for line in (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines():
        if line.strip() and "Loading weights" not in line and "it/s]" not in line:
            log(line.rstrip())
    if proc.returncode != 0:
        return "Parse falhou -- veja o log.", current_log(), gr.update()

    enriched = run_dir / "parse" / "scenes_enriched.json"
    structural = run_dir / "parse" / "scenes.json"
    path = enriched if enriched.exists() else structural
    scenes = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(scenes, dict):
        scenes = scenes.get("scenes", [])
    STATE["scenes"] = scenes

    # The cast stage writes each character's fixed visual descriptor, which
    # build_prompt() repeats in every shot's prompt -- that repetition is the only
    # identity-continuity mechanism this pipeline has (no IP-Adapter/face-lock,
    # per generate_storyboards' own note). Skipping it, as the first run did,
    # leaves cast.json absent and every beat free to invent a different person.
    cast_path = run_dir / "characters" / "cast.json"
    if not cast_path.exists():
        cast_cmd = [sys.executable, "-u", "-m", "script_pipeline.cast_characters",
                    "--run-dir", str(run_dir)]
        log("cast: gerando descritores de personagem...")
        cproc = subprocess.run(cast_cmd, cwd=str(ROOT), capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        for line in (cproc.stdout or "").splitlines() + (cproc.stderr or "").splitlines():
            if line.strip() and "Loading weights" not in line and "it/s]" not in line:
                log(line.rstrip())
    STATE["cast"] = json.loads(cast_path.read_text(encoding="utf-8")) if cast_path.exists() else {}
    if STATE["cast"]:
        log(f"cast: {len(STATE['cast'])} personagem(ns) com descritor.")
    else:
        log("cast: AVISO -- sem descritores; os quadros podem nao manter a identidade entre planos.")

    # Build the per-scene shot list. Falls back to one shot per dialogue line when
    # the enrichment produced no shot_list (e.g. --no-llm or a failed LLM call), so
    # the storyboard stage still has beats to draw.
    # Prefer beats read straight from the screenplay text: deterministic and
    # complete. The LLM's shot_list is the fallback (see _beats_from_screenplay).
    auto_path = run_dir / "parse" / "screenplay_auto.txt"
    sp_text = auto_path.read_text(encoding="utf-8") if auto_path.exists() else         Path(script_path).read_text(encoding="utf-8")
    try:
        by_scene = _beats_from_screenplay(sp_text)
    except Exception as e:
        log(f"extracao deterministica de planos falhou ({e}); usando shot_list do LLM.")
        by_scene = []

    shots: dict[int, list[dict]] = {}
    for pos, scene in enumerate(scenes):
        idx = scene["index"]
        if pos < len(by_scene) and by_scene[pos]:
            shots[idx] = by_scene[pos]
            log(f"cena {idx}: {len(by_scene[pos])} plano(s) lidos do roteiro "
                f"(shot_list do LLM tinha {len(scene.get('shot_list') or [])}).")
            continue
        sl = scene.get("shot_list") or []
        beats = []
        if sl:
            for shot in sl:
                if shot.get("type") == "dialogue":
                    li = shot.get("line_index", 0)
                    line = (scene.get("dialogue") or [{}])[li] if li < len(scene.get("dialogue") or []) else {}
                    beats.append({
                        "kind": "dialogue",
                        "character": line.get("character", ""),
                        "text": line.get("text", ""),
                        "visual": line.get("beat_visual") or scene.get("visual_prompt") or "",
                    })
                else:
                    beats.append({"kind": "action", "character": "",
                                  "text": "", "visual": shot.get("visual", "")})
        else:
            for line in scene.get("dialogue") or []:
                beats.append({"kind": "dialogue", "character": line.get("character", ""),
                              "text": line.get("text", ""),
                              "visual": line.get("beat_visual") or scene.get("visual_prompt") or ""})
            if not beats:
                beats.append({"kind": "action", "character": "", "text": "",
                              "visual": scene.get("visual_prompt") or scene.get("action_text", "")})
        shots[idx] = beats
    STATE["shots"] = shots

    total = sum(len(v) for v in shots.values())
    summary = [f"**{len(scenes)} cena(s), {total} plano(s)** — run: `{run_dir.name}`", ""]
    for scene in scenes:
        idx = scene["index"]
        summary.append(f"**Cena {idx}** — {scene.get('heading_raw') or scene.get('location', '')}")
        for i, beat in enumerate(shots[idx], 1):
            who = f"{beat['character']}: " if beat["character"] else ""
            body = beat["text"] or beat["visual"]
            # Dialogue is shown in FULL. Cutting it at a fixed width with no
            # ellipsis made a complete 238-char line look truncated, i.e. read as
            # data loss when nothing had been lost. Only long action/visual text
            # is shortened, and then with a visible marker.
            if not beat["text"] and len(body) > 160:
                body = body[:160] + "…"
            summary.append(f"  {i}. [{beat['kind']}] {who}{body}")
        summary.append("")
    return "\n".join(summary), current_log(), gr.update(choices=[s["index"] for s in scenes],
                                                       value=(scenes[0]["index"] if scenes else None))


# --------------------------------------------------------------------------
# Etapa 3: storyboard, uma imagem por plano
# --------------------------------------------------------------------------

def _beats_from_screenplay(text: str) -> list[list[dict]]:
    """Beats per scene, in document order, straight from the screenplay text.

    Preferred over the LLM's `shot_list` because it is deterministic and
    complete. MEASURED: for the same scene the enrichment returned 6 shots on
    one run and 5 on the next, and in the 5-shot run it kept ONE of the five
    action beats -- dropping "Thoren plants his feet and points his sword",
    arguably the scene's most cinematic moment. The screenplay itself has every
    beat, in order; only its position relative to each line is lost in
    Scene.action_text (which flattens the whole scene into one string).

    Recognises the same shape parse_screenplay's CUE_RE does: a lone uppercase
    name is a cue, the next non-blank line (after an optional parenthetical) is
    its dialogue, and any other non-empty line outside the heading is action.
    """
    cue_re = re.compile(r"^([A-ZÀ-Ý][A-ZÀ-Ý0-9 .'\-]{1,38})$")
    heading_re = re.compile(r"^\s*(INT|EXT|INT/EXT|I/E)[./]", re.IGNORECASE)

    lines = [ln.strip() for ln in text.splitlines()]
    scenes: list[list[dict]] = []
    beats: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if heading_re.match(line):
            # A new heading closes the previous scene, so beats stay grouped the
            # same way parse_structure groups them.
            if beats:
                scenes.append(beats)
            beats = []
            i += 1
            continue
        if not line:
            i += 1
            continue
        if cue_re.match(line):
            name = line
            j = i + 1
            parenthetical = ""
            while j < len(lines) and not lines[j]:
                j += 1
            if j < len(lines) and lines[j].startswith("(") and lines[j].endswith(")"):
                parenthetical = lines[j].strip("()")
                j += 1
                while j < len(lines) and not lines[j]:
                    j += 1
            if j < len(lines) and lines[j] and not cue_re.match(lines[j]):
                beats.append({"kind": "dialogue", "character": name,
                              "text": lines[j], "visual": parenthetical})
                i = j + 1
                continue
        beats.append({"kind": "action", "character": "", "text": "", "visual": line})
        i += 1
    if beats:
        scenes.append(beats)
    return scenes


# Beats that are bookkeeping, not pictures. "Lyra closes her mouth" / "becomes
# silent" exist in LTX prompts to mark where a line ends for lip-sync, and a
# camera instruction is already carried by the video prompt -- as storyboard
# frames they waste a slot and draw nothing. MEASURED: with a 4-frame limit,
# even spacing over the raw beat list picked "closes her mouth", "becomes
# silent" and zero dialogue beats.
_FILLER_RE = re.compile(
    r"^\s*(a\s+)?c[âa]mera\b"
    r"|fecha\s+a\s+boca"
    r"|fica\s+em\s+sil[êe]ncio"
    r"|closes?\s+(her|his|their)\s+mouth"
    r"|becomes?\s+silent",
    re.IGNORECASE,
)


def _is_filler(beat: dict) -> bool:
    return beat.get("kind") == "action" and bool(_FILLER_RE.search(beat.get("visual") or ""))


def _pick_beats(beats: list[dict], limit: int) -> list[dict]:
    """At most *limit* beats, evenly spaced and always keeping the first and last,
    so a reduced storyboard still spans the scene's arc instead of only its
    opening. Filler beats are dropped first (see _FILLER_RE)."""
    meaningful = [b for b in beats if not _is_filler(b)]
    beats = meaningful or beats  # never return nothing
    if limit <= 0 or len(beats) <= limit:
        return beats
    if limit == 1:
        return [beats[0]]
    step = (len(beats) - 1) / (limit - 1)
    idxs = sorted({int(round(i * step)) for i in range(limit)})
    return [beats[i] for i in idxs]


def _beat_prompt(scene: dict, cast: dict, beat: dict, style_anchor: str = PHOTOREAL_ANCHOR) -> str:
    """Scene-wide context + this beat's action, keeping every consistency device
    that generate_storyboards.build_prompt() provides.

    build_prompt() returns "<scene>. <cast descriptors>. <photoreal style anchor>".
    The beat action is inserted BEFORE the style anchor so the anchor stays last
    (it is what stops FLUX from drifting into illustration style), while the
    per-beat difference still reaches the model.
    """
    base = sb.build_prompt(scene, cast)
    # Swap build_prompt's photoreal anchor for the requested style. Replacing
    # rather than appending matters: two competing style sentences in one prompt
    # produced visibly mixed looks across beats in the first runs.
    if PHOTOREAL_ANCHOR in base:
        base = base.replace(PHOTOREAL_ANCHOR, style_anchor)
    else:
        base = f"{base.rstrip('. ')}. {style_anchor}"

    action = (beat.get("visual") or "").strip().rstrip(".")
    if not action:
        return base
    if style_anchor in base:
        head, tail = base.split(style_anchor, 1)
        return f"{head.rstrip('. ')}. {action}. {style_anchor}{tail}"
    return f"{base.rstrip('. ')}. {action}."


def do_storyboard(scene_index, sb_checkpoint, sb_clip, sb_vae, sb_width, sb_height, sb_steps, seed,
                  style_key="fotorrealista", max_beats=0):
    if not STATE["scenes"]:
        return [], "Rode o parse primeiro.", current_log()
    scene = next((s for s in STATE["scenes"] if s["index"] == scene_index), None)
    if scene is None:
        return [], "Cena nao encontrada.", current_log()

    beats = _pick_beats(STATE["shots"].get(scene_index, []), int(max_beats))
    style_anchor = STYLES.get(style_key, PHOTOREAL_ANCHOR)
    STATE.setdefault("style", {})[scene_index] = style_key
    log(f"cena {scene_index}: {len(beats)} plano(s) selecionado(s), estilo '{style_key}'.")
    run_dir = STATE["run_dir"]
    out_dir = run_dir / "storyboard" / f"scene_{scene_index:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sb.ensure_comfyui_running(COMFY_SERVER, log=log):
        return [], "ComfyUI nao subiu.", current_log()

    gallery, paths = [], []
    for i, beat in enumerate(beats, 1):
        # Each beat gets the FULL scene context plus its own action -- not the
        # action alone.
        #
        # MEASURED, first run: prompting only with the beat's short phrase
        # produced 8 unrelated images (black-and-white street, illustration,
        # photoreal close-up, empty doorway) with none of the described
        # characters, and the video then faithfully followed that incoherence.
        # generate_storyboards.build_prompt() exists precisely to prevent this:
        # it appends the cast descriptors (identity across shots) and a
        # photoreal style anchor whose own comment warns that scene text alone
        # "left FLUX free to answer in illustration style, which then set the
        # look of the whole clip". Overriding it threw all of that away.
        prompt = _beat_prompt(scene, STATE["cast"], beat, style_anchor)
        out_path = out_dir / f"beat_{i:02d}.png"
        log(f"cena {scene_index} plano {i}/{len(beats)}: gerando quadro...")
        ok = sb.generate_scene_storyboard(
            scene, STATE["cast"], server=COMFY_SERVER, checkpoint=sb_checkpoint,
            width=int(sb_width), height=int(sb_height), steps=int(sb_steps),
            cfg=1.0, seed=int(seed) + i, out_path=out_path, log=log,
            clip=sb_clip, vae=sb_vae, prompt_override=prompt,
        )
        if ok and out_path.exists():
            label = f"{i}. {beat['kind']}" + (f" — {beat['character']}" if beat["character"] else "")
            gallery.append((str(out_path), label))
            paths.append(str(out_path))
        else:
            log(f"cena {scene_index} plano {i}: falhou.")

    STATE.setdefault("keyframe_paths", {})[scene_index] = paths
    return gallery, f"{len(paths)} de {len(beats)} quadro(s) gerado(s).", current_log()


# --------------------------------------------------------------------------
# Etapa 4: video com os quadros como keyframes
# --------------------------------------------------------------------------

def do_video(scene_index, seconds, fps, width, height, variant, strength, seed, use_keyframes,
             style_key="fotorrealista"):
    paths = (STATE.get("keyframe_paths") or {}).get(scene_index, [])
    scene = next((s for s in STATE["scenes"] if s["index"] == scene_index), None)
    if scene is None:
        return None, "Rode o parse e o storyboard primeiro.", current_log()

    # 1 + multiple of 8, per the model's frame constraint. The parentheses matter:
    # round(x-1)/8 (rounding before dividing) yielded 713 for a 30s@24fps request
    # instead of 721 -- close enough to look right in a log, wrong in the output.
    num_frames = max(9, 1 + int(round((float(seconds) * float(fps) - 1) / 8)) * 8)

    keyframes = None
    if use_keyframes and paths:
        # Spread the beats evenly across the clip. frame_idx must be a multiple of
        # 8; the backend rounds, but computing it here keeps the spacing visible
        # in the log so an odd distribution is diagnosable.
        n = len(paths)
        keyframes = []
        for i, p in enumerate(paths):
            idx = 0 if i == 0 else int(round(i * (num_frames - 1) / n / 8)) * 8
            keyframes.append((p, idx, float(strength) if i else 1.0))
        log(f"keyframes: {[(Path(p).name, i, s) for p, i, s in keyframes]}")
    elif use_keyframes:
        log("nenhum quadro gerado ainda; gerando sem keyframes.")

    prompt = scene.get("visual_prompt") or scene.get("action_text") or ""
    lines = [f"{d.get('character','')}: “{d.get('text','')}”"
             for d in (scene.get("dialogue") or [])]
    if lines:
        prompt = prompt + " " + " ".join(lines)
    # The same anchor the storyboard used, or the video drifts to a different
    # look than the keyframes it is being steered by.
    prompt = f"{STYLES.get(style_key, PHOTOREAL_ANCHOR)} {prompt}"

    out_path = STATE["run_dir"] / "scenes" / f"scene_{scene_index:02d}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"gerando video da cena {scene_index}: {num_frames} frames, {width}x{height}, variante {variant}")
    try:
        ltx25_backend.generate(
            prompt, str(out_path), width=int(width), height=int(height),
            num_frames=num_frames, frame_rate=float(fps), seed=int(seed),
            variant=variant, keyframes=keyframes, log_cb=log, timeout=10800,
        )
    except Exception as e:
        log(f"ERRO: {e}")
        log(traceback.format_exc(limit=3))
        return None, f"Falhou: {e}", current_log()
    return str(out_path), f"Cena {scene_index} pronta ({num_frames} frames).", current_log()


# --------------------------------------------------------------------------
with gr.Blocks(title="StoryPlay 2.5 - storyboard com quadros intermediarios") as demo:
    gr.Markdown(
        "# StoryPlay 2.5\n"
        "Storyboard com **um quadro por plano**, visivel antes de virar video — e os quadros "
        "entram na geracao como **keyframes** (LTX 2.5 aceita condicionamento em posicoes "
        "intermediarias, nao so no primeiro frame).\n\n"
        "Aceita roteiro formatado **ou** prosa/prompt descritivo (reestruturado automaticamente)."
    )

    with gr.Row():
        with gr.Column(scale=3):
            script_text = gr.Textbox(label="Roteiro ou texto descritivo", lines=12,
                                     placeholder="Cole aqui um roteiro (INT./EXT.) ou um paragrafo descritivo...")
        with gr.Column(scale=1):
            # Ordem por MEDICAO, nao por preferencia (MEMORIAL.md 3.13): na mesma
            # entrada, qwen3.6 via Ollama preservou 4/4 falas em 6s e produziu 9
            # planos com beat_visual em 4/4 linhas; gemma4-e2b preservou 2/4 em
            # 91s, com 5 planos e beat_visual em 2/4.
            engine = gr.Dropdown(
                [("qwen3.6-35b (Ollama) -- recomendado", "qwen3.6-35b-a3b:latest"),
                 ("Gemma4 E2B (local, transformers)", "gemma4"),
                 ("Gemma3 12B (local, transformers)", "gemma3")],
                value="qwen3.6-35b-a3b:latest", allow_custom_value=True,
                label="LLM de estrutura/enriquecimento")
            parse_btn = gr.Button("1. Analisar texto", variant="primary")

    scenes_md = gr.Markdown("")

    with gr.Row():
        scene_pick = gr.Dropdown([], label="Cena", interactive=True)

    with gr.Tab("2. Storyboard (quadros intermediarios)"):
        with gr.Row():
            sb_checkpoint = gr.Textbox(value=DEFAULT_SB_CHECKPOINT, label="Checkpoint")
            sb_clip = gr.Textbox(value=DEFAULT_SB_CLIP, label="CLIP")
            sb_vae = gr.Textbox(value=DEFAULT_SB_VAE, label="VAE")
        with gr.Row():
            sb_width = gr.Number(value=768, label="Largura", precision=0)
            sb_height = gr.Number(value=512, label="Altura", precision=0)
            sb_steps = gr.Slider(1, 30, value=8, step=1, label="Steps")
            sb_seed = gr.Number(value=1234, label="Seed", precision=0)
        with gr.Row():
            style_pick = gr.Dropdown(list(STYLES.keys()), value="fotorrealista",
                                     label="Estetica")
            max_beats = gr.Slider(0, 12, value=4, step=1,
                                  label="Maximo de quadros intermediarios (0 = todos os planos)")
        sb_btn = gr.Button("2. Gerar quadros da cena", variant="primary")
        gallery = gr.Gallery(label="Quadros por plano", columns=4, height=340)
        sb_status = gr.Markdown("")

    with gr.Tab("3. Video da cena"):
        with gr.Row():
            v_seconds = gr.Slider(1, 60, value=10, step=1, label="Duracao (s)")
            v_fps = gr.Number(value=24, label="FPS", precision=0)
            v_width = gr.Number(value=768, label="Largura", precision=0)
            v_height = gr.Number(value=512, label="Altura", precision=0)
        with gr.Row():
            v_variant = gr.Radio(
                [("distilled (rapido)", "distilled"), ("dev (CFG real, lento)", "dev"),
                 ("distilled-int8 (20 GiB, int8)", "distilled-int8"),
                 ("redgraft (17 GiB, INT4 comunitario, mais rapido -- MEMORIAL 3.51)", "redgraft"),
                 ("w4a8-v10 (14,9 GiB, conversor oficial comfy-kitchen -- MEMORIAL 3.55)", "w4a8-v10")],
                value="distilled", label="Variante LTX 2.5")
            v_strength = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                   label="Forca dos keyframes intermediarios")
            v_seed = gr.Number(value=42, label="Seed", precision=0)
        use_kf = gr.Checkbox(value=True, label="Usar os quadros como keyframes")
        v_btn = gr.Button("3. Gerar video", variant="primary")
        video_out = gr.Video(label="Cena gerada")
        v_status = gr.Markdown("")

    log_box = gr.Textbox(label="Log", lines=14, max_lines=30, interactive=False)

    parse_btn.click(do_parse, [script_text, engine], [scenes_md, log_box, scene_pick])
    sb_btn.click(do_storyboard,
                 [scene_pick, sb_checkpoint, sb_clip, sb_vae, sb_width, sb_height, sb_steps, sb_seed,
                  style_pick, max_beats],
                 [gallery, sb_status, log_box])
    v_btn.click(do_video,
                [scene_pick, v_seconds, v_fps, v_width, v_height, v_variant, v_strength, v_seed, use_kf,
                 style_pick],
                [video_out, v_status, log_box])


    # Pos-producao: diagnostico e correcao temporal do video ja gerado.

    video_doctor_ui.build_doctor_tab(label="Diagnostico e correcao")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7911)
    args = ap.parse_args()
    demo.launch(server_name=os.environ.get("LTX_UI_HOST", "127.0.0.1"), server_port=args.port, inbrowser=False)
