"""Screenplay -> video pipeline (v1) -- top-level CLI orchestrator.

Turns a plain-text screenplay into a movie by running nine stages in order, each as
its own subprocess (matching this project's established convention of isolating
heavy-model stages so VRAM from one is fully released before the next starts):

  parse -> cast -> storyboard -> dialogue -> render -> lipsync -> mix -> assemble -> verify

The last one renders nothing: it inspects the finished movie and reports defects that
looked fine at every individual stage. It exists because two separate silent films
shipped as "successful" runs -- ffprobe showed a healthy AAC track both times, and only
measuring the actual volume revealed -91 dB. Skip it with --no-verify.

Each stage writes its artifacts into a run folder (outputs/screenplay/<timestamp>_
<slug>/) and marks itself complete in generation_manifest.json, so a re-run with
--resume-from skips whatever's already done -- useful after hand-editing cast.json
or a scene's visual_prompt between stages.

Usage:
  python screenplay_to_video.py --script roteiro.txt
  python screenplay_to_video.py --script roteiro.txt --stage cast --llm
  python screenplay_to_video.py --resume-from outputs/screenplay/20260101_120000_x --stage render
  python screenplay_to_video.py --resume-from outputs/screenplay/20260101_120000_x --stage all --force

See script_pipeline/examples/roteiro_exemplo.txt for a short sample script and the
expected INT./EXT. heading + ALL-CAPS character cue + (parenthetical) format the
structural parser (script_pipeline/parse_screenplay.py) expects.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

STAGE_ORDER = ["parse", "cast", "storyboard", "dialogue", "render", "lipsync", "mix",
               "assemble", "verify"]
STAGE_MODULES = {
    "parse": "script_pipeline.parse_screenplay",
    "cast": "script_pipeline.cast_characters",
    "storyboard": "script_pipeline.generate_storyboards",
    "dialogue": "script_pipeline.synthesize_dialogue",
    "render": "script_pipeline.render_scenes",
    "lipsync": "script_pipeline.lipsync_scenes",
    "mix": "script_pipeline.mix_audio",
    "assemble": "script_pipeline.assemble_final",
    "verify": "script_pipeline.verify_output",
}

# verify inspects the finished movie instead of producing one. It runs non-strict by
# default so a defect is REPORTED without destroying the run's completion state -- the
# operator still gets the file and can judge it. Pass --strict-verify for CI-style use
# where a silent or truncated film should fail the whole invocation.


def find_saved_script(run_dir: Path) -> str:
    """run_folder.create_run() copies the source screenplay into run_dir/input/ --
    that copy is what every re-run of the parse stage (including a plain
    --resume-from --stage parse, or --stage all after --force) must read from,
    since the ORIGINAL path the user passed to --script may no longer exist
    (e.g. a UI's scratch textbox file) and shouldn't need to."""
    candidates = sorted((run_dir / "input").glob("*"))
    if not candidates:
        raise FileNotFoundError(f"No saved screenplay found in {run_dir / 'input'}")
    return str(candidates[0])


def run_stage(stage: str, run_dir: Path, args: argparse.Namespace, *, log) -> bool:
    argv = [sys.executable, "-u", "-m", STAGE_MODULES[stage], "--run-dir", str(run_dir)]

    if stage == "parse":
        argv += ["--script", find_saved_script(run_dir)]
        if args.no_llm:
            argv.append("--no-llm")
        argv += ["--language", args.language,
                 "--translate-scenes-en" if args.translate_scenes_en else "--no-translate-scenes-en",
                 "--enrich-engine", args.enrich_engine]
    elif stage == "cast":
        # O `--llm` carrega o Gemma 3 EM PROCESSO. O `--engine` usa o Ollama, que
        # ja esta no ar para o parse e escreve descritores melhores (MEMORIAL
        # 3.29): sem nenhum dos dois, o descritor e um recorte do texto de acao e
        # so vira aparencia se o roteiro apresentar o personagem entre
        # parenteses. Padrao continua sendo nenhum dos dois -- comportamento
        # validado -- e quem quiser o melhor pede.
        if args.llm:
            argv.append("--llm")
        elif args.cast_engine:
            argv += ["--engine", args.cast_engine]
    elif stage == "storyboard":
        # Com --image-engine, os padroes do motor (checkpoint, encoders, passos,
        # CFG) vem de generate_storyboards.IMAGE_ENGINES e os valores explicitos
        # daqui SAEM do caminho -- passar o checkpoint do FLUX junto com
        # --image-engine sd35 pediria o arquivo errado. Sem a flag nada muda.
        if args.image_engine:
            argv += ["--image-engine", args.image_engine,
                     "--width", str(args.storyboard_width),
                     "--height", str(args.storyboard_height),
                     "--per-shot" if args.storyboard_per_shot else "--per-scene"]
        else:
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

    log(f"\n=== ETAPA: {stage} ===")
    # encoding/errors explicit: subprocess.Popen(text=True) without them defaults to
    # the console's codepage (cp1252 on Windows) -- MEASURED to crash the whole
    # orchestrator with UnicodeDecodeError when a child (e.g. a stray multilingual
    # LLM enrichment response) prints a byte sequence cp1252 can't represent.
    process = subprocess.Popen(argv, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, universal_newlines=True,
                                encoding="utf-8", errors="replace")
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    return process.returncode == 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", help="Path to a screenplay text file (starts a new run).")
    parser.add_argument("--resume-from", help="Existing run folder (skip already-completed stages).")
    parser.add_argument("--stage", default="all", choices=STAGE_ORDER + ["all"])
    parser.add_argument("--force", action="store_true", help="Re-run stages even if already marked complete.")

    # parse / cast
    parser.add_argument("--no-llm", action="store_true", help="parse: skip Gemma3 enrichment (structure only).")
    parser.add_argument("--llm", action="store_true", help="cast: use Gemma3 for richer character descriptors.")
    parser.add_argument("--cast-engine", default=None,
                        help="cast: tag de modelo do Ollama para escrever os descritores "
                             "visuais (ex.: qwen3.6-35b-a3b:latest). Melhor e muito mais "
                             "barato que --llm, que carrega o Gemma 3 em processo. "
                             "Ver MEMORIAL.md 3.29.")
    parser.add_argument("--image-engine", default=None, choices=["flux", "sd35", "sdxl"],
                        help="storyboard: motor de imagem. Sem isto usa os "
                             "--storyboard-* explicitos, que e o comportamento "
                             "validado. sd35 carrega em ~1 min contra ~4 e cabe em "
                             "~12 GB contra ~24, mas obedece menos o enquadramento.")

    # storyboard (kept at 2x the render width/height below, per user instruction --
    #  sharper source to downscale into the video, same aspect ratio)
    parser.add_argument("--storyboard-checkpoint", default="flux-2-klein-9b-fp8.safetensors")
    parser.add_argument("--storyboard-width", type=int, default=1792)
    parser.add_argument("--storyboard-height", type=int, default=1024)
    parser.add_argument("--storyboard-steps", type=int, default=8)
    parser.add_argument("--storyboard-clip", default="Qwen3-8B-FP8-native-bf16.safetensors")
    parser.add_argument("--storyboard-vae", default="flux2-vae.safetensors")
    parser.add_argument("--storyboard-guidance", type=float, default=3.5)
    parser.add_argument("--storyboard-per-shot", dest="storyboard_per_shot", action="store_true", default=True)
    parser.add_argument("--storyboard-per-scene", dest="storyboard_per_shot", action="store_false")

    # language (drives both parse's scene-description language and dialogue's TTS voice
    # language -- dialogue TEXT itself is never translated regardless of these flags)
    parser.add_argument("--language", default="pt", choices=["pt", "en", "es", "fr", "de", "ja", "ko"],
                         help="Screenplay/voice language, default pt = Brazilian Portuguese.")
    parser.add_argument("--translate-scenes-en", dest="translate_scenes_en", action="store_true", default=True,
                         help="Write scene/camera visual_prompt in English (default on -- FLUX/LTX read English prompts better). Dialogue text stays in --language either way.")
    parser.add_argument("--no-translate-scenes-en", dest="translate_scenes_en", action="store_false")
    # gemma4 (E2B, isolated venv) is the default because gemma3 MEASURED as unusable for
    # this task: greedy decoding returns multilingual token salad instead of JSON, e.g.
    #   'reso ...  repart... unresponsive... Yn ...cnicas...'
    # The parse stage then falls back to the structural prompt, which silently cascades --
    # no visual_prompt, no shot_list, one storyboard per scene instead of per shot, and a
    # render prompt made of the RAW screenplay text (the original cause of the "bizarre
    # scenes"). Keeping gemma3 as the default meant every plain CLI run reproduced that.
    parser.add_argument("--enrich-engine", default="gemma4", choices=["gemma3", "gemma4"])

    # dialogue
    parser.add_argument("--tts-engine", default="auto", choices=["auto", "xtts", "qwen"])

    # render
    parser.add_argument("--checkpoint", default="./models/ltx-2.3-22b-distilled-fp8.safetensors")
    parser.add_argument("--gemma-root", default="./models/gemma3")
    parser.add_argument("--upsampler", default="./models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-clip-seconds", type=float, default=5.0)
    parser.add_argument("--camera-movement", default=None)
    parser.add_argument("--action-beat-seconds", type=float, default=3.0)
    parser.add_argument("--end-keyframe-strength", type=float, default=0.0)
    parser.add_argument("--ambient-audio", action="store_true",
                        help="render: let LTX generate ambient sound for clips that have no dialogue "
                             "(otherwise those clips are silent and drag the film's volume down).")
    parser.add_argument("--engine", default="ltx", choices=["ltx", "wan"],
                         help="Video backend: ltx (native, audio-aware) or wan (ComfyUI Wan 2.1/2.2, silent + muxed audio).")
    parser.add_argument("--wan-checkpoint", default="")
    parser.add_argument("--wan-clip", default="")
    parser.add_argument("--wan-vae", default="")
    parser.add_argument("--wan-weight-dtype", default="default")
    parser.add_argument("--wan-cfg", type=float, default=5.0)
    parser.add_argument("--wan-first-last", action="store_true")
    parser.add_argument("--chain-continuity", dest="chain_continuity", action="store_true", default=True)
    parser.add_argument("--no-chain-continuity", dest="chain_continuity", action="store_false")
    parser.add_argument("--default-scene-seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1234)

    # lipsync
    parser.add_argument("--lipsync-engine", default="auto", choices=["auto", "latentsync", "wav2lip"])

    # mix
    parser.add_argument("--room-preset", default="none")
    parser.add_argument("--room-distance", type=float, default=0.0)
    parser.add_argument("--ambient-track", default=None)
    parser.add_argument("--ambient-volume", type=float, default=0.25)

    # assemble
    parser.add_argument("--output", default="movie.mp4")

    # verify (post-render self-review)
    parser.add_argument("--min-audio-db", type=float, default=-60.0,
                        help="verify: below this mean volume the film counts as mute.")
    parser.add_argument("--strict-verify", action="store_true",
                        help="verify: make the run fail when a defect is found (default: report only).")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the post-render verification stage in --stage all.")

    args = parser.parse_args(argv)

    if not args.script and not args.resume_from:
        parser.error("Either --script (new run) or --resume-from (continue a run) is required.")

    from script_pipeline import run_folder

    if args.resume_from:
        run_dir = Path(args.resume_from).resolve()
        if not run_dir.is_dir():
            parser.error(f"--resume-from directory does not exist: {run_dir}")
    else:
        run_dir = run_folder.create_run(args.script)
        print(f"Nova execucao: {run_dir}")

    stages = STAGE_ORDER if args.stage == "all" else [args.stage]
    if args.no_verify:
        stages = [s for s in stages if s != "verify"]
    log = lambda msg: run_folder.append_log(run_dir, msg)  # noqa: E731

    for stage in stages:
        if not args.force and run_folder.stage_complete(run_dir, stage):
            log(f"Etapa '{stage}' ja concluida; pulando (use --force para refazer).")
            continue
        ok = run_stage(stage, run_dir, args, log=log)
        if not ok:
            log(f"Etapa '{stage}' falhou. Corrija e rode de novo com --resume-from {run_dir} --stage {stage}.")
            return 1

    log(f"\nConcluido. Pasta de execucao: {run_dir}")
    final_video = run_dir / "final" / args.output
    if final_video.exists():
        log(f"Video final: {final_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
