"""Drop-in CLI shim: same arguments as `ltx_pipelines.distilled` /
`ltx_pipelines.music_to_video`, but generates with **LTX-2.5** through
`ltx25_backend` (ComfyUI route).

Why a shim instead of porting each UI's generation code: every UI in this repo
builds a `[sys.executable, "-m", "ltx_pipelines.<mod>", ...]` argv list and
streams the subprocess' stdout into its live log. Matching that contract means
each *_25 UI differs from its 2.3 original by the module name and the default
model paths -- a two-line diff -- instead of a rewrite of 700-1900 lines of
scene/audio/lip-sync logic that has nothing to do with the model version.

Usage mirrors the real pipelines:
    python -m ltx_pipelines_25 --prompt "..." --output-path out.mp4 \
        --width 768 --height 512 --num-frames 121 --frame-rate 24 --seed 42

Arguments that do not apply to the 2.5 ComfyUI route are ACCEPTED and IGNORED,
with a warning line on stdout, so the UIs keep working unmodified:
  --distilled-checkpoint-path / --gemma-root / --spatial-upsampler-path
        The 2.5 models are selected inside the ComfyUI workflow (and resolved
        through ComfyUI/extra_model_paths.yaml -> models/2.5/), not by path.
  --quantization / --torch-compile / --teacache-threshold
        Runtime knobs of the native pipeline; the ComfyUI graph has its own.
  --num-inference-steps
        The distilled 2.5 graph uses a fixed 8-step sigma schedule
        (ManualSigmas). A different value is reported, not applied.
  --lora
        The official single-stage 2.5 T2V graph has no LoRA loader node.

`--audio-input-path` IS honoured (since 2026-08-24): the track is encoded with
LTXVAudioVAEEncode and pinned by a per-modality noise mask, so the video is
generated on top of the real music instead of one the model invents. See
ltx25_backend._apply_audio_conditioning and MEMORIAL.md secao 3.15.

Conditioning images: 2.5's single-stage graph takes ONE first-frame image. If
several `--image` are passed, the one with the lowest latent index is used and
the rest are reported as skipped.
"""
import argparse
import os
import sys

import ltx25_backend


def main() -> int:
    ap = argparse.ArgumentParser(description="LTX-2.5 generation with an ltx_pipelines-compatible CLI")
    # --- honoured ---
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--num-frames", type=int, default=121)
    ap.add_argument("--frame-rate", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negative-prompt", default=ltx25_backend.DEFAULT_NEGATIVE)
    ap.add_argument("--disable-audio", action="store_true")
    ap.add_argument("--image", action="append", nargs=3,
                    metavar=("PATH", "LATENT_INDEX", "STRENGTH"), default=[])
    # Variant choice. The UIs build a fixed argv and know nothing about this, so
    # the practical selector for a UI run is the LTX25_VARIANT env var (which
    # ltx25_backend.DEFAULT_VARIANT reads); the flag is here for direct CLI use
    # and wins over the env var when given.
    ap.add_argument("--variant", choices=sorted(ltx25_backend.VARIANTS),
                    default=ltx25_backend.DEFAULT_VARIANT)
    ap.add_argument("--steps", type=int, default=None, help="passos (só dev; padrão 15)")
    ap.add_argument("--video-cfg", type=float, default=ltx25_backend.DEV_VIDEO_CFG)
    ap.add_argument("--audio-cfg", type=float, default=ltx25_backend.DEV_AUDIO_CFG)
    # --- accepted for compatibility, ignored ---
    ap.add_argument("--distilled-checkpoint-path", default=None)
    ap.add_argument("--checkpoint-path", default=None)
    ap.add_argument("--gemma-root", default=None)
    ap.add_argument("--spatial-upsampler-path", default=None)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--num-inference-steps", type=int, default=None)
    ap.add_argument("--torch-compile", action="store_true")
    ap.add_argument("--teacache-threshold", type=float, default=None)
    ap.add_argument("--enhance-prompt", action="store_true")
    ap.add_argument("--audio-input-path", default=None)
    ap.add_argument("--lora", action="append", nargs=2, metavar=("PATH", "STRENGTH"), default=[])
    ap.add_argument("--sampler", default=None)
    ap.add_argument("--skip-stage-2", action="store_true")
    # Two-stage is opt-in. NOT derived from --spatial-upsampler-path: the 2.3
    # UIs pass that path on every run, so keying off it would silently make
    # every existing UI render several times slower.
    ap.add_argument("--two-stage", action="store_true",
                    default=ltx25_backend.TWO_STAGE_DEFAULT,
                    help="upscale latente x2 + refino (bem mais lento, muito mais detalhe)")
    args, unknown = ap.parse_known_args()

    print("[ltx25-shim] gerando com LTX-2.5 via ComfyUI "
          "(ltx_pipelines não suporta 2.5 -- ver MEMORIAL.md).", flush=True)
    if unknown:
        print(f"[ltx25-shim] argumentos desconhecidos ignorados: {unknown}", flush=True)

    print(f"[ltx25-shim] variante: {args.variant}", flush=True)

    ignored = []
    if args.quantization:
        ignored.append(f"--quantization {args.quantization}")
    if args.variant == "distilled" and args.num_inference_steps not in (None, 8):
        ignored.append(f"--num-inference-steps {args.num_inference_steps} "
                       "(distilled usa 8 passos fixos; use --variant dev para variar)")
    if args.torch_compile:
        ignored.append("--torch-compile")
    if args.teacache_threshold is not None:
        ignored.append(f"--teacache-threshold {args.teacache_threshold}")
    if args.lora:
        ignored.append(f"--lora x{len(args.lora)} (grafo 2.5 single-stage não tem loader de LoRA)")
    if args.audio_input_path and not os.path.exists(args.audio_input_path):
        ignored.append(f"--audio-input-path (arquivo não encontrado: {args.audio_input_path})")
        args.audio_input_path = None
    if args.enhance_prompt:
        ignored.append("--enhance-prompt (enhancer do grafo desligado)")
    if ignored:
        print("[ltx25-shim] ignorado nesta rota: " + "; ".join(ignored), flush=True)

    num_frames = int(args.num_frames)
    if (num_frames - 1) % 8 != 0:
        adjusted = 1 + max(0, round((num_frames - 1) / 8)) * 8
        print(f"[ltx25-shim] num_frames={num_frames} não é 1+múltiplo de 8 "
              f"(exigência do modelo); ajustando para {adjusted}.", flush=True)
        num_frames = adjusted

    image_path = None
    image_strength = 1.0
    if args.image:
        ordered = sorted(args.image, key=lambda t: int(float(t[1])))
        image_path = ordered[0][0]
        # The caller's third field is the conditioning strength. It used to be
        # dropped here, which mattered less than it looks: the graph itself was
        # pinning strength to 0 -- see ltx25_backend, N_IMG2VID.
        image_strength = max(0.0, min(1.0, float(ordered[0][2])))
        print(f"[ltx25-shim] imagem de condicionamento (primeiro frame, "
              f"força {image_strength}): {image_path}", flush=True)
        if len(ordered) > 1:
            print(f"[ltx25-shim] {len(ordered)-1} imagem(ns) adicional(is) ignorada(s): "
                  "o grafo 2.5 single-stage aceita apenas o primeiro frame.", flush=True)

    try:
        out = ltx25_backend.generate(
            args.prompt,
            args.output_path,
            negative=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_frames=num_frames,
            frame_rate=args.frame_rate,
            seed=args.seed,
            image_path=image_path,
            image_strength=image_strength,
            disable_audio=args.disable_audio,
            variant=args.variant,
            # On dev the UI's own step count is meaningful (real CFG ramp), so
            # honour --num-inference-steps there; --steps still wins if given.
            steps=args.steps or (args.num_inference_steps if args.variant == "dev" else None),
            video_cfg=args.video_cfg,
            audio_cfg=args.audio_cfg,
            audio_conditioning=args.audio_input_path,
            two_stage=args.two_stage,
            log_cb=lambda m: print(m, flush=True),
        )
    except Exception as e:
        print(f"[ltx25-shim] ERRO: {e}", file=sys.stderr, flush=True)
        return 1

    print(f"[ltx25-shim] OK -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
