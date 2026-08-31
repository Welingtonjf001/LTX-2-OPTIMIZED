"""Patch the API-format LTX-2.3 IC-LoRA *Union-Control* workflow to run from a
ChoreoEngine pose bundle on the local GGUF stack.

The official workflow (example_workflows/2.3/LTX-2.3_ICLoRA_Union_Control_
Distilled.json) derives the control video with DWPreprocessor / DepthAnything /
Canny nodes that are NOT installed here. That is fine: the ChoreoEngine bundle
already ships a DWPose-rendered pose.mp4, so we feed it straight into the
IC-LoRA guide and drop the preprocessor branch.

Model chain (same as the validated Ingredients flow):
    GGUF UNET -> distilled LoRA (few-step) -> Union-Control IC-LoRA.

Usage:
  python comfy_union_patch.py --pose E:/path/bundle/pose.mp4 \
      --prompt "..." [--out _api_union_test.json] [--strength 1.0]
  python comfy_run.py --workflow _api_union_test.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from comfy_gguf_patch import GGUF_LOADER_ID, VAE_LOADER_ID, prune_orphans, rewire

BASE = "_api_union_base.json"
COMFY_INPUT = Path("ComfyUI/input")

# node ids do workflow oficial Union-Control (apos conversao UI->API)
CKPT_NODE = "3940"            # CheckpointLoaderSimple (checkpoint dev de 46GB)
DISTILLED_LORA_NODE = "4922"  # LoraLoaderModelOnly (few-step)
ICLORA_NODE = "5011"          # LTXICLoRALoaderModelOnly (Union-Control)
GUIDE_NODE = "5012"           # LTXAddVideoICLoRAGuide
LOAD_VIDEO_NODE = "5001"      # LoadVideo (era o video de referencia)
VIDEO_COMPONENTS_NODE = "5000"  # GetVideoComponents
GUIDE_RESIZE_NODE = "5028"    # ResizeImageMaskNode (era alimentado pelo DWPreprocessor)
TEXT_ENC_NODE = "5023"        # LTXAVTextEncoderLoader
AUDIO_VAE_NODE = "4010"       # LTXVAudioVAELoader
POSITIVE_NODE = "2483"
NEGATIVE_NODE = "2612"
BYPASS_IMG_NODE = "5019"      # PrimitiveBoolean: bypass do frame inicial
START_RESIZE_NODE = "5035"    # ResizeImageMaskNode do frame inicial (mesmo DynamicCombo)
LOAD_IMAGE_NODE = "2004"      # LoadImage (frame inicial opcional)
NOISE_NODE = "4832"           # RandomNoise (noise_seed)
SAVE_NODE = "4852"

DEFAULT_GGUF = "ltx-2.3-22b-dev-Q6_K.gguf"
DEFAULT_VAE = "vae\\ltx-2.3-22b-dev_video_vae.safetensors"
DEFAULT_DISTILLED_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
DEFAULT_DISTILLED_STRENGTH = 0.5  # o unico valor validado ponta a ponta aqui
DEFAULT_UNION_LORA = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
# valores do fluxo Ingredients validado ponta a ponta nesta maquina:
# o tokenizer vem junto do gemma re-empacotado; os embeddings connectors sao
# lidos do checkpoint dev. Os defaults do comfy_gguf_patch (shard 00001 +
# connectors avulsos) dao "invalid tokenizer" neste no.
DEFAULT_TEXT_ENCODER = "gemma_3_12B_it.safetensors"
DEFAULT_CONNECTORS = "ltx-2.3-22b-dev.safetensors"
DEFAULT_AUDIO_CKPT = "ltx-2.3-22b-distilled-fp8.safetensors"
DEFAULT_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly, blurry"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True, help="pose.mp4 do bundle do ChoreoEngine")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default="_api_union_test.json")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="forca do guia (LTXAddVideoICLoRAGuide). TETO 1.0 -- o no "
                         "rejeita valores maiores na validacao")
    ap.add_argument("--iclora-strength", type=float, default=1.0,
                    help="peso da IC-LoRA no modelo (LTXICLoRALoaderModelOnly). "
                         "Este sim aceita >1 (max 100); e o knob real de intensidade")
    ap.add_argument("--union-lora", default=DEFAULT_UNION_LORA)
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--distilled-lora", default=DEFAULT_DISTILLED_LORA)
    ap.add_argument("--distilled-strength", type=float, default=DEFAULT_DISTILLED_STRENGTH)
    ap.add_argument("--start-image", default="", help="frame inicial opcional (png)")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42,
                    help="noise_seed. O workflow oficial vem com 42 fixo: sem variar "
                         "isto, um sweep de parametros mede UMA amostra so")
    ap.add_argument("--prefix", default="union_test")
    args = ap.parse_args()

    api = json.load(open(args.base, encoding="utf-8"))

    # --- modelo: GGUF -> distilled LoRA -> IC-LoRA -------------------------
    api[GGUF_LOADER_ID] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": args.gguf}}
    api[VAE_LOADER_ID] = {"class_type": "VAELoader", "inputs": {"vae_name": DEFAULT_VAE}}
    api[DISTILLED_LORA_NODE]["inputs"]["model"] = [GGUF_LOADER_ID, 0]
    api[DISTILLED_LORA_NODE]["inputs"]["lora_name"] = args.distilled_lora
    api[DISTILLED_LORA_NODE]["inputs"]["strength_model"] = args.distilled_strength
    api[ICLORA_NODE]["inputs"]["lora_name"] = args.union_lora
    api[ICLORA_NODE]["inputs"]["strength_model"] = args.iclora_strength
    rewire(api, CKPT_NODE, 0, [DISTILLED_LORA_NODE, 0])
    rewire(api, CKPT_NODE, 2, [VAE_LOADER_ID, 0])
    api.pop(CKPT_NODE, None)

    api[TEXT_ENC_NODE]["inputs"]["text_encoder"] = DEFAULT_TEXT_ENCODER
    api[TEXT_ENC_NODE]["inputs"]["ckpt_name"] = DEFAULT_CONNECTORS
    if AUDIO_VAE_NODE in api:
        api[AUDIO_VAE_NODE]["inputs"]["ckpt_name"] = DEFAULT_AUDIO_CKPT

    # --- guia: nosso pose.mp4 no lugar do branch DWPreprocessor ------------
    pose_src = Path(args.pose)
    if not pose_src.exists():
        raise SystemExit(f"pose nao existe: {pose_src}")
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    # O nome PRECISA ser unico por execucao. Todo bundle tem um "pose.mp4",
    # entao um nome derivado do stem colide: execucoes concorrentes sobrescrevem
    # o input uma da outra e o ComfyUI ainda cacheia por nome de arquivo --
    # uma geracao de 97 frames chegou a sair com 25 porque leu o pose de outro
    # job. O prefix ja carrega bundle+tag, e o hash cobre o resto.
    digest = hashlib.sha1(
        f"{pose_src.resolve()}|{pose_src.stat().st_mtime_ns}".encode()
    ).hexdigest()[:8]
    pose_name = f"choreo_{args.prefix}_{digest}.mp4"
    shutil.copy2(pose_src, COMFY_INPUT / pose_name)

    api[LOAD_VIDEO_NODE]["inputs"]["file"] = pose_name
    # O ResizeImageMaskNode usa um widget DynamicCombo que o conversor generico
    # mapeia errado (mesma armadilha ja anotada no patch do Ingredients).
    # Reescrevemos os inputs por inteiro no formato do /object_info:
    # o guia le os frames do proprio pose.mp4, ja renderizado na resolucao alvo,
    # entao um resize exato para as dimensoes do bundle e o mais previsivel.
    api[GUIDE_RESIZE_NODE]["inputs"] = {
        "input": [VIDEO_COMPONENTS_NODE, 0],
        "scale_method": "lanczos",  # required a parte do combo (interpolacao)
        "resize_type": "scale dimensions",
        "resize_type.width": args.width,
        "resize_type.height": args.height,
        "resize_type.crop": "center",
    }
    if args.strength > 1.0:
        raise SystemExit(
            f"--strength {args.strength} excede o teto 1.0 do LTXAddVideoICLoRAGuide.\n"
            f"  Para intensificar o controle use --iclora-strength (aceita ate 100)."
        )
    api[GUIDE_NODE]["inputs"]["strength"] = args.strength

    # --- frame inicial opcional --------------------------------------------
    if args.start_image:
        img = Path(args.start_image)
        img_name = f"choreo_{args.prefix}_{img.name}"  # mesma regra: nome unico
        shutil.copy2(img, COMFY_INPUT / img_name)
        api[LOAD_IMAGE_NODE]["inputs"]["image"] = img_name
        api[BYPASS_IMG_NODE]["inputs"]["value"] = False
    else:
        # sem frame inicial: bypass ativo. O LoadImage precisa apontar para um
        # arquivo existente mesmo em bypass (validacao roda antes da execucao),
        # entao usamos um frame do proprio pose video.
        api[BYPASS_IMG_NODE]["inputs"]["value"] = True
        placeholder = COMFY_INPUT / "choreo_placeholder.png"
        if not placeholder.exists():
            import cv2  # noqa: PLC0415

            cap = cv2.VideoCapture(str(pose_src))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise SystemExit("nao consegui ler o primeiro frame do pose.mp4")
            cv2.imwrite(str(placeholder), frame)
        api[LOAD_IMAGE_NODE]["inputs"]["image"] = placeholder.name

    # mesmo conserto de DynamicCombo no resize do frame inicial (5035)
    if START_RESIZE_NODE in api:
        api[START_RESIZE_NODE]["inputs"] = {
            "input": api[START_RESIZE_NODE]["inputs"].get("input", [LOAD_IMAGE_NODE, 0]),
            "scale_method": "lanczos",
            "resize_type": "scale dimensions",
            "resize_type.width": args.width,
            "resize_type.height": args.height,
            "resize_type.crop": "center",
        }

    api[POSITIVE_NODE]["inputs"]["text"] = args.prompt
    api[NEGATIVE_NODE]["inputs"]["text"] = args.negative
    api[NOISE_NODE]["inputs"]["noise_seed"] = args.seed
    api[SAVE_NODE]["inputs"]["filename_prefix"] = args.prefix

    api = prune_orphans(api, SAVE_NODE)
    json.dump(api, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"[patch] {args.out}: {len(api)} nos | guia={pose_name} | "
          f"iclora={args.union_lora} strength={args.strength}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
