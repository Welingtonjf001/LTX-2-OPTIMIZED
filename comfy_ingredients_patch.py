"""Patch the API-format LTX-2.3 IC-LoRA "Ingredients" workflow to run from a
GGUF UNET using only the model files present in this installation.

Reference-image-conditioned generation: given ONE reference image (a person,
object, or scene composite), the IC-LoRA "Ingredients" guides the video's
appearance toward that reference. Callers may supply an exact target resolution;
the original 544-pixel-short-edge behavior remains the CLI fallback.

Model chain: GGUF UNET -> distilled LoRA (few-step) -> Ingredients IC-LoRA.
Sampler is already KSamplerSelect natively in this workflow (no ClownSampler
substitution needed, unlike the plain T2V/Full workflow).
"""
import argparse
import json
import shutil
import sys

CKPT_NODE = "3940"          # CheckpointLoaderSimple (absent model)
DISTILLED_LORA_NODE = "4922"  # LoraLoaderModelOnly (distilled few-step lora)
ICLORA_NODE = "5011"        # LTXICLoRALoaderModelOnly (Ingredients IC-LoRA)
TEXT_ENC_NODE = "5023"      # LTXAVTextEncoderLoader
AUDIO_VAE_NODE = "4010"     # LTXVAudioVAELoader
LOAD_IMAGE_NODE = "2004"    # LoadImage (the reference "ingredient")
VIDEO_LENGTH_NODE = "5072"  # PrimitiveInt "Video Length"
SAVE_NODE = "4852"          # SaveVideo
POSITIVE_CLIP_NODE = "2483"  # CLIPTextEncode (positive)
RESIZE_NODE = "5069"         # ResizeImageMaskNode (DynamicCombo widget mismapped by the generic converter)

GGUF_LOADER_ID = "91001"
VAE_LOADER_ID = "91002"

# Named defaults (also used by gguf_backend.py -- keep in one place so the CLI
# tool and the UI backend never drift apart).
DEFAULT_GGUF = "ltx-2.3-22b-dev-Q6_K.gguf"  # less aggressive quantization than UD-Q5_K_S
DEFAULT_VAE = "vae\\ltx-2.3-22b-dev_video_vae.safetensors"
DEFAULT_DISTILLED_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
DEFAULT_DISTILLED_LORA_STRENGTH = 0.5
DEFAULT_INGREDIENTS_LORA = "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
DEFAULT_INGREDIENTS_LORA_STRENGTH = 1.0  # the only value validated end-to-end so far
DEFAULT_AUDIO_CKPT = "ltx-2.3-22b-distilled-fp8.safetensors"
DEFAULT_TEXT_ENCODER = "gemma_3_12B_it.safetensors"
DEFAULT_CONNECTORS = "text_encoders\\ltx-2.3-22b-dev_embeddings_connectors.safetensors"


def resize_inputs(link, width: int | None = None, height: int | None = None) -> dict:
    if width and height:
        return {
            "input": link,
            "resize_type": "scale dimensions",
            "resize_type.width": int(width),
            "resize_type.height": int(height),
            "resize_type.crop": "center",
            "scale_method": "lanczos",
        }
    return {
        "input": link,
        "resize_type": "scale shorter dimension",
        "resize_type.shorter_size": 544,
        "scale_method": "lanczos",
    }


def rewire(api: dict, old_id: str, old_slot, new_ref: list) -> int:
    n = 0
    for node in api.values():
        for k, v in list(node["inputs"].items()):
            if isinstance(v, list) and len(v) == 2 and v[0] == old_id:
                if old_slot is None or v[1] == old_slot:
                    node["inputs"][k] = list(new_ref)
                    n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--distilled-lora", default=DEFAULT_DISTILLED_LORA)
    ap.add_argument("--distilled-lora-strength", type=float, default=DEFAULT_DISTILLED_LORA_STRENGTH)
    ap.add_argument("--ingredients-lora", default=DEFAULT_INGREDIENTS_LORA)
    ap.add_argument("--ingredients-lora-strength", type=float, default=DEFAULT_INGREDIENTS_LORA_STRENGTH)
    ap.add_argument("--audio-ckpt", default=DEFAULT_AUDIO_CKPT)
    ap.add_argument("--text-encoder", default=DEFAULT_TEXT_ENCODER)
    ap.add_argument("--connectors", default=DEFAULT_CONNECTORS)
    ap.add_argument("--reference-image", required=True, help="path to the reference/ingredient image on disk")
    ap.add_argument("--comfy-input-dir", default="ComfyUI/input")
    ap.add_argument("--prompt")
    ap.add_argument("--frames", type=int, help="override the default 241-frame Video Length primitive")
    ap.add_argument("--width", type=int, help="resize the reference and output latent to this width")
    ap.add_argument("--height", type=int, help="resize the reference and output latent to this height")
    ap.add_argument("--prefix", default="ingredients_test")
    args = ap.parse_args()

    api = json.load(open(args.inp, encoding="utf-8"))

    # --- copy the reference image into ComfyUI's input/ dir (LoadImage reads
    #     by filename from there, not by absolute path) -----------------------
    import os
    os.makedirs(args.comfy_input_dir, exist_ok=True)
    ref_filename = os.path.basename(args.reference_image)
    dest = os.path.join(args.comfy_input_dir, ref_filename)
    if os.path.abspath(dest) != os.path.abspath(args.reference_image):
        shutil.copy2(args.reference_image, dest)

    # --- new source nodes ---------------------------------------------------
    api[GGUF_LOADER_ID] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": args.gguf}}
    api[VAE_LOADER_ID] = {"class_type": "VAELoader", "inputs": {"vae_name": args.vae}}

    # --- model chain: GGUF -> distilled LoRA -> Ingredients IC-LoRA --------
    n_model = rewire(api, CKPT_NODE, 0, [GGUF_LOADER_ID, 0])
    if DISTILLED_LORA_NODE in api:
        api[DISTILLED_LORA_NODE]["inputs"]["lora_name"] = args.distilled_lora
        api[DISTILLED_LORA_NODE]["inputs"]["strength_model"] = args.distilled_lora_strength
    if ICLORA_NODE in api:
        api[ICLORA_NODE]["inputs"]["lora_name"] = args.ingredients_lora
        api[ICLORA_NODE]["inputs"]["strength_model"] = args.ingredients_lora_strength

    # --- VAE consumers (checkpoint's vae output, slot 2) --------------------
    n_vae = rewire(api, CKPT_NODE, 2, [VAE_LOADER_ID, 0])
    api.pop(CKPT_NODE, None)

    # --- fix ResizeImageMaskNode: its DynamicCombo widget (resize_type) isn't
    # recognized by the generic UI->API converter, which shifts every widget
    # value after it by one slot. Hardcode the known-correct values (verified
    # against the node's source: nodes_post_processing.py) instead of relying
    # on the mismapped conversion.
    if RESIZE_NODE in api:
        inp = api[RESIZE_NODE]["inputs"]
        link = inp.get("input")  # the only real link-based input; preserve it
        api[RESIZE_NODE]["inputs"] = resize_inputs(link, args.width, args.height)

    # --- text encoder / audio vae / reference image / length / prompt ------
    if TEXT_ENC_NODE in api:
        api[TEXT_ENC_NODE]["inputs"]["text_encoder"] = args.text_encoder
        api[TEXT_ENC_NODE]["inputs"]["ckpt_name"] = args.connectors
    if AUDIO_VAE_NODE in api:
        api[AUDIO_VAE_NODE]["inputs"]["ckpt_name"] = args.audio_ckpt
    if LOAD_IMAGE_NODE in api:
        api[LOAD_IMAGE_NODE]["inputs"]["image"] = ref_filename
    if args.frames and VIDEO_LENGTH_NODE in api:
        api[VIDEO_LENGTH_NODE]["inputs"]["value"] = args.frames
    if args.prompt and POSITIVE_CLIP_NODE in api:
        api[POSITIVE_CLIP_NODE]["inputs"]["text"] = args.prompt
    if SAVE_NODE in api:
        api[SAVE_NODE]["inputs"]["filename_prefix"] = args.prefix

    # --- validate: no dangling refs -----------------------------------------
    dangling = []
    for nid, node in api.items():
        for k, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] not in api:
                dangling.append(f"{nid}({node['class_type']}).{k} -> {v[0]}")
    if dangling:
        print("[patch] ERRO: referencias penduradas:", dangling, file=sys.stderr)
        return 1

    json.dump(api, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[patch] model rewires={n_model} vae={n_vae}")
    print(f"[patch] gguf={args.gguf} ref_image={ref_filename} -> {args.out} ({len(api)} nodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
