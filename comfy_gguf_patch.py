"""Patch the API-format LTX-2.3 'Full' workflow to run from a GGUF UNET using
only the model files present in this installation.

The upstream example workflow assumes a single ``ltx-2.3-22b-dev.safetensors``
checkpoint (model+clip+vae), a distilled LoRA, and the RES4LYF ClownSampler.
None of those are installed here. What we do have:

  UNET  -> models/*.gguf                                     (UnetLoaderGGUF)
  VAE   -> models/vae/ltx-2.3-22b-dev_video_vae.safetensors  (VAELoader)
  audio -> models/ltx-2.3-22b-distilled-fp8.safetensors      (LTXVAudioVAELoader)
  text  -> models/gemma3 shards + dev embeddings connectors  (LTXAVTextEncoderLoader)

So we rewire: CheckpointLoaderSimple is dropped (its MODEL comes from the GGUF
loader, its VAE from VAELoader), the LoRA node is bypassed (no LoRA available,
so the dev model needs a real step count instead of the distilled few-step
schedule), and ClownSampler_Beta is replaced by stock KSamplerSelect(euler).
"""
import argparse
import json
import sys

GGUF_LOADER_ID = "90001"
VAE_LOADER_ID = "90002"
SAMPLER_ID = "90003"

CKPT_NODE = "3940"          # CheckpointLoaderSimple (absent model)
LORA_NODE = "4968"          # LoraLoaderModelOnly (no LoRA installed)
CLOWN_NODE = "4967"         # ClownSampler_Beta (custom pack not installed)
AUDIO_VAE_NODE = "4010"     # LTXVAudioVAELoader
TEXT_ENC_NODE = "4960"      # LTXAVTextEncoderLoader
SCHEDULER_NODE = "4966"     # LTXVScheduler
LATENT_NODE = "3059"        # EmptyLTXVLatentVideo
FRAMES_NODE = "4979"        # PrimitiveInt feeding length
SAVE_NODE = "4823"          # SaveVideo
IMG_COND_NODE = "3159"      # LTXVImgToVideoConditionOnly
PREPROCESS_NODE = "3336"    # LTXVPreprocess
RESIZE_NODE = "4981"        # ResizeImageMaskNode
LOADIMAGE_NODE = "2004"     # LoadImage


def prune_orphans(api: dict, keep_id: str) -> dict:
    """Keep only what the output node still transitively depends on."""
    keep: set[str] = set()
    stack = [str(keep_id)]
    while stack:
        nid = stack.pop()
        if nid in keep or nid not in api:
            continue
        keep.add(nid)
        for v in api[nid]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                stack.append(v[0])
    return {k: v for k, v in api.items() if k in keep}


def rewire(api: dict, old_id: str, old_slot: int | None, new_ref: list) -> int:
    """Point every consumer of (old_id, old_slot) at new_ref. slot None = any."""
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
    ap.add_argument("--gguf", default="ltx-2.3-22b-dev-Q6_K.gguf")
    ap.add_argument("--vae", default="vae\\ltx-2.3-22b-dev_video_vae.safetensors")
    ap.add_argument("--audio-ckpt", default="ltx-2.3-22b-distilled-fp8.safetensors")
    ap.add_argument("--text-encoder", default="model-00001-of-00005.safetensors")
    ap.add_argument("--connectors", default="text_encoders\\ltx-2.3-22b-dev_embeddings_connectors.safetensors")
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--lora", default="", help="distilled LoRA filename; enables few-step generation. Empty = no LoRA.")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=30, help="dev model without the distilled LoRA needs real steps")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--prefix", default="gguf_test")
    ap.add_argument("--t2v", action="store_true",
                    help="drop the image-conditioning branch and feed the empty latent straight "
                         "into the AV concat (pure text-to-video)")
    args = ap.parse_args()

    api = json.load(open(args.inp, encoding="utf-8"))

    # --- new source nodes -------------------------------------------------
    api[GGUF_LOADER_ID] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": args.gguf}}
    api[VAE_LOADER_ID] = {"class_type": "VAELoader", "inputs": {"vae_name": args.vae}}
    api[SAMPLER_ID] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": args.sampler}}

    # --- rewire model path -----------------------------------------------
    if args.lora:
        # GGUF -> LoRA(distilled) -> model consumers. Keep the LoRA node, feed it
        # from the GGUF UNET, and point its file at the distilled LoRA so the dev
        # model runs in the few-step distilled regime.
        if LORA_NODE in api:
            api[LORA_NODE]["inputs"]["model"] = [GGUF_LOADER_ID, 0]
            api[LORA_NODE]["inputs"]["lora_name"] = args.lora
            api[LORA_NODE]["inputs"]["strength_model"] = args.lora_strength
            n_model = 1
        else:
            n_model = 0
        # Anything that read the raw checkpoint model now reads the LoRA output.
        n_model += rewire(api, CKPT_NODE, 0, [LORA_NODE, 0])
    else:
        # No LoRA available: consumers read the GGUF model directly.
        n_model = rewire(api, LORA_NODE, None, [GGUF_LOADER_ID, 0])
        n_model += rewire(api, CKPT_NODE, 0, [GGUF_LOADER_ID, 0])
        api.pop(LORA_NODE, None)

    # --- rewire VAE consumers --------------------------------------------
    n_vae = rewire(api, CKPT_NODE, 2, [VAE_LOADER_ID, 0])
    # --- replace the missing sampler node --------------------------------
    n_samp = rewire(api, CLOWN_NODE, None, [SAMPLER_ID, 0])

    api.pop(CKPT_NODE, None)
    api.pop(CLOWN_NODE, None)

    # --- point the remaining loaders at files we actually have ------------
    if AUDIO_VAE_NODE in api:
        api[AUDIO_VAE_NODE]["inputs"]["ckpt_name"] = args.audio_ckpt
    if TEXT_ENC_NODE in api:
        api[TEXT_ENC_NODE]["inputs"]["text_encoder"] = args.text_encoder
        api[TEXT_ENC_NODE]["inputs"]["ckpt_name"] = args.connectors

    # --- optional: strip the image (i2v) branch for pure text-to-video ----
    if args.t2v:
        # LoadImage -> ResizeImageMaskNode -> LTXVPreprocess -> ImgToVideoConditionOnly
        # feeds LTXVConcatAVLatent.video_latent; send the empty latent instead.
        n_img = rewire(api, IMG_COND_NODE, 0, [LATENT_NODE, 0])
        for nid in (IMG_COND_NODE, PREPROCESS_NODE, RESIZE_NODE, LOADIMAGE_NODE):
            api.pop(nid, None)
        # Drop anything now orphaned that only existed to feed the image branch.
        api = prune_orphans(api, SAVE_NODE)
        print(f"[patch] t2v: image branch removed ({n_img} rewire), {len(api)} nodes left")

    # --- generation parameters -------------------------------------------
    if SCHEDULER_NODE in api:
        api[SCHEDULER_NODE]["inputs"]["steps"] = args.steps
    if args.width and LATENT_NODE in api:
        api[LATENT_NODE]["inputs"]["width"] = args.width
    if args.height and LATENT_NODE in api:
        api[LATENT_NODE]["inputs"]["height"] = args.height
    if args.frames and FRAMES_NODE in api:
        api[FRAMES_NODE]["inputs"]["value"] = args.frames
    if SAVE_NODE in api:
        api[SAVE_NODE]["inputs"]["filename_prefix"] = args.prefix

    # --- validate: no dangling refs --------------------------------------
    dangling = []
    for nid, node in api.items():
        for k, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] not in api:
                dangling.append(f"{nid}({node['class_type']}).{k} -> {v[0]}")
    if dangling:
        print("[patch] ERRO: referencias penduradas:", dangling, file=sys.stderr)
        return 1

    json.dump(api, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[patch] model rewires={n_model} vae={n_vae} sampler={n_samp}")
    print(f"[patch] gguf={args.gguf} steps={args.steps} -> {args.out} ({len(api)} nodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
