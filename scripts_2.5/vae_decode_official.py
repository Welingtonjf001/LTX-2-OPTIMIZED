"""Decode the saved LTX-2.5 latent using the REAL NADiffusionDecoder from
ComfyUI core (comfy/ldm/lightricks/vae/na_diffusion_decoder.py), pulled in via
updating the vendored ComfyUI to upstream master on 2026-08-22. Supersedes the
hand-reverse-engineered scripts_2.5/na_diffusion_decoder.py, which loaded and
ran but produced visually wrong output (see MEMORIAL.md, section 3.3).
"""

import json
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

COMFY_ROOT = Path(r"E:\Users\home\Documents\LTX-2-OPTIMIZED\ComfyUI")
sys.path.insert(0, str(COMFY_ROOT))

from comfy.ldm.lightricks.vae.na_diffusion_decoder import CausalDiffusionVAE  # noqa: E402

VAE_CHECKPOINT = Path(
    r"E:\Users\home\Documents\LTX-2-OPTIMIZED\models\2.5\vae\ltx-2.5-video-vae-bf16.safetensors"
)
LATENT_PATH = Path(__file__).parent / "last_latent.pt"


def load_vae_config() -> dict:
    with open(VAE_CHECKPOINT, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    return json.loads(header["__metadata__"]["config"])["vae"]


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("Building CausalDiffusionVAE from checkpoint's own config...", flush=True)
    config = load_vae_config()
    vae = CausalDiffusionVAE(config=config)

    print("Loading state dict (checkpoint keys match directly, per the module's own docstring)...", flush=True)
    sd = load_file(str(VAE_CHECKPOINT))
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    print(f"  missing: {len(missing)}, unexpected: {len(unexpected)}")
    for k in missing[:20]:
        print("   MISSING", k)
    for k in unexpected[:20]:
        print("   UNEXPECTED", k)

    vae = vae.to(device=device, dtype=dtype).eval()
    print("  model on device", flush=True)

    saved = torch.load(LATENT_PATH, map_location="cpu", weights_only=False)
    latent_tokens = saved["latent"].to(device=device, dtype=dtype)
    ts = saved["video_tools_target_shape"]
    latent = latent_tokens.reshape(ts.batch, ts.frames, ts.height, ts.width, ts.channels).permute(0, 4, 1, 2, 3).contiguous()
    print(f"  latent grid: B={ts.batch} C={ts.channels} F={ts.frames} H={ts.height} W={ts.width}", flush=True)

    print("Decoding...", flush=True)
    with torch.no_grad():
        pixels = vae.decode(latent)
    print("  output shape:", tuple(pixels.shape), pixels.dtype)
    print("  has NaN:", bool(torch.isnan(pixels).any()), "has Inf:", bool(torch.isinf(pixels).any()))
    print("  stats: min", pixels.float().min().item(), "max", pixels.float().max().item(), "mean", pixels.float().mean().item())

    save_frames(pixels, Path(__file__).parent / "decoded_frames_official")
    print("VAE_DECODE_OFFICIAL_OK", flush=True)


def save_frames(sample: torch.Tensor, out_dir: Path) -> None:
    import numpy as np
    from PIL import Image

    out_dir.mkdir(exist_ok=True)
    px = sample[0].float().clamp(-1, 1)
    px = ((px + 1) / 2 * 255).to(torch.uint8).cpu().numpy()  # [3, F, H, W]
    for i in range(px.shape[1]):
        frame = np.transpose(px[:, i], (1, 2, 0))
        Image.fromarray(frame).save(out_dir / f"frame_{i:03d}.png")
    print(f"  saved {px.shape[1]} frames to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
