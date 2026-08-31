"""Load na_diffusion_decoder.NADiffusionDecoder from the real 2.5 checkpoint and run
a forward pass against the latent produced by generate_latent_test.py.

See na_diffusion_decoder.py's module docstring for the confidence caveat: this
architecture was reverse-engineered with no reference implementation anywhere.
This script validates load-completeness and forward-pass sanity (shape, no
NaN/Inf) -- it does NOT validate that the output pixels are visually correct.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
from na_diffusion_decoder import NADiffusionDecoder  # noqa: E402

VAE_CHECKPOINT = Path(
    r"E:\Users\home\Documents\LTX-2-OPTIMIZED\models\2.5\vae\ltx-2.5-video-vae-bf16.safetensors"
)
LATENT_PATH = Path(__file__).parent / "last_latent.pt"

# Attribute names that are bare nn.Parameter in NADiffusionDecoder but ship as
# `<name>.weight` in the checkpoint (RMSNorm-style single-vector norms).
BARE_PARAM_NAMES = {"norm1", "norm2", "norm_out", "q_norm", "k_norm"}


def load_decoder_state_dict() -> dict:
    raw = load_file(str(VAE_CHECKPOINT))
    renamed = {}
    for k, v in raw.items():
        if k.startswith("per_channel_statistics."):
            renamed[k] = v
            continue
        if not k.startswith("decoder."):
            continue
        rest = k[len("decoder."):]
        parts = rest.split(".")
        if len(parts) >= 2 and parts[-2] in BARE_PARAM_NAMES and parts[-1] == "weight":
            rest = ".".join(parts[:-1])  # strip trailing ".weight"
        renamed[rest] = v
    return renamed


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("Loading NADiffusionDecoder state dict from checkpoint...", flush=True)
    sd = load_decoder_state_dict()
    print(f"  {len(sd)} tensors after rename", flush=True)

    model = NADiffusionDecoder()
    expected = set(dict(model.named_parameters()).keys())
    provided = set(sd.keys())
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    print(f"  missing: {len(missing)}, unexpected: {len(unexpected)}")
    for k in missing[:20]:
        print("   MISSING", k)
    for k in unexpected[:20]:
        print("   UNEXPECTED", k)

    model.load_state_dict(sd, strict=False, assign=False)
    model = model.to(device=device, dtype=dtype).eval()
    print("  model loaded on device", flush=True)

    if not LATENT_PATH.exists():
        print(f"No latent found at {LATENT_PATH}; run generate_latent_test.py first.", flush=True)
        return

    saved = torch.load(LATENT_PATH, map_location="cpu", weights_only=False)
    latent_tokens = saved["latent"].to(device=device, dtype=dtype)  # [B, N, 128]
    target_shape = saved["video_tools_target_shape"]
    b, c_lat, f, h, w = target_shape.batch, target_shape.channels, target_shape.frames, target_shape.height, target_shape.width
    print(f"  target latent grid: B={b} C={c_lat} F={f} H={h} W={w}", flush=True)
    latent = latent_tokens.reshape(b, f, h, w, c_lat).permute(0, 4, 1, 2, 3).contiguous()

    print("Running decoder forward pass...", flush=True)
    with torch.no_grad():
        out = model(latent)
    print("  output shape:", tuple(out.sample.shape), out.sample.dtype)
    print("  has NaN:", bool(torch.isnan(out.sample).any()), "has Inf:", bool(torch.isinf(out.sample).any()))
    print("  output stats: min", out.sample.float().min().item(), "max", out.sample.float().max().item(), "mean", out.sample.float().mean().item())

    save_frames(out.sample, Path(__file__).parent / "decoded_frames")
    print("VAE_DECODE_TEST_DONE", flush=True)


def save_frames(sample: torch.Tensor, out_dir: Path) -> None:
    """sample: [B, 3, F, H, W], assumed range [-1, 1]. Saves each frame as a PNG for visual inspection."""
    import numpy as np
    from PIL import Image

    out_dir.mkdir(exist_ok=True)
    px = sample[0].float().clamp(-1, 1)
    px = ((px + 1) / 2 * 255).to(torch.uint8).cpu().numpy()  # [3, F, H, W]
    for i in range(px.shape[1]):
        frame = np.transpose(px[:, i], (1, 2, 0))  # HWC
        Image.fromarray(frame).save(out_dir / f"frame_{i:03d}.png")
    print(f"  saved {px.shape[1]} frames to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
