"""Encode a prompt with the NATIVE LTX-2.3 Gemma text encoder (ltx_pipelines'
ModelLedger -- the same code path proven to work, ~146s cold) and write the
result in the safetensors format `ComfyUI-LTXVideo`'s LTXVSaveConditioning/
LTXVLoadConditioning nodes use, so ComfyUI's video-sampling graph can consume
it directly without ever loading Gemma3 through ComfyUI's own model
management.

Why this exists: ComfyUI's own Gemma3 loading for this custom node is broken
on this GPU -- investigated in depth (see memory/project_lora_storyboard_test_ui.md):
`comfy.memory_management.aimdo_enabled` forces the CLIP to always initialize
on CPU regardless of `--disable-dynamic-vram`, and even the `--highvram`
workaround that bypasses that just barely fits (24.1 of 24.5GB) and hangs
under load. Bypassing ComfyUI for JUST this one step and reusing the already-
working native encoder sidesteps the bug entirely.

Run as a SEPARATE subprocess (not imported into the ComfyUI process or a long-
lived UI process) -- matches this project's established convention of
isolating GPU-heavy stages in their own process so CUDA memory is actually
returned to the OS on exit (see CLAUDE.md, "Liberação de memória").

Structural note, RESOLVED by reading comfy/ldm/lightricks/av_model.py's
`preprocess_text_embeds` directly: `ltx-2.3-22b-distilled-fp8.safetensors` is
an audio+video JOINT model (LTXAV). Its sampler expects ONE context tensor
whose LAST dim is `cross_attention_dim + audio_cross_attention_dim` (4096 +
2048 = 6144) -- video and audio embeddings concatenated channel-wise, not two
separate tensors. Passing only the 4096-wide video_encoding hit
`self.audio_connector(context)` with a video-shaped tensor and crashed with
"Expected size 4096 but got size 2048" inside `embeddings_connector.py`
(reproduced live). `ltx_core`'s `EmbeddingsProcessorOutput.audio_encoding` is
populated automatically by the SAME `encode_prompts()` call (no separate
request needed) whenever the checkpoint's embeddings processor has an audio
connector configured -- this script was just discarding it. Concatenating
video_encoding+audio_encoding before saving lands on exactly 6144, which
`preprocess_text_embeds` recognizes as "already fully processed" and returns
untouched (see its first `if`) -- the fast path, no connector call needed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "packages" / "ltx-pipelines" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "ltx-core" / "src"))

from ltx_pipelines.utils.helpers import encode_prompts  # noqa: E402
from ltx_pipelines.utils.model_ledger import ModelLedger  # noqa: E402
from ltx_core.quantization import QuantizationPolicy  # noqa: E402


def _save_conditioning(
    video_encoding: torch.Tensor,
    audio_encoding: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    out_path: Path,
) -> None:
    """Mirrors ComfyUI-LTXVideo/conditioning_saver.py's LTXVSaveConditioning
    key names (conditioning_data_0[, attention_mask_0]) and bfloat16 storage
    dtype, so LTXVLoadConditioning reads it back unmodified -- but the tensor
    itself is video_encoding concatenated with audio_encoding on the last
    dim (see module docstring: this checkpoint is an audio+video joint
    model and its sampler expects that single concatenated context)."""
    context = video_encoding
    if audio_encoding is not None:
        context = torch.cat([video_encoding, audio_encoding], dim=-1)
    tensors = {"conditioning_data_0": context.to(dtype=torch.bfloat16).contiguous().cpu()}
    if attention_mask is not None:
        tensors["attention_mask_0"] = attention_mask.contiguous().cpu()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gemma-root", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--spatial-upsampler-path", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Mirrors MusicToVideoPipeline.__init__ exactly (music_to_video.py):
    # quantization=fp8_cast(), a real spatial_upsampler_path, AND -- this is
    # the part that actually mattered -- ONE prompt per encode_prompts() call
    # (production does `(context_p,) = encode_prompts([prompt], ...)`, never
    # two at once). This script originally encoded [prompt, negative] together
    # in one call and Gemma3's forward pass threw `torch.OutOfMemoryError`
    # reporting a physically impossible "60.73 GiB allocated" on a 24GB card
    # -- reproduced identically 4x, including with quantization/upsampler
    # matched to production. Switching to one encode_prompts([prompt]) call
    # per subprocess invocation (this script now does exactly that; the
    # backend runs it twice, once per prompt) fixed it. Root cause not fully
    # understood -- looks like a leak in accelerate's CPU-offload hooks that
    # only shows up encoding a second prompt in the same process without a
    # `cleanup_memory()` in between -- but matching production's one-encode-
    # per-process pattern avoids it empirically.
    model_ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=args.checkpoint,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root_path=args.gemma_root,
        loras=[],
        quantization=QuantizationPolicy.fp8_cast(),
    )

    (out,) = encode_prompts([args.prompt], model_ledger)
    print(f"video_encoding {tuple(out.video_encoding.shape)}, "
          f"audio_encoding {tuple(out.audio_encoding.shape) if out.audio_encoding is not None else None}",
          file=sys.stderr)
    _save_conditioning(out.video_encoding, out.audio_encoding, out.attention_mask, args.out)
    print(f"OK: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
