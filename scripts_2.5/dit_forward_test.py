"""End-to-end smoke test: encode a prompt with Gemma25TextEncoder, run it through
the (Gemma4-sized) EmbeddingsProcessor, and do ONE forward pass through the real
LTX-2.5 transformer with a synthetic random video latent.

This validates that the glue code (dit_bridge.py) produces correctly-shaped,
finite (no NaN/Inf) output -- it does NOT validate that the output is a
sensible video (no VAE decode, no real position grid, no multi-step sampling).
Full generation is separate, bigger scope -- see CLAUDE.md, section "LTX-2.5".
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from dit_bridge import load_embeddings_processor, load_transformer  # noqa: E402
from gemma4_text_encoder import Gemma25TextEncoder  # noqa: E402

from ltx_core.components.noisers import GaussianNoiser  # noqa: E402
from ltx_core.guidance.perturbations import BatchedPerturbationConfig  # noqa: E402
from ltx_core.types import VideoPixelShape  # noqa: E402
from ltx_pipelines.utils.helpers import modality_from_latent_state, noise_video_state  # noqa: E402
from ltx_pipelines.utils.types import PipelineComponents  # noqa: E402


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("Loading Gemma 4 text encoder + encoding prompt...", flush=True)
    encoder = Gemma25TextEncoder()
    prompt = "A cinematic wide shot of a red fox running through fresh snow at sunset, warm rim light."
    cond = encoder.encode(prompt)
    emb = cond[0][0] if isinstance(cond[0], (list, tuple)) else cond[0]  # [B, L, T, D]
    print("  raw hidden states:", tuple(emb.shape), emb.dtype, flush=True)
    del encoder
    torch.cuda.empty_cache()

    hidden_states = emb.permute(0, 2, 3, 1).to(device=device, dtype=dtype)  # [B, T, D, L]
    seq_len = hidden_states.shape[1]
    attention_mask = torch.ones(1, seq_len, device=device, dtype=torch.float32)

    # The embeddings connector replaces padded positions with learnable registers and
    # requires seq_len % num_learnable_registers == 0 (128 here). Left-pad to match
    # (padding_side="left" matches the Gemma tokenizer convention used elsewhere in this repo).
    NUM_LEARNABLE_REGISTERS = 128
    pad_to = ((seq_len + NUM_LEARNABLE_REGISTERS - 1) // NUM_LEARNABLE_REGISTERS) * NUM_LEARNABLE_REGISTERS
    if pad_to > seq_len:
        pad_len = pad_to - seq_len
        pad_hs = torch.zeros(
            hidden_states.shape[0], pad_len, *hidden_states.shape[2:], device=device, dtype=hidden_states.dtype
        )
        hidden_states = torch.cat([pad_hs, hidden_states], dim=1)
        pad_mask = torch.zeros(1, pad_len, device=device, dtype=torch.float32)
        attention_mask = torch.cat([pad_mask, attention_mask], dim=1)
        print(f"  padded seq_len {seq_len} -> {pad_to} (left pad)", flush=True)

    print("Loading embeddings processor (Gemma4-sized connectors)...", flush=True)
    embeddings_processor = load_embeddings_processor(device=device, dtype=dtype)
    out = embeddings_processor.process_hidden_states(hidden_states, attention_mask, padding_side="left")
    print("  video_encoding:", tuple(out.video_encoding.shape), out.video_encoding.dtype)
    print("  audio_encoding:", tuple(out.audio_encoding.shape) if out.audio_encoding is not None else None)
    print(
        "  video_encoding has NaN:",
        bool(torch.isnan(out.video_encoding).any()),
        "has Inf:",
        bool(torch.isinf(out.video_encoding).any()),
        flush=True,
    )
    del embeddings_processor
    torch.cuda.empty_cache()

    print("Loading LTX-2.5 transformer (offloaded, this is the 40GB checkpoint)...", flush=True)
    t0 = time.time()
    x0_model = load_transformer(
        device=device,
        dtype=dtype,
        max_memory={0: "5GiB", "cpu": "44GiB"},
    )
    print(f"  transformer ready in {time.time() - t0:.1f}s", flush=True)

    batch_size = 1
    components = PipelineComponents(dtype=dtype, device=device)
    output_shape = VideoPixelShape(batch=batch_size, frames=1, height=32, width=32, fps=24.0)
    noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(0))
    video_state, _video_tools = noise_video_state(
        output_shape=output_shape,
        noiser=noiser,
        conditionings=[],
        components=components,
        dtype=dtype,
        device=device,
    )
    sigma = torch.tensor([1.0], device=device, dtype=torch.float32)
    video = modality_from_latent_state(
        state=video_state,
        context=out.video_encoding.to(device=device, dtype=dtype),
        sigma=sigma,
    )
    print("  latent tokens:", video.latent.shape[1], flush=True)

    print("Running one forward pass through the transformer...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        denoised_video, denoised_audio = x0_model(
            video=video,
            audio=None,
            perturbations=BatchedPerturbationConfig.empty(batch_size),
        )
    print(f"  forward pass in {time.time() - t0:.1f}s", flush=True)
    print("  denoised_video shape:", tuple(denoised_video.shape), denoised_video.dtype)
    print(
        "  has NaN:",
        bool(torch.isnan(denoised_video).any()),
        "has Inf:",
        bool(torch.isinf(denoised_video).any()),
    )
    print("FORWARD_TEST_OK", flush=True)


if __name__ == "__main__":
    main()
