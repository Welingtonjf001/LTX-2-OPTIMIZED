"""Multi-step LTX-2.5 T2V latent generation (no VAE decode yet -- see CLAUDE.md /
MEMORIAL.md, section "LTX-2.5": the 2.5 video VAE decoder is a new,
undocumented architecture (NADiffusionDecoder) and hasn't been implemented).

This runs the REAL multi-step denoising loop (LinearQuadraticScheduler +
EulerDiffusionStep + euler_denoising_loop, all reused unmodified from
ltx_core/ltx_pipelines) against the real LTX-2.5 transformer, producing a
final denoised video latent. Purpose: validate that multi-step sampling itself
works correctly, independent of the VAE-decode risk.

Step count / threshold_noise are a best guess (no official 2.5-distilled sigma
schedule is available anywhere -- see MEMORIAL.md).
"""

import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from dit_bridge import load_embeddings_processor, load_transformer  # noqa: E402
from gemma4_text_encoder import Gemma25TextEncoder  # noqa: E402

from ltx_core.components.diffusion_steps import EulerDiffusionStep  # noqa: E402
from ltx_core.components.noisers import GaussianNoiser  # noqa: E402
from ltx_core.components.schedulers import LinearQuadraticScheduler  # noqa: E402
from ltx_core.guidance.perturbations import BatchedPerturbationConfig  # noqa: E402
from ltx_core.types import LatentState, VideoPixelShape  # noqa: E402
from ltx_pipelines.utils.helpers import modality_from_latent_state, noise_video_state  # noqa: E402
from ltx_pipelines.utils.samplers import euler_denoising_loop  # noqa: E402
from ltx_pipelines.utils.types import PipelineComponents  # noqa: E402

NUM_STEPS = 8  # best guess, matches the "8-step distilled" pattern documented for 2.3/2.5


def encode_prompt(prompt: str, device: torch.device, dtype: torch.dtype):
    encoder = Gemma25TextEncoder()
    cond = encoder.encode(prompt)
    emb = cond[0][0] if isinstance(cond[0], (list, tuple)) else cond[0]  # [B, L, T, D]
    del encoder
    torch.cuda.empty_cache()

    hidden_states = emb.permute(0, 2, 3, 1).to(device=device, dtype=dtype)  # [B, T, D, L]
    seq_len = hidden_states.shape[1]
    attention_mask = torch.ones(1, seq_len, device=device, dtype=torch.float32)

    num_registers = 128
    pad_to = ((seq_len + num_registers - 1) // num_registers) * num_registers
    if pad_to > seq_len:
        pad_len = pad_to - seq_len
        pad_hs = torch.zeros(hidden_states.shape[0], pad_len, *hidden_states.shape[2:], device=device, dtype=hidden_states.dtype)
        hidden_states = torch.cat([pad_hs, hidden_states], dim=1)
        pad_mask = torch.zeros(1, pad_len, device=device, dtype=torch.float32)
        attention_mask = torch.cat([pad_mask, attention_mask], dim=1)

    embeddings_processor = load_embeddings_processor(device=device, dtype=dtype)
    out = embeddings_processor.process_hidden_states(hidden_states, attention_mask, padding_side="left")
    del embeddings_processor
    torch.cuda.empty_cache()
    return out.video_encoding.to(device=device, dtype=dtype)


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    prompt = "A cinematic wide shot of a red fox running through fresh snow at sunset, warm rim light."
    print("Encoding prompt...", flush=True)
    context = encode_prompt(prompt, device, dtype)
    print("  context:", tuple(context.shape), flush=True)

    print("Loading LTX-2.5 transformer (offloaded)...", flush=True)
    t0 = time.time()
    x0_model = load_transformer(device=device, dtype=dtype, max_memory={0: "5GiB", "cpu": "44GiB"})
    print(f"  ready in {time.time() - t0:.1f}s", flush=True)

    batch_size = 1
    components = PipelineComponents(dtype=dtype, device=device)
    output_shape = VideoPixelShape(batch=batch_size, frames=9, height=320, width=512, fps=24.0)
    noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(0))
    video_state, video_tools = noise_video_state(
        output_shape=output_shape,
        noiser=noiser,
        conditionings=[],
        components=components,
        dtype=dtype,
        device=device,
    )
    print("  latent tokens:", video_state.latent.shape[1], flush=True)

    sigmas = LinearQuadraticScheduler().execute(steps=NUM_STEPS).to(device=device, dtype=torch.float32)
    print("  sigmas:", sigmas.tolist(), flush=True)
    stepper = EulerDiffusionStep()
    perturbations = BatchedPerturbationConfig.empty(batch_size)

    def denoise_fn(v_state: LatentState, _a_state, sigs: torch.Tensor, step_index: int):
        sigma = sigs[step_index].reshape(1)
        modality = modality_from_latent_state(state=v_state, context=context, sigma=sigma)
        with torch.no_grad():
            denoised_video, _denoised_audio = x0_model(video=modality, audio=None, perturbations=perturbations)
        return denoised_video, None

    print(f"Running {NUM_STEPS}-step denoising loop...", flush=True)
    t0 = time.time()
    final_video_state, _final_audio_state = euler_denoising_loop(
        sigmas=sigmas,
        video_state=video_state,
        audio_state=replace(video_state),  # unused (disable_audio=True), just needs to be a LatentState-shaped value
        stepper=stepper,
        denoise_fn=denoise_fn,
        disable_audio=True,
    )
    print(f"  denoising loop done in {time.time() - t0:.1f}s", flush=True)

    latent = final_video_state.latent
    print("  final latent shape:", tuple(latent.shape), latent.dtype)
    print("  has NaN:", bool(torch.isnan(latent).any()), "has Inf:", bool(torch.isinf(latent).any()))
    print("  latent stats: mean", latent.float().mean().item(), "std", latent.float().std().item())

    out_path = Path(__file__).parent / "last_latent.pt"
    torch.save(
        {"latent": latent.cpu(), "positions": final_video_state.positions.cpu(), "video_tools_target_shape": video_tools.target_shape},
        out_path,
    )
    print(f"  saved latent to {out_path}", flush=True)
    print("LATENT_GENERATION_OK", flush=True)


if __name__ == "__main__":
    main()
