"""Best-effort reimplementation of LTX-2.5's video VAE decoder ("NADiffusionDecoder").

WARNING -- confidence level is much lower than every other module in scripts_2.5/.
Unlike the Gemma4 encoder (found ready-made in ComfyUI) and the DiT embeddings
connector (found ready-made in ltx_core), NO reference implementation of this
architecture exists anywhere (grepped this repo including ComfyUI/ -- nothing).
This was reverse-engineered purely from tensor names/shapes in
models/2.5/vae/ltx-2.5-video-vae-bf16.safetensors and the config embedded in
its own metadata. See MEMORIAL.md, section "LTX-2.5", for the full reasoning
and confidence levels per piece. Verified mechanically (state-dict completeness,
output shape, no NaN/Inf) -- NOT verified visually. Treat pixel output as
unverified until someone looks at an actual decoded frame.

Architecture (config key -> meaning, from the checkpoint's own metadata):
  in_channels=128, out_channels=3, patch_size=4, head_dim=64
  stage_channels=[2048, 1024, 512, 512, 256], stage_depths=[4, 6, 4, 2, 8]
    -> det_stages 0..3 use stage_channels[0:4] with stage_depths[0:4] plain
       (non-AdaLN) pre-norm transformer blocks each; the 5th "stage" (8 blocks,
       stage_channels[4]=256) is diff_blocks, AdaLN-modulated, cross-attending
       to the det_stages output as "context" and refining a patchified pixel
       estimate x_t.
  upsamples: 4 Linear-based "depth-to-space" upsamplers between det_stages
    (and a final one producing the diff_blocks context), resampler_kind="linear".
    Shapes verified by construction against the checkpoint (out_features =
    target_stage_channels * t_mult*h_mult*w_mult, confirmed for all 4).
  model_output_type="x0", default_num_inference_steps=1 -> single-step,
    direct clean-image prediction (matches the X0Model pattern used
    everywhere else in this codebase).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ltx_core.model.transformer.timestep_embedding import Timesteps
from ltx_core.model.video_vae.ops import PerChannelStatistics
from ltx_core.utils import rms_norm

HEAD_DIM = 64
STAGE_CHANNELS = [2048, 1024, 512, 512, 256]
STAGE_DEPTHS = [4, 6, 4, 2]  # det_stages; diff_blocks depth (8) handled separately
DIFF_DEPTH = 8
DIFF_DIM = 256
UPSAMPLE_FACTORS = [(1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)]  # (t, h, w) per stage transition
IN_CHANNELS = 128
OUT_CHANNELS = 3
PATCH_SIZE = 4


def _qk_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # x: [..., heads, head_dim], weight: [head_dim]
    return rms_norm(x) * weight


class _FusedQKVAttention(nn.Module):
    """Plain (non-RoPE) multi-head self-attention with fused QKV and per-head QK RMSNorm."""

    def __init__(self, dim: int, head_dim: int = HEAD_DIM):
        super().__init__()
        self.dim = dim
        self.heads = dim // head_dim
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.k_norm = nn.Parameter(torch.ones(head_dim))
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = _qk_rms_norm(q, self.q_norm).transpose(1, 2)
        k = _qk_rms_norm(k, self.k_norm).transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, self.dim)
        return self.proj(out)


class _SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = dim * mult
        self.w_gate = nn.Linear(dim, inner, bias=False)
        self.w_up = nn.Linear(dim, inner, bias=False)
        self.w_down = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class _DetStageBlock(nn.Module):
    """Plain pre-norm ViT block, no AdaLN (matches: no scale_shift_table in det_stages tensors)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.Parameter(torch.ones(dim))
        self.attn = _FusedQKVAttention(dim)
        self.norm2 = nn.Parameter(torch.ones(dim))
        self.mlp = _SwiGLUFeedForward(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rms_norm(x) * self.norm1)
        x = x + self.mlp(rms_norm(x) * self.norm2)
        return x


class _DiffBlock(nn.Module):
    """AdaLN-modulated block with a single JOINT attention (x tokens + projected context
    tokens concatenated, one shared fused-QKV) -- checkpoint has exactly one `attn.*` group
    and one `context_proj` per block, ruling out a separate self+cross-attention design.
    7-slot scale_shift_table layout (best-effort guess): [0:3]=shift/scale/gate for the joint
    attention residual, [3]=multiplicative gate applied to the context tokens before they're
    concatenated in, [4:7]=shift/scale/gate for the MLP residual.
    """

    def __init__(self, dim: int, context_dim: int):
        super().__init__()
        self.norm1 = nn.Parameter(torch.ones(dim))
        self.attn = _FusedQKVAttention(dim)
        self.context_proj = nn.Linear(context_dim, dim, bias=True)
        self.norm2 = nn.Parameter(torch.ones(dim))
        self.mlp = _SwiGLUFeedForward(dim)
        self.scale_shift_table = nn.Parameter(torch.empty(7, dim))

    def forward(self, x: torch.Tensor, context: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        # mod: [B, 7, dim] dynamic (per-sample) modulation from shared_adaln; combined with the
        # per-block learned base (scale_shift_table), matching get_ada_values() elsewhere in this repo.
        values = self.scale_shift_table[None] + mod
        shift1, scale1, gate1 = values[:, 0], values[:, 1], values[:, 2]
        gate_ctx = values[:, 3]
        shift2, scale2, gate2 = values[:, 4], values[:, 5], values[:, 6]

        n_x = x.shape[1]
        normed = rms_norm(x) * self.norm1
        normed = normed * (1 + scale1[:, None]) + shift1[:, None]
        ctx = self.context_proj(context) * gate_ctx[:, None]
        joint = torch.cat([normed, ctx], dim=1)
        attn_out = self.attn(joint)[:, :n_x]
        x = x + attn_out * gate1[:, None]

        normed = rms_norm(x) * self.norm2
        normed = normed * (1 + scale2[:, None]) + shift2[:, None]
        x = x + self.mlp(normed) * gate2[:, None]
        return x


class _Upsample(nn.Module):
    """Linear "depth-to-space" upsampler: dim -> dim_out * (t*h*w), then rearrange the extra
    factor into the spatiotemporal grid. resampler_kind="linear" per the checkpoint's own config.
    """

    def __init__(self, dim: int, dim_out: int, factors: tuple[int, int, int]):
        super().__init__()
        self.dim_out = dim_out
        self.factors = factors
        t, h, w = factors
        self.proj = nn.Linear(dim, dim_out * t * h * w, bias=True)

    def forward(self, x: torch.Tensor, grid: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
        # x: [B, N, C] with N = f*h*w tokens over `grid`
        b = x.shape[0]
        f, h, w = grid
        t_m, h_m, w_m = self.factors
        x = self.proj(x)  # [B, N, dim_out*t_m*h_m*w_m]
        x = x.view(b, f, h, w, t_m, h_m, w_m, self.dim_out)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7)  # [B, f, t_m, h, h_m, w, w_m, C]
        new_grid = (f * t_m, h * h_m, w * w_m)
        x = x.reshape(b, new_grid[0], new_grid[1], new_grid[2], self.dim_out)
        x = x.reshape(b, math.prod(new_grid), self.dim_out)
        return x, new_grid


class _TEmbedder(nn.Module):
    """Matches checkpoint keys t_embedder.mlp.0.*/t_embedder.mlp.2.* (Linear, SiLU, Linear)."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _SharedAdaLN(nn.Module):
    """Matches checkpoint keys shared_adaln.proj.*."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@dataclass
class DecoderOutput:
    sample: torch.Tensor  # [B, 3, F, H, W] pixel output
    grid: tuple[int, int, int]


class NADiffusionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.per_channel_statistics = PerChannelStatistics(latent_channels=IN_CHANNELS)
        self.type_emb = nn.Parameter(torch.zeros(IN_CHANNELS))
        self.conv_in = nn.Linear(IN_CHANNELS, STAGE_CHANNELS[0], bias=True)

        self.det_stages = nn.ModuleList(
            [
                nn.ModuleList([_DetStageBlock(STAGE_CHANNELS[i]) for _ in range(STAGE_DEPTHS[i])])
                for i in range(len(STAGE_DEPTHS))
            ]
        )
        self.upsamples = nn.ModuleList(
            [
                _Upsample(STAGE_CHANNELS[i], STAGE_CHANNELS[i + 1], UPSAMPLE_FACTORS[i])
                for i in range(len(UPSAMPLE_FACTORS))
            ]
        )

        self.conv_in_x_t = nn.Linear(OUT_CHANNELS * PATCH_SIZE * PATCH_SIZE, DIFF_DIM, bias=True)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.t_embedder = _TEmbedder(256, 384)
        self.shared_adaln = _SharedAdaLN(384, 7 * DIFF_DIM)
        self.diff_blocks = nn.ModuleList([_DiffBlock(DIFF_DIM, DIFF_DIM) for _ in range(DIFF_DEPTH)])
        self.norm_out = nn.Parameter(torch.ones(DIFF_DIM))
        self.conv_out = nn.Linear(DIFF_DIM, OUT_CHANNELS * PATCH_SIZE * PATCH_SIZE, bias=True)

        self.timestep_scale_multiplier = 1000.0
        self.default_num_inference_steps = 1

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor | None = None) -> DecoderOutput:
        """latent: [B, C=128, F, H, W] (raw VAE latent grid, still per-channel normalized)."""
        latent = self.per_channel_statistics.un_normalize(latent)
        b, c, f, h, w = latent.shape
        x = latent.permute(0, 2, 3, 4, 1).reshape(b, f * h * w, c)  # [B, N, 128]
        x = x + self.type_emb[None, None]
        x = self.conv_in(x)

        grid = (f, h, w)
        for stage_idx, blocks in enumerate(self.det_stages):
            for block in blocks:
                x = block(x)
            if stage_idx < len(self.upsamples):
                x, grid = self.upsamples[stage_idx](x, grid)

        context = x  # [B, N_out, 256], N_out = prod(grid) at full output resolution

        f2, h2, w2 = grid
        n_out = f2 * h2 * w2
        if timestep is None:
            timestep = torch.full((b,), 0.05, device=latent.device, dtype=torch.float32)
        timesteps_proj = self.time_proj(timestep).to(dtype=latent.dtype)  # [B, 256]

        x_t_patch = torch.zeros(b, n_out, OUT_CHANNELS * PATCH_SIZE * PATCH_SIZE, device=latent.device, dtype=latent.dtype)
        x_t_emb = self.conv_in_x_t(x_t_patch)  # [B, N_out, 256]

        # shared_adaln modulation is per-sample (one timestep per sample), broadcast across all
        # diff_block tokens -- matches the module name "shared" and its Linear(384, 7*256) shape
        # (no token dimension baked in).
        t_hidden = self.t_embedder(timesteps_proj)  # [B, 384]
        dyn_mod = self.shared_adaln(t_hidden).view(b, 1, 7, DIFF_DIM)  # broadcast over tokens

        h_state = x_t_emb
        for block in self.diff_blocks:
            h_state = block(h_state, context, dyn_mod[:, 0])

        h_state = rms_norm(h_state) * self.norm_out
        out = self.conv_out(h_state)  # [B, N_out, 48]

        out = out.view(b, f2, h2, w2, OUT_CHANNELS, PATCH_SIZE, PATCH_SIZE)
        out = out.permute(0, 4, 1, 2, 5, 3, 6)  # [B, 3, f2, H, ph, W, pw]
        out = out.reshape(b, OUT_CHANNELS, f2, h2 * PATCH_SIZE, w2 * PATCH_SIZE)
        return DecoderOutput(sample=out, grid=grid)
