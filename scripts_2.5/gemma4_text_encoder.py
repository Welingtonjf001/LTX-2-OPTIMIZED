"""Standalone LTX-2.5 text encoder (Gemma 4 12B, "gemma4_unified").

LTX-2.5's transformer (DiT) is not wired into ltx_core yet -- see CLAUDE.md,
section "LTX-2.5". This module only covers the text encoder half, which is
validated and works standalone. It borrows the encoder implementation from
the vendored ComfyUI (`ComfyUI/comfy/text_encoders/gemma4.py`), since that is
the only place a working "gemma4_unified" implementation exists anywhere
(nothing in ltx_core or the official Lightricks repo yet). ComfyUI's own
architecture-detection (`comfy.sd.load_text_encoder_state_dicts`) is reused
rather than hand-instantiating `Gemma4_12B`, so the correct config comes from
the checkpoint itself instead of being duplicated here.

Requires the process to run with CUDA_VISIBLE_DEVICES=1 (RTX 3090) per this
machine's convention -- see CLAUDE.md.
"""

import os
import sys
from pathlib import Path

COMFY_ROOT = Path(r"E:\Users\home\Documents\LTX-2-OPTIMIZED\ComfyUI")
DEFAULT_CHECKPOINT = Path(
    r"E:\Users\home\Documents\LTX-2-OPTIMIZED\models\2.5\text_encoders\gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
)


def _ensure_comfy_on_path() -> None:
    p = str(COMFY_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


class Gemma25TextEncoder:
    """Loads the LTX-2.5 Gemma4 text encoder checkpoint and encodes prompts.

    Returned embeddings have shape [batch, num_hidden_layers+1, seq_len, 3840]
    (all intermediate layer outputs, matching how the 2.3 EmbeddingsProcessor
    consumes multi-layer Gemma hidden states). Downstream DiT wiring is not
    implemented yet -- this class only proves the encoder itself loads and
    runs correctly.
    """

    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT, device_index: int = 1):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(device_index))
        _ensure_comfy_on_path()

        import comfy.sd
        import comfy.utils

        self._comfy_sd = comfy.sd
        sd, _metadata = comfy.utils.load_torch_file(str(checkpoint_path), safe_load=True, return_metadata=True)
        te_model = comfy.sd.detect_te_model(sd)
        if te_model != comfy.sd.TEModel.GEMMA_4_12B:
            raise ValueError(f"Expected checkpoint to be detected as GEMMA_4_12B, got {te_model}")

        self.clip = comfy.sd.load_text_encoder_state_dicts(
            [sd],
            embedding_directory=None,
            clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
        )

    def encode(self, prompt: str):
        """Encode a single prompt. Returns the raw conditioning list from ComfyUI's CLIP.encode_from_tokens_scheduled."""
        tokens = self.clip.tokenize(prompt)
        return self.clip.encode_from_tokens_scheduled(tokens)


if __name__ == "__main__":
    import time

    import torch

    print("Loading Gemma 4 12B (LTX-2.5) text encoder...", flush=True)
    t0 = time.time()
    encoder = Gemma25TextEncoder()
    print(f"  ready in {time.time() - t0:.1f}s", flush=True)

    prompt = "A cinematic wide shot of a red fox running through fresh snow at sunset, warm rim light."
    t0 = time.time()
    cond = encoder.encode(prompt)
    print(f"  encoded in {time.time() - t0:.1f}s", flush=True)

    emb = cond[0][0] if isinstance(cond[0], (list, tuple)) else cond[0]
    print("  embedding shape:", tuple(emb.shape), "dtype:", emb.dtype)
    print("  has NaN:", bool(torch.isnan(emb).any()), "has Inf:", bool(torch.isinf(emb).any()))
    if torch.cuda.is_available():
        print("  peak CUDA mem (GB):", torch.cuda.max_memory_allocated() / 1e9)
