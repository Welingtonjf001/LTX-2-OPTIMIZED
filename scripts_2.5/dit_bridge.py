"""LTX-2.5 DiT (transformer) + embeddings-processor bridge.

Reuses existing ltx_core building blocks UNMODIFIED (LTXModelConfigurator,
Embeddings1DConnector[Configurator], FeatureExtractorV2, EmbeddingsProcessor) --
these already implement the "22B / caption_proj_before_connector" pattern the
2.5 checkpoint uses (see CLAUDE.md, section "LTX-2.5"). Only new code here is:

- Gemma25EmbeddingsProcessorConfigurator: same shape as ltx_core's
  EmbeddingsProcessorConfigurator, but sized for the Gemma 4 12B encoder
  (hidden_size=3840, 49 hidden-state layers) instead of Gemma 3.
- Loading from two checkpoint shards (DiT + text encoder) instead of one,
  since LTX-2.5 ships split per-component files.

Known gap: `keyframes_abs_pos_embedding` (multishot-related, see CLAUDE.md) is
not wired into LTXModel and is silently dropped on load -- harmless for plain
T2V, would matter for native multishot.
"""

import sys
from pathlib import Path

import torch

CORE_SRC = Path(r"E:\Users\home\Documents\LTX-2-OPTIMIZED\packages\ltx-core\src")
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder  # noqa: E402
from ltx_core.model.model_protocol import ModelConfigurator  # noqa: E402
from ltx_core.model.transformer import LTXModelConfigurator, X0Model  # noqa: E402
from ltx_core.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP  # noqa: E402
from ltx_core.text_encoders.gemma import EMBEDDINGS_PROCESSOR_KEY_OPS  # noqa: E402
from ltx_core.text_encoders.gemma.embeddings_connector import (  # noqa: E402
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnectorConfigurator,
)
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor  # noqa: E402
from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV2  # noqa: E402

DIT_CHECKPOINT = Path(
    r"E:\Users\home\Documents\LTX-2-OPTIMIZED\models\2.5\diffusion_models\ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)
GEMMA_CHECKPOINT = Path(
    r"E:\Users\home\Documents\LTX-2-OPTIMIZED\models\2.5\text_encoders\gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
)

GEMMA4_HIDDEN_SIZE = 3840
GEMMA4_NUM_LAYERS = 48 + 1  # 48 transformer layers + embedding layer = 49 hidden-state entries


class Gemma25EmbeddingsProcessorConfigurator(ModelConfigurator[EmbeddingsProcessor]):
    """Same role as ltx_core's EmbeddingsProcessorConfigurator, sized for Gemma 4 12B."""

    @classmethod
    def from_config(cls, config: dict) -> EmbeddingsProcessor:
        transformer_config = config.get("transformer", {})
        video_connector = Embeddings1DConnectorConfigurator.from_config(config)
        audio_connector = AudioEmbeddings1DConnectorConfigurator.from_config(config)

        flat_dim = GEMMA4_HIDDEN_SIZE * GEMMA4_NUM_LAYERS
        video_inner_dim = transformer_config["num_attention_heads"] * transformer_config["attention_head_dim"]
        audio_inner_dim = (
            transformer_config["audio_num_attention_heads"] * transformer_config["audio_attention_head_dim"]
        )
        feature_extractor = FeatureExtractorV2(
            video_aggregate_embed=torch.nn.Linear(flat_dim, video_inner_dim, bias=True),
            embedding_dim=GEMMA4_HIDDEN_SIZE,
            audio_aggregate_embed=torch.nn.Linear(flat_dim, audio_inner_dim, bias=True),
        )
        return EmbeddingsProcessor(
            video_connector=video_connector,
            audio_connector=audio_connector,
            feature_extractor=feature_extractor,
        )


def _transformer_builder() -> SingleGPUModelBuilder:
    return SingleGPUModelBuilder(
        model_class_configurator=LTXModelConfigurator,
        model_path=str(DIT_CHECKPOINT),
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )


def _embeddings_processor_builder() -> SingleGPUModelBuilder:
    return SingleGPUModelBuilder(
        model_class_configurator=Gemma25EmbeddingsProcessorConfigurator,
        # DiT checkpoint first: SingleGPUModelBuilder.model_config() reads metadata
        # from shard[0] only, and the "transformer" config dict lives in the DiT
        # checkpoint's metadata, not the text encoder's.
        model_path=(str(DIT_CHECKPOINT), str(GEMMA_CHECKPOINT)),
        model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
    )


def load_transformer(
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    max_memory: dict[int | str, str] | None = None,
) -> X0Model:
    velocity_model = _transformer_builder().build(device=device, dtype=dtype, max_memory=max_memory)
    return X0Model(velocity_model)


def load_embeddings_processor(device: torch.device, dtype: torch.dtype = torch.bfloat16) -> EmbeddingsProcessor:
    return _embeddings_processor_builder().build(device=device, dtype=dtype)


def audit_missing_unexpected_keys(builder: SingleGPUModelBuilder) -> tuple[list[str], list[str]]:
    """Report which state-dict keys the meta model expects vs. what the checkpoint(s) provide, after sd_ops."""
    config = builder.model_config()
    meta_model = builder.meta_model(config, builder.module_ops)
    model_paths = list(builder.model_path) if isinstance(builder.model_path, tuple) else [builder.model_path]
    state_dict = builder.load_sd(model_paths, sd_ops=builder.model_sd_ops, registry=builder.registry, device=torch.device("cpu"))
    expected = set(meta_model.state_dict().keys())
    provided = set(state_dict.sd.keys())
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    return missing, unexpected


if __name__ == "__main__":
    print("=== Auditing transformer state dict ===", flush=True)
    missing, unexpected = audit_missing_unexpected_keys(_transformer_builder())
    print(f"  missing: {len(missing)}, unexpected: {len(unexpected)}")
    for k in missing[:20]:
        print("   missing:", k)
    for k in unexpected[:20]:
        print("   unexpected:", k)

    print("=== Auditing embeddings processor state dict ===", flush=True)
    missing2, unexpected2 = audit_missing_unexpected_keys(_embeddings_processor_builder())
    print(f"  missing: {len(missing2)}, unexpected: {len(unexpected2)}")
    for k in missing2[:20]:
        print("   missing:", k)
    for k in unexpected2[:20]:
        print("   unexpected:", k)
