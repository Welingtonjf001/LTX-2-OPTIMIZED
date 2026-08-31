import tempfile
import unittest

import comfy_ingredients_patch
import torch
from accelerate import dispatch_model
from ltx_core.model.transformer.attention import Attention, AttentionFunction
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator, LTXVideoOnlyModelConfigurator
from ltx_pipelines.utils.prompt_cache import prompt_cache_path


class OptimizationPathTests(unittest.TestCase):
    def test_ingredients_resize_uses_requested_dimensions(self):
        link = ["source", 0]
        inputs = comfy_ingredients_patch.resize_inputs(link, 640, 384)

        self.assertEqual(
            inputs,
            {
                "input": link,
                "resize_type": "scale dimensions",
                "resize_type.width": 640,
                "resize_type.height": 384,
                "resize_type.crop": "center",
                "scale_method": "lanczos",
            },
        )

    def test_ingredients_resize_preserves_legacy_fallback(self):
        inputs = comfy_ingredients_patch.resize_inputs(["source", 0])

        self.assertEqual(inputs["resize_type"], "scale shorter dimension")
        self.assertEqual(inputs["resize_type.shorter_size"], 544)

    def test_prompt_cache_path_matches_native_cache_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = prompt_cache_path("same prompt", cache_dir=temp_dir)
            second = prompt_cache_path("same prompt", seed=999, cache_dir=temp_dir)
            changed = prompt_cache_path("different prompt", cache_dir=temp_dir)

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertTrue(first.endswith(".pt"))

    def test_accelerate_keeps_ltx_blocks_indivisible(self):
        self.assertEqual(LTXModelConfigurator.no_split_modules, ["BasicAVTransformerBlock"])
        self.assertEqual(LTXVideoOnlyModelConfigurator.no_split_modules, ["BasicAVTransformerBlock"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required to exercise Accelerate CPU offload")
    def test_attention_norms_materialize_from_accelerate_offload(self):
        attention = Attention(
            query_dim=4,
            heads=1,
            dim_head=4,
            attention_function=AttentionFunction.PYTORCH,
        )
        attention = dispatch_model(
            attention,
            device_map={
                "q_norm": "cpu",
                "k_norm": "cpu",
                "to_q": 0,
                "to_k": 0,
                "to_v": 0,
                "to_out": 0,
            },
        )

        output = attention(torch.randn(1, 2, 4, device="cuda"))

        self.assertEqual(tuple(output.shape), (1, 2, 4))
        self.assertEqual(output.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
