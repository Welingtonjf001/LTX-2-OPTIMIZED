import tempfile
import unittest
from pathlib import Path

import torch

from tensorxx_ge.cache import ConditioningCache, build_cache_key
from tensorxx_ge.contracts import TextConditioning
from tensorxx_ge.devices import parse_cuda_device
from tensorxx_ge.export_onnx import _model_source_signature
from tensorxx_ge.metrics import compare_tensors
from tensorxx_ge.tensorrt_engine import tensorrt_status


class TensorXXGETests(unittest.TestCase):
    def setUp(self):
        self.conditioning = TextConditioning(
            video_encoding=torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
            audio_encoding=torch.ones(1, 3, 2),
            attention_mask=torch.tensor([[0, 1, 1]], dtype=torch.int64),
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.int64),
            raw_hidden_states=(torch.ones(1, 3, 4), torch.zeros(1, 3, 4)),
            metadata={"prompt": "test"},
        )

    def test_capture_round_trip_keeps_ltx_contract(self):
        reconstructed = TextConditioning.from_capture_payload(self.conditioning.capture_payload())

        self.assertTrue(torch.equal(reconstructed.video_encoding, self.conditioning.video_encoding))
        self.assertTrue(torch.equal(reconstructed.audio_encoding, self.conditioning.audio_encoding))
        self.assertTrue(torch.equal(reconstructed.attention_mask, self.conditioning.attention_mask))
        self.assertEqual(len(reconstructed.raw_hidden_states), 2)

    def test_cache_key_is_order_independent(self):
        first = build_cache_key({"prompt": "same", "max_length": 1024})
        second = build_cache_key({"max_length": 1024, "prompt": "same"})
        third = build_cache_key({"prompt": "different", "max_length": 1024})

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_projected_cache_excludes_raw_hidden_states(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ConditioningCache(directory)
            key = build_cache_key({"prompt": "cache"})
            path = cache.save(key, self.conditioning, {"prompt": "cache"})
            loaded, payload = cache.load(key)

            self.assertTrue(path.is_file())
            self.assertTrue(torch.equal(loaded.video_encoding, self.conditioning.video_encoding))
            self.assertIsNone(loaded.raw_hidden_states)
            self.assertEqual(payload["metadata"]["prompt"], "cache")

    def test_tensor_comparison_reports_exact_match(self):
        metrics = compare_tensors(self.conditioning.video_encoding, self.conditioning.video_encoding.clone())

        self.assertEqual(metrics["max_abs_error"], 0.0)
        self.assertEqual(metrics["relative_l2"], 0.0)
        self.assertEqual(metrics["cosine_similarity"], 1.0)

    def test_device_parsing_rejects_unindexed_cuda(self):
        self.assertEqual(parse_cuda_device("cuda:0"), torch.device("cuda:0"))
        with self.assertRaises(ValueError):
            parse_cuda_device("cuda")

    def test_tensorrt_status_is_queryable_without_runtime(self):
        status = tensorrt_status()

        self.assertIsInstance(status.available, bool)
        if not status.available:
            self.assertIsNotNone(status.detail)

    def test_model_source_signature_changes_with_local_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "model-00001.safetensors"
            shard.write_bytes(b"first")
            first = _model_source_signature(directory)
            self.assertEqual(first, _model_source_signature(directory))

            shard.write_bytes(b"changed-weights")
            self.assertNotEqual(first, _model_source_signature(directory))


if __name__ == "__main__":
    unittest.main()
