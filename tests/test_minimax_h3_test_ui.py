import unittest
from unittest.mock import patch

from minimax_h3_test_ui import build_workflow, validate_environment, VIDEO_VAE, AUDIO_VAE


class MiniMaxWorkflowTests(unittest.TestCase):
    def test_av_contract_and_separate_decoders(self):
        graph = build_workflow("Bamboo forest", turbo=False)
        self.assertEqual(graph["6"]["inputs"]["model"], ["1", 0])
        self.assertEqual(graph["2"]["inputs"]["type"], "minimax")
        self.assertEqual(graph["10"]["inputs"]["latent_image"], ["5", 1])
        self.assertEqual(graph["5"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(graph["3"]["inputs"]["vae_name"], VIDEO_VAE)
        self.assertEqual(graph["4"]["inputs"]["vae_name"], AUDIO_VAE)
        self.assertEqual(graph["13"]["inputs"]["audio"], ["12", 0])
        forbidden = {"ModelSamplingAuraFlow", "EmptySD3LatentImage", "SaveImage"}
        self.assertFalse(forbidden & {n["class_type"] for n in graph.values()})

    def test_turbo_and_references_are_connected(self):
        graph = build_workflow("Use <Picture 1>.", refs=["a.png", "b.png"])
        self.assertEqual(graph["9"]["inputs"]["steps"], 4)
        self.assertEqual(graph["6"]["inputs"]["model"], ["15", 0])
        self.assertEqual(graph["5"]["inputs"]["ref_images.ref_image_1"], ["21", 0])
        self.assertIn("<Picture 2>", graph["5"]["inputs"]["prompt"])

    def test_frame_grid_and_invalid_parameters(self):
        for seconds in [1, 5, 10, 15]:
            length = build_workflow("test", seconds=seconds)["5"]["inputs"]["length"]
            self.assertEqual(length % 17, 5)
            self.assertGreaterEqual(length, round(seconds * 24))
        for params in [dict(width=641), dict(seconds=40), dict(seconds=float("nan")), dict(seed=-1)]:
            with self.assertRaises(ValueError):
                build_workflow("test", **params)

    def test_missing_node_is_reported_before_submission(self):
        with patch("minimax_h3_test_ui.request", return_value={}):
            with self.assertRaisesRegex(ValueError, "UNETLoader"):
                validate_environment(build_workflow("test"))


if __name__ == "__main__":
    unittest.main()
