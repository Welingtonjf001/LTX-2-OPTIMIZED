"""Evaluate one LatentSync result with the official SyncNet confidence metric."""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latentsync-root", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--detect-dir", required=True)
    parser.add_argument("--temp-dir", required=True)
    args = parser.parse_args()

    root = os.path.abspath(args.latentsync_root)
    os.chdir(root)
    sys.path.insert(0, root)

    import torch

    from eval.eval_sync_conf import syncnet_eval
    from eval.syncnet import SyncNetEval
    from eval.syncnet_detect import SyncNetDetector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    syncnet = SyncNetEval(device=device)
    syncnet.loadParameters(os.path.abspath(args.model_path))
    detector = SyncNetDetector(
        device=device,
        detect_results_dir=os.path.abspath(args.detect_dir),
    )
    av_offset, confidence = syncnet_eval(
        syncnet,
        detector,
        os.path.abspath(args.video_path),
        os.path.abspath(args.temp_dir),
        detect_results_dir=os.path.abspath(args.detect_dir),
    )
    print(
        "SYNCNET_RESULT_JSON="
        + json.dumps(
            {"confidence": float(confidence), "av_offset": int(av_offset)},
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
