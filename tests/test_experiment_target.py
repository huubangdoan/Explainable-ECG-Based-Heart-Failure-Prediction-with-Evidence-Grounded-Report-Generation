"""Run only on the school CUDA target: ECG_TARGET_TESTS=1.

No Torch import occurs when the gate is absent. Every record below is synthetic;
these tests do not read or modify PTB-XL data. They exercise actual model forward,
backward and epoch-boundary recovery, not a mocked optimizer.
"""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


TARGET = os.environ.get("ECG_TARGET_TESTS") == "1"
if TARGET:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@unittest.skipUnless(TARGET, "Not verified here; explicit school-target ECG_TARGET_TESTS=1 required")
class TargetEngineTests(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():
            self.fail("Target tests require CUDA; no silent CPU substitution")

    def assert_state_equal(self, first, second):
        import torch
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_state_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_architecture_shapes_and_frequency_details(self):
        import torch
        from ptbxl_experiments.model import build_model
        counts = []
        for fs in (100, 200, 500):
            model = build_model(fs).cuda().eval()
            with torch.no_grad():
                result = model(torch.zeros(2, 12, int(fs * 2.5), device="cuda"))
            self.assertEqual(tuple(result.shape), (2, 5))
            self.assertTrue(torch.isfinite(result).all().item())
            self.assertEqual(model.architecture_details["block_strides"], [1, 2, 2])
            self.assertEqual(model.architecture_details["stem_kernel"], {100: 7, 200: 15, 500: 35}[fs])
            for layer in model.modules():
                if isinstance(layer, (torch.nn.Conv1d, torch.nn.Linear)) and layer.bias is not None:
                    self.assertEqual(int(torch.count_nonzero(layer.bias)), 0)
            counts.append(model.architecture_details["parameters"])
        self.assertTrue(counts[0] < counts[1] < counts[2])

    def test_crop_repeatability_record_alignment_and_max_probability_aggregation(self):
        import torch
        from ptbxl_experiments import training as engine
        x = np.zeros((2, 12, 1000), dtype=np.float32)
        x[0] = np.arange(1000, dtype=np.float32)
        x[1] = -np.arange(1000, dtype=np.float32)
        manifest = pd.DataFrame({"ecg_id": [13, 4], "patient_id": ["p13", "p4"],
                                 **{f"y_{name}": [1, 0] for name in engine.CLASSES}})
        payload = (x, manifest, np.zeros((12, 1)), np.ones((12, 1)), {"fs": 100})
        with patch.object(engine, "_load_prepared", return_value=payload):
            train = engine.ECGCropDataset(Path("unused"), "train", "global", 250,
                                          seed=42, epoch=3, stride=125)
            a, target, idx = train[0]
            b, _, _ = train[0]
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(tuple(a.shape), (12, 250))
            self.assertEqual(idx, 0)
            self.assertEqual(target.tolist(), [1] * 5)
            self.assertEqual(engine._window_starts(1000, 250, 125), list(range(0, 751, 125)))
            dataset = engine.ECGCropDataset(Path("unused"), "validation", "global", 250,
                                            seed=42, epoch=0, stride=125)

            class Probe(torch.nn.Module):
                def forward(self, signal):
                    # Increasing first record / decreasing second record:
                    # classwise max must come from different windows.
                    return (signal[:, 0, 0:1] / 1000).expand(-1, 5)

            predictions = engine._predict_dataset(Probe().cuda(), dataset, batch_size=3,
                                                   device=torch.device("cuda"), workers=0, amp=False)
        np.testing.assert_array_equal(predictions["ecg_id"], [13, 4])
        np.testing.assert_allclose(predictions["probabilities"][0], 1 / (1 + np.exp(-0.75)), rtol=1e-6)
        np.testing.assert_allclose(predictions["probabilities"][1], 0.5, rtol=1e-6)

    def test_epoch_resume_matches_uninterrupted_training_and_rejects_tampering(self):
        import torch
        from ptbxl_experiments import training as engine
        rng = np.random.default_rng(123)
        signals = rng.normal(size=(8, 12, 1000)).astype(np.float32)
        manifest = pd.DataFrame({"ecg_id": np.arange(8), "patient_id": [f"p{i}" for i in range(8)],
                                 **{f"y_{name}": (np.arange(8) + j) % 2
                                    for j, name in enumerate(engine.CLASSES)}})
        metadata = {"fs": 100, "fingerprint": "fixture-prepared", "source_fingerprint": "fixture-source",
                    "variant": {"id": "fixture", "fs": 100}}
        payload = (signals, manifest, np.zeros((12, 1)), np.ones((12, 1)), metadata)
        config = {"architecture": "resnet1d_wang", "epochs": 2, "batch_size": 4,
                  "learning_rate": 0.001, "weight_decay": 0.01, "crop_seconds": 2.5,
                  "eval_stride_seconds": 1.25, "aggregation": "max", "workers": 0,
                  "amp": True, "scheduler": "onecycle", "kernel_policy": "scale_odd"}
        job = {"id": "fixture_seed42", "experiment_id": "fixture", "seed": 42,
               "variant": {"id": "fixture", "fs": 100}, "normalization": "global", "training": config}
        identity = {"protocol_hash": "fixture", "code_fingerprint": "fixture-code",
                    "source_fingerprint": "fixture-source", "prepared_fingerprint": "fixture-prepared"}
        with tempfile.TemporaryDirectory() as temporary, patch.object(engine, "_load_prepared", return_value=payload):
            root = Path(temporary)
            full = root / "full"
            resumed = root / "resumed"
            engine.train_run(job, root, full, identity=identity)
            original = engine._atomic_torch_save
            failed = False

            def crash_after_committed_epoch(path, payload):
                nonlocal failed
                if Path(path).name == "best.pt" and not failed:
                    failed = True
                    raise RuntimeError("simulated interruption after last.pt commit")
                return original(path, payload)

            with patch.object(engine, "_atomic_torch_save", side_effect=crash_after_committed_epoch):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    engine.train_run(job, root, resumed, identity=identity)
            self.assertTrue((resumed / "last.pt").is_file())
            self.assertFalse((resumed / "best.pt").exists())
            engine.train_run(job, root, resumed, identity=identity)
            a = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
            b = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            for key in ("optimizer", "scheduler", "scaler", "rng", "best"):
                self.assert_state_equal(a[key], b[key])
            self.assertEqual(len(b["history"]), 2)
            np.testing.assert_array_equal(
                np.load(full / "validation_predictions.npz", allow_pickle=False)["probabilities"],
                np.load(resumed / "validation_predictions.npz", allow_pickle=False)["probabilities"])
            with self.assertRaisesRegex(ValueError, "identity"):
                engine.train_run(job, root, resumed, identity={**identity, "protocol_hash": "changed"})
            with (resumed / "validation_predictions.npz").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum|hash"):
                engine.train_run(job, root, resumed, identity=identity)


if __name__ == "__main__":
    unittest.main()
