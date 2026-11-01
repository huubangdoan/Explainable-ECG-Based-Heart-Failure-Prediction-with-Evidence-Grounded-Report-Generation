"""Offline evaluation integration: real metrics/artifacts, no model or Torch.

The21 jobs come from the shipped protocol. Only bootstrap repetitions are reduced
to32 for these tiny fixtures. Checkpoint files contain fake bytes and are hashed,
never deserialized. Source fingerprint is fixed inside tests to isolate concurrent
development edits from the behavior under test.
"""
from __future__ import annotations

import builtins
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from ptbxl_experiments import metrics, protocol, suite

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CODE = "f" * 64


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = protocol.load_protocol(ROOT / "configs/ptbxl_experiments.json")
        self.config["evaluation"]["bootstrap_replicates"] = 32
        self.jobs = protocol.build_jobs(self.config)
        self.winner = "derived200"
        self.prepared = {variant["id"]: self.root / "cache" / variant["id"]
                         for variant in self.config["variants"]}
        self.validation, self.test_predictions = {}, {}
        self.populate()

    def tearDown(self):
        self.temp.cleanup()

    def prediction(self, job, *, split):
        y = np.tile(np.eye(5, dtype=np.int8), (2, 1))
        seed_offset = [-.02, 0., .02][self.config["seeds"].index(job["seed"])]
        if split == "validation":
            p = np.full((10, 5), .05)
            for column in range(5):
                p[column, column] = .28 + .03 * column
                p[column + 5, column] = .38 + .03 * column
                if job["experiment_id"] != self.winner:
                    p[(column + 1) % 5, column] = .9
            ids = np.arange(1, 11)
        else:
            # Test calibration deliberately differs: test-fitted thresholds would
            # be .8+, while validation thresholds are .28–.40. Leakage is visible.
            p = np.full((10, 5), .60)
            for column in range(5):
                p[y[:, column] == 1, column] = .8 + .01 * column
            ids = np.arange(101, 111)
        return {"ecg_id": ids, "patient_id": np.array([f"p{i // 2}" for i in range(10)]),
                "y_true": y, "probabilities": p + seed_offset}

    def populate(self):
        results = []
        for job in self.jobs:
            run = self.root / "runs" / job["id"]
            run.mkdir(parents=True, exist_ok=True)
            val = self.prediction(job, split="validation")
            self.validation[job["id"]] = val
            self.test_predictions[job["id"]] = self.prediction(job, split="test")
            suite.save_predictions(run / "validation_predictions.npz", val)
            (run / "best.pt").write_bytes(b"FAKE CHECKPOINT NEVER LOAD: " + job["id"].encode())
            (run / "last.pt").write_bytes(b"FAKE LAST CHECKPOINT NEVER LOAD")
            result = {"job_id": job["id"], "experiment_id": job["experiment_id"],
                      "seed": job["seed"], "status": "completed",
                      "val_metrics": metrics.multilabel_metrics(val["y_true"], val["probabilities"])}
            protocol.write_json(run / "result.json", result)
            results.append(result)
        self.selection = protocol.select_experiment(self.config, results)
        self.selection["identity"] = {"protocol_hash": protocol.protocol_hash(self.config),
                                      "source_fingerprint": self.config["source"]["expected_fingerprint"],
                                      "code_fingerprint": FIXTURE_CODE}
        self.selection["validation_evidence_sha256"] = suite._selection_evidence(results, self.root)
        protocol.write_json(self.root / "protocol.json", self.config)
        protocol.write_json(self.root / "selection.json", self.selection)
        protocol.write_json(self.root / "status.json", {"state": "completed"})
        self.assertEqual(len(self.jobs), 21)
        self.assertEqual(self.selection["selected_experiment"], self.winner)

    @contextmanager
    def evaluation_environment(self, *, forbid_predict=False):
        self.assertNotIn("torch", sys.modules)
        real_import = builtins.__import__
        def no_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("Torch import forbidden in offline evaluation fixtures")
            return real_import(name, *args, **kwargs)
        def predict(job, directory, run_dir, split, *, device):
            if forbid_predict:
                self.fail("Resume invoked predict_run instead of using bound cached test predictions")
            self.assertTrue((self.root / "selection.json").exists())
            self.assertEqual(directory, self.prepared[job["variant"]["id"]])
            self.assertEqual(run_dir, self.root / "runs" / job["id"])
            self.assertEqual(split, "test")
            self.assertEqual(device, "cuda")
            return copy.deepcopy(self.test_predictions[job["id"]])
        mocked = Mock(side_effect=predict)
        with patch.dict(sys.modules, {"ptbxl_experiments.training": types.SimpleNamespace(predict_run=mocked)}), \
             patch.object(suite, "code_fingerprint", return_value=FIXTURE_CODE), \
             patch.object(builtins, "__import__", side_effect=no_torch):
            yield mocked
        self.assertNotIn("torch", sys.modules)

    def evaluate(self):
        suite._evaluate(self.config, self.jobs, self.prepared, self.root, self.selection, "cuda")

    def test_only_reference_and_validation_winner_and_thresholds_from_validation(self):
        with self.evaluation_environment() as predict, \
             patch.object(metrics, "fit_thresholds", wraps=metrics.fit_thresholds) as fitted:
            self.evaluate()
        selected = {self.config["reference_experiment"], self.winner}
        self.assertEqual(predict.call_count, 6)
        self.assertEqual({call.args[0]["experiment_id"] for call in predict.call_args_list}, selected)
        self.assertEqual(fitted.call_count, 8)  #3seed +1ensemble per condition
        for call in fitted.call_args_list:
            self.assertTrue(np.any(call.args[1] < .1), "Threshold fitter received test probabilities")
        self.assertEqual({p.name for p in (self.root / "evaluation").iterdir()}, selected)
        for experiment in selected:
            path = self.root / "evaluation" / experiment
            val = suite.load_predictions(path / "ensemble_validation_predictions.npz")
            test = suite.load_predictions(path / "ensemble_test_predictions.npz")
            threshold_file = suite.read_json(path / "ensemble_thresholds.json")
            expected_threshold = metrics.fit_thresholds(val["y_true"], val["probabilities"])
            np.testing.assert_allclose(threshold_file["thresholds"], expected_threshold)
            self.assertEqual(threshold_file["fit_split"], "validation")
            self.assertFalse(np.allclose(expected_threshold,
                                        metrics.fit_thresholds(test["y_true"], test["probabilities"])))
            expected_metrics = metrics.multilabel_metrics(test["y_true"], test["probabilities"], expected_threshold)
            report = suite.read_json(path / "ensemble_metrics.json")
            self.assertEqual(report["metrics"], expected_metrics)
            self.assertLess(report["metrics"]["macro_f1"], 1.)
            self.assertEqual(report["patient_bootstrap"]["n_patients"], 5)
            self.assertEqual(report["patient_bootstrap"]["n_records"], 10)
            self.assertEqual(report["patient_bootstrap"]["n_bootstrap"], 32)
            self.assertEqual(report["selection_identity"], protocol.protocol_hash(self.selection))
            group = [self.validation[job["id"]]["probabilities"] for job in self.jobs
                     if job["experiment_id"] == experiment]
            np.testing.assert_allclose(val["probabilities"], np.mean(group, axis=0))

    def test_resume_reuses_bound_predictions_without_predicting(self):
        with self.evaluation_environment():
            self.evaluate()
        paths = list((self.root / "evaluation").glob("*/seed*/test_predictions.npz"))
        self.assertEqual(len(paths), 6)
        stamps = {path: (path.stat().st_mtime_ns, suite.file_hash(path)) for path in paths}
        first_metrics = {path: path.read_bytes() for path in (self.root / "evaluation").glob("*/ensemble_metrics.json")}
        with self.evaluation_environment(forbid_predict=True) as predict:
            self.evaluate()
            predict.assert_not_called()
        for path, expected in stamps.items():
            self.assertEqual((path.stat().st_mtime_ns, suite.file_hash(path)), expected)
        for path, contents in first_metrics.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_cached_prediction_modification_is_rejected_before_predicting(self):
        with self.evaluation_environment():
            self.evaluate()
        path = self.root / "evaluation" / self.config["reference_experiment"] / "seed42/test_predictions.npz"
        prediction = suite.load_predictions(path)
        prediction["probabilities"][0, 0] = .12345
        suite.save_predictions(path, prediction)
        with self.evaluation_environment(forbid_predict=True) as predict:
            with self.assertRaisesRegex(ValueError, "Changed test evidence"):
                self.evaluate()
            predict.assert_not_called()

    def test_changed_best_checkpoint_invalidates_cached_test(self):
        with self.evaluation_environment():
            self.evaluate()
        first = next(job for job in self.jobs if job["experiment_id"] == self.config["reference_experiment"])
        (self.root / "runs" / first["id"] / "best.pt").write_bytes(b"changed fake checkpoint")
        with self.evaluation_environment(forbid_predict=True) as predict:
            with self.assertRaisesRegex(ValueError, "Changed test evidence"):
                self.evaluate()
            predict.assert_not_called()

    def test_changed_frozen_validation_cannot_refit_cached_test_thresholds(self):
        with self.evaluation_environment():
            self.evaluate()
        first = next(job for job in self.jobs if job["experiment_id"] == self.config["reference_experiment"])
        path = self.root / "runs" / first["id"] / "validation_predictions.npz"
        prediction = suite.load_predictions(path)
        prediction["probabilities"] = 1 - prediction["probabilities"]
        suite.save_predictions(path, prediction)
        with self.evaluation_environment(forbid_predict=True) as predict:
            with self.assertRaisesRegex(ValueError, "Changed test evidence"):
                self.evaluate()
            predict.assert_not_called()

    def test_source_change_after_selection_is_rejected(self):
        with self.evaluation_environment(forbid_predict=True) as predict:
            with patch.object(suite, "code_fingerprint", return_value="0" * 64):
                with self.assertRaisesRegex(ValueError, "Source changed after validation selection"):
                    self.evaluate()
            predict.assert_not_called()

    def test_when_reference_wins_it_is_evaluated_once(self):
        self.winner = self.config["reference_experiment"]
        self.populate()
        with self.evaluation_environment() as predict:
            self.evaluate()
        self.assertEqual(predict.call_count, 3)
        self.assertEqual({p.name for p in (self.root / "evaluation").iterdir()}, {self.winner})

    def test_result_bundle_has_evidence_checkpoints_and_verifiable_manifest(self):
        with self.evaluation_environment():
            self.evaluate()
            summary = suite.summarize(self.root)
            cache = self.root / "cache"
            cache.mkdir()
            (cache / "unshared.npy").write_bytes(b"synthetic excluded waveform")
            (self.root / ".env").write_bytes(b"synthetic excluded file")
            bundle = suite.collect_results(self.root, include_source=False)
        self.assertIn(self.winner, summary.read_text(encoding="utf-8"))
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            manifest = json.loads(archive.read("RESULTS_MANIFEST.json"))
            self.assertIn("selection.json", names)
            self.assertIn("SUMMARY.vi.md", names)
            self.assertIn(f"evaluation/{self.winner}/ensemble_metrics.json", names)
            for job in self.jobs:
                self.assertIn(f"runs/{job['id']}/best.pt", names)
                self.assertIn(f"runs/{job['id']}/validation_predictions.npz", names)
            self.assertFalse(any(name.startswith("cache/") or name.endswith(".env") for name in names))
            self.assertEqual(set(manifest["files"]), set(names) - {"RESULTS_MANIFEST.json"})
            for name, expected in manifest["files"].items():
                payload = archive.read(name)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), expected["sha256"])
                self.assertEqual(len(payload), expected["size"])
        checksum = bundle.with_suffix(".zip.sha256").read_text().split()[0]
        self.assertEqual(checksum, suite.file_hash(bundle))


if __name__ == "__main__":
    unittest.main()
