"""Offline protocol/orchestration tests. No Torch or actual dataset execution."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = importlib.import_module("ptbxl_experiments.protocol")
        self.config = self.protocol.load_protocol(ROOT / "configs/ptbxl_experiments.json")

    def results(self):
        return [dict(job_id=j["id"], experiment_id=j["experiment_id"], seed=j["seed"],
                     status="completed", val_metrics={"macro_auroc": .8, "macro_average_precision": .6})
                for j in self.protocol.build_jobs(self.config)]

    def test_exact_default_jobs_and_fair_controls(self):
        jobs = self.protocol.build_jobs(self.config)
        self.assertEqual(len(jobs), 21)
        self.assertEqual(len({j["id"] for j in jobs}), 21)
        self.assertEqual(jobs[0]["variant"]["source"], "official100")
        self.assertEqual(jobs[0]["normalization"], "global")
        self.assertEqual({j["seed"] for j in jobs}, {42, 2024, 3407})
        jobs[0]["training"]["epochs"] = 2
        self.assertEqual(self.config["training"]["epochs"], 50)

    def test_reject_unknown_duplicates_invalid_variant(self):
        for mutation in (lambda c: c.update(typo=True),
                         lambda c: c["seeds"].append(42),
                         lambda c: c["variants"][0].update(fs=200),
                         lambda c: c["training"].update(batch_size=True),
                         lambda c: c["experiments"][0].update(variant="missing"),
                         lambda c: c["experiments"][0].update(id="../outside")):
            config = copy.deepcopy(self.config)
            mutation(config)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.protocol.load_protocol(path)

    def test_selection_complete_unique_and_validation_only(self):
        results = self.results()
        winner = next(r for r in results if r["experiment_id"] == "reference100")
        winner["val_metrics"]["macro_auroc"] = .99
        selected = self.protocol.select_experiment(self.config, results)
        self.assertEqual(selected["selected_experiment"], "reference100")
        for invalid in (results[:-1], results + [results[0]],
                        [dict(r, status="failed") if i == 0 else r for i, r in enumerate(results)]):
            with self.assertRaises(ValueError):
                self.protocol.select_experiment(self.config, invalid)
        bad = copy.deepcopy(results)
        bad[0]["seed"] = 999
        with self.assertRaises(ValueError):
            self.protocol.select_experiment(self.config, bad)
        bad = copy.deepcopy(results)
        bad[0]["val_metrics"]["macro_auroc"] = None
        with self.assertRaises(ValueError):
            self.protocol.select_experiment(self.config, bad)

    def test_ties_use_ap_then_lexical_id_and_hash_canonical(self):
        results = self.results()
        for r in results:
            if r["experiment_id"] == "minimal500_global":
                r["val_metrics"]["macro_average_precision"] = .7
        self.assertEqual(self.protocol.select_experiment(self.config, results)["selected_experiment"],
                         "minimal500_global")
        self.assertEqual(self.protocol.protocol_hash(self.config),
                         self.protocol.protocol_hash(dict(reversed(list(self.config.items())))))

    def test_atomic_json_handles_numpy_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            self.protocol.write_json(path, {"x": np.array([1., float("nan")]), "n": np.int64(2)})
            self.assertEqual(json.loads(path.read_text()), {"x": [1., None], "n": 2})
            self.assertEqual(list(Path(tmp).iterdir()), [path])


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.suite = importlib.import_module("ptbxl_experiments.suite")

    def test_os_lock_excludes_second_owner_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "suite.lock"
            with self.suite.SuiteLock(path):
                with self.assertRaises(RuntimeError):
                    with self.suite.SuiteLock(path):
                        pass
            with self.suite.SuiteLock(path):
                self.assertTrue(path.exists())

    def test_frozen_selection_refuses_changed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.json"
            self.suite.freeze_json(path, {"winner": "a", "evidence": "1"})
            self.suite.freeze_json(path, {"winner": "a", "evidence": "1"})
            with self.assertRaises(ValueError):
                self.suite.freeze_json(path, {"winner": "b", "evidence": "2"})

    def test_scaler_snapshot_is_immutable_and_collected_without_signals(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ROOT / "configs/ptbxl_experiments.json"
            (root / "protocol.json").write_bytes(config.read_bytes())
            metadata = {"variant": {"id": "minimal500"}, "source_fingerprint": "fixture-source",
                        "normalization": {"global": {"mean": [.1], "std": [.9], "fit_split": "train"}},
                        "provenance": {"transform": "minimal", "runtime": "fixture"}}
            path = self.suite.snapshot_prepared_metadata(root, "minimal500", metadata)
            self.suite.snapshot_prepared_metadata(root, "minimal500", metadata)
            changed = copy.deepcopy(metadata)
            changed["normalization"]["global"]["mean"] = [2.]
            with self.assertRaises(ValueError):
                self.suite.snapshot_prepared_metadata(root, "minimal500", changed)
            self.assertEqual(json.loads(path.read_text()), metadata)
            with zipfile.ZipFile(self.suite.collect_results(root, include_source=False)) as archive:
                self.assertEqual(json.loads(archive.read("prepared_metadata/minimal500.json")), metadata)
                self.assertFalse(any(name.endswith(".npy") for name in archive.namelist()))

    def test_preflight_rejects_wrong_cuda_build_before_data_or_torch(self):
        from ptbxl_experiments.protocol import load_protocol
        config = load_protocol(ROOT / "configs/ptbxl_experiments.json")
        packages = {"numpy": "2.5.4", "pandas": "3.1.0", "scipy": "1.18.1", "wfdb": "4.3.1",
                    "matplotlib": "3.11.2", "requests": "2.34.2", "pytest": "9.1.1"}
        for version in ("2.14.1", "2.14.1+cpu", "2.14.1+cu130"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fake_data = types.SimpleNamespace(verify_release=lambda *a, **kw: self.fail("Should reject runtime before audit"))
                with patch.dict(sys.modules, {"ptbxl_experiments.data": fake_data}), \
                     patch.object(self.suite, "environment", return_value={"packages": dict(packages, torch=version)}):
                    with self.assertRaisesRegex(RuntimeError, "2.14.1\\+cu126"):
                        self.suite.preflight(config, root / "release", root / "raw", root / "work")

    def test_predictions_alignment_rejects_shuffled_ids_or_labels(self):
        a = {"ecg_id": np.array([1, 2]), "patient_id": np.array(["p1", "p2"]),
             "y_true": np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]),
             "probabilities": np.full((2, 5), .5)}
        self.assertEqual(self.suite.ensemble_predictions([a, a])["probabilities"].shape, (2, 5))
        b = dict(a, ecg_id=a["ecg_id"][::-1])
        with self.assertRaises(ValueError):
            self.suite.ensemble_predictions([a, b])
        b = dict(a, y_true=1-a["y_true"])
        with self.assertRaises(ValueError):
            self.suite.ensemble_predictions([a, b])

    def test_offline_import_path_does_not_import_torch(self):
        self.assertNotIn("torch", sys.modules)

    def test_collect_allowlist_excludes_cache_secrets_and_has_manifest(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "status.json").write_text('{"state":"failed"}')
            (root / ".env").write_text("not-a-real-secret")
            (root / "cache").mkdir()
            (root / "cache" / "x.npy").write_bytes(b"not-real-array")
            (root / "runs" / "job").mkdir(parents=True)
            (root / "runs" / "job" / "error.json").write_text('{"error":"fixture"}')
            bundle = self.suite.collect_results(root, include_source=False)
            with zipfile.ZipFile(bundle) as z:
                self.assertIn("status.json", z.namelist())
                self.assertIn("runs/job/error.json", z.namelist())
                self.assertIn("RESULTS_MANIFEST.json", z.namelist())
                self.assertFalse(any(".env" in x or "cache/" in x for x in z.namelist()))

    def _mock_suite(self, root, fail_first=False):
        from ptbxl_experiments.protocol import load_protocol, write_json
        config_path = ROOT / "configs/ptbxl_experiments.json"
        load_protocol(config_path)
        calls = []

        def train(job, directory, run_dir, **kwargs):
            calls.append(job["id"])
            if fail_first and len(calls) == 1:
                raise RuntimeError("synthetic failed job")
            run_dir.mkdir(parents=True, exist_ok=True)
            result = dict(job_id=job["id"], experiment_id=job["experiment_id"], seed=job["seed"],
                          status="completed", val_metrics={"macro_auroc": .8, "macro_average_precision": .6})
            write_json(run_dir / "result.json", result)
            (run_dir / "best.pt").write_bytes(b"fixture checkpoint; never loaded")
            (run_dir / "validation_predictions.npz").write_bytes(b"fixture evidence; never evaluated")
            return result

        fake_data = types.SimpleNamespace(prepare_variant=lambda release, raw, cache, variant, **kw: cache / variant["id"],
                                         verify_prepared=lambda directory: {"fingerprint": "fixture"})
        fake_training = types.SimpleNamespace(train_run=train)
        evaluated = []

        def evaluate(p, jobs, prepared, work_dir, selection, device):
            self.assertTrue((work_dir / "selection.json").exists())
            self.assertEqual(len(calls), 21)
            evaluated.append(selection)

        with patch.dict(sys.modules, {"ptbxl_experiments.data": fake_data, "ptbxl_experiments.training": fake_training}), \
             patch.object(self.suite, "preflight", return_value={}), \
             patch.object(self.suite, "_target_tests"), \
             patch.object(self.suite, "_check_common_cohort"), \
             patch.object(self.suite, "_naive_validation", return_value={}), \
             patch.object(self.suite, "_evaluate", side_effect=evaluate):
            code = self.suite.run_suite(config_path, root / "release", root / "raw", root / "work")
        return code, calls, evaluated

    def test_partial_suite_continues_jobs_but_never_opens_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code, calls, evaluated = self._mock_suite(root, fail_first=True)
            self.assertEqual(code, 1)
            self.assertEqual(len(calls), 21)
            self.assertEqual(evaluated, [])
            self.assertFalse((root / "work/selection.json").exists())
            self.assertEqual(json.loads((root / "work/status.json").read_text())["state"], "failed")
            self.assertTrue(list((root / "work/outbound").glob("*.zip")))

    def test_full_suite_freezes_selection_before_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code, calls, evaluated = self._mock_suite(root)
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 21)
            self.assertEqual(len(evaluated), 1)
            self.assertEqual(len(evaluated[0]["validation_evidence_sha256"]), 21)
            self.assertEqual(json.loads((root / "work/status.json").read_text())["state"], "completed")


if __name__ == "__main__":
    unittest.main()
