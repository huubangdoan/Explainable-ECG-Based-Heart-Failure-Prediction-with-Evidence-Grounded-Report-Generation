"""Synthetic recovery only: no Torch import, network, model, or real ECG files."""
from contextlib import contextmanager, redirect_stdout
import ast
import copy
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ptbxl_experiments import data, protocol
from ptbxl_experiments.suite import SuiteLock, file_hash, read_json
from ptbxl_pipeline import bootstrap, recovery
from test_experiment_data import bind, make_release

ROOT = Path(__file__).resolve().parents[1]


class FixtureCodec:
    """JSON checkpoints and the unchanged completion validator, without Torch."""
    def __init__(self):
        tree = ast.parse((ROOT / "ptbxl_experiments/training.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_verify_completed")
        namespace = {"Path": Path, "json": json, "_object_hash": protocol.protocol_hash,
                     "_sha256": file_hash, "_ARTIFACTS": recovery.ARTIFACTS}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "training._verify_completed", "exec"), namespace)
        self.verify_completed = namespace["_verify_completed"]

    def load(self, path):
        return read_json(path)

    def save(self, stream, value):
        stream.write(json.dumps(value, sort_keys=True).encode("utf-8"))

    def assert_equal(self, first, second):
        if first != second:
            raise ValueError("Checkpoint payload invariance failed")


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old, self.new, self.raw = (self.root / n for n in ("old work", "new work", "raw"))
        self.old.mkdir()
        self.raw.mkdir()
        self.catalog_hash = data.OFFICIAL_CHECKSUMS_SHA256
        if self._testMethodName.startswith("test_official100"):
            lines = []
            for index in range(1, 7):
                for suffix in ("hea", "dat"):
                    relative = f"records100/00000/{index:05d}_lr.{suffix}"
                    path = self.old / "cache/source100" / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"synthetic {index} {suffix}".encode())
                    lines.append(f"{file_hash(path)}  {relative}\n")
            (self.raw / "SHA256SUMS.txt").write_text("".join(lines), encoding="ascii")
            self.catalog_hash = file_hash(self.raw / "SHA256SUMS.txt")
        self.release = make_release(self.root)
        pipeline = read_json(self.release / "pipeline.json")
        pipeline.update(implementation_sha256=file_hash(ROOT / "ptbxl_pipeline/core.py"),
                        checksum_catalog_sha256=self.catalog_hash,
                        source_metadata_sha256={"ptbxl_database.csv": "d" * 64, "scp_statements.csv": "e" * 64})
        protocol.write_json(self.release / "pipeline.json", pipeline)
        bind(self.release)
        self.contract = dict(schema_version=1, source_version="1.0.3", classes=pipeline["classes"],
                             leads=pipeline["leads"], source_records=6, accepted_records=6,
                             split_counts={s: 2 for s in data.SPLITS},
                             checksum_catalog_sha256=pipeline["checksum_catalog_sha256"],
                             source_metadata_sha256=pipeline["source_metadata_sha256"],
                             producer_implementation_sha256=pipeline["implementation_sha256"],
                             cohort_sha256=bootstrap.cohort_digest(self.release / "records.csv"),
                             reference_release={"fingerprint": "b" * 64, "audit_sha256": "f" * 64})
        self.base = protocol.load_protocol(ROOT / "configs/ptbxl_experiments.json")
        self.base["source"].update(expected_records=6, expected_accepted=6,
                                    expected_fingerprint="b" * 64, expected_audit_sha256="f" * 64)
        self.config, self.contract_path, self.certificate = (self.root / n for n in ("base.json", "contract.json", "certificate.json"))
        protocol.write_json(self.config, self.base)
        protocol.write_json(self.contract_path, self.contract)
        self.cert = read_json(ROOT / "configs/official100_recovery_v1.json")
        protocol.write_json(self.certificate, self.cert)
        self.target = copy.deepcopy(self.base)
        self.target["source"].update(expected_fingerprint="a" * 64,
                                      expected_audit_sha256=file_hash(self.release / "audit.json"))
        self.identity = dict(protocol_hash=protocol.protocol_hash(self.target),
                             code_fingerprint=self.cert["old_code_fingerprint"], source_fingerprint="a" * 64)
        for name, value in (("protocol.json", self.target), ("target_protocol.json", self.target),
                            ("identity.json", self.identity), ("cohort_contract.json", self.contract)):
            protocol.write_json(self.old / name, value)
        self.jobs = protocol.build_jobs(self.target)
        errors = [{"phase": "prepare", "variant": "official100_minimal",
                   "error": "Official100 header contract mismatch: records100/00000/00001_lr"}]
        errors.extend({"phase": "train", "job_id": j["id"], "error": "prepared variant unavailable"}
                      for j in self.jobs if j["variant"]["source"] == "official100")
        protocol.write_json(self.old / "errors.json", errors)
        self.caches = {}
        # Fixtures use small, physical synthetic arrays; only the old identity is simulated.
        with redirect_stdout(io.StringIO()):
            for variant in self.target["variants"]:
                if variant["source"] == "official100":
                    continue
                prepared = data.prepare_variant(self.release, self.raw, self.old / "cache", variant)
                metadata = read_json(prepared / "variant.json")
                metadata["provenance"]["implementation_sha256"] = self.cert["old_data_sha256"]
                metadata["fingerprint"] = data._digest(metadata["provenance"])
                metadata["metadata_sha256"] = data._digest({k: v for k, v in metadata.items() if k != "metadata_sha256"})
                protocol.write_json(prepared / "variant.json", metadata)
                old_path = prepared.with_name(f"{variant['id']}-{metadata['fingerprint'][:16]}")
                prepared.rename(old_path)
                self.caches[variant["id"]] = old_path
        self.runtime = {"fixture": "same target runtime"}
        for job in self.jobs:
            if job["variant"]["source"] == "official100":
                continue
            directory = self.old / "runs" / job["id"]
            metadata = read_json(self.caches[job["variant"]["id"]] / "variant.json")
            binding = dict(schema_version=1, job=job, runtime=self.runtime,
                           identity=dict(self.identity, prepared_fingerprint=metadata["fingerprint"]))
            protocol.write_json(directory / "identity.json", binding)
            history = [{"epoch": i, "elapsed_seconds": i * 3.25} for i in range(1, job["training"]["epochs"] + 1)]
            protocol.write_json(directory / "history.json", history)
            (directory / "history.csv").write_text("epoch,elapsed_seconds\n1,3.25\n", encoding="utf-8")
            best = {"epoch": 7, "score": .8, "model": {"weights": [0., 1., -2.]}}
            protocol.write_json(directory / "last.pt", dict(schema_version=1, binding=binding,
                                epoch=job["training"]["epochs"], model={"weights": [3., 4.]}, optimizer={"step": 50},
                                scheduler={"last_epoch": 50}, scaler={"scale": 65536.}, rng={"python": [1, 2]},
                                history=history, best=best, elapsed_seconds=170., peak_gpu_memory_bytes=1024))
            protocol.write_json(directory / "best.pt", dict(schema_version=1, binding=binding, **best))
            frame = data._frame(self.release / "validation.csv", accepted=True)
            np.savez_compressed(directory / "validation_predictions.npz", ecg_id=frame.ecg_id.to_numpy(),
                                patient_id=frame.patient_id.to_numpy(dtype=str),
                                y_true=frame[list(data.LABEL_COLUMNS)].to_numpy(dtype=np.float32),
                                probabilities=np.full((len(frame), 5), .25))
            np.save(directory / "val_thresholds.npy", np.full(5, .25), allow_pickle=False)
            protocol.write_json(directory / "result.json", dict(job_id=job["id"], experiment_id=job["experiment_id"],
                                seed=job["seed"], status="completed", val_metrics={"macro_auroc": .8},
                                run_dir=str(directory), identity_hash=protocol.protocol_hash(binding),
                                epochs_completed=job["training"]["epochs"], best_epoch=7, completed_at="original-time"))
            self.reseal(directory)
        self.codec = FixtureCodec()

    def tearDown(self):
        self.temp.cleanup()

    def reseal(self, directory):
        protocol.write_json(directory / "completed.json", dict(schema_version=1,
                            identity_hash=protocol.protocol_hash(read_json(directory / "identity.json")),
                            artifact_sha256={n: file_hash(directory / n) for n in recovery.ARTIFACTS},
                            completed_at="original-marker-time"))

    def kwargs(self):
        return dict(old_work_dir=self.old, work_dir=self.new, release_dir=self.release, raw_dir=self.raw,
                    certificate_path=self.certificate, contract_path=self.contract_path, config_path=self.config)

    @contextmanager
    def target_runtime(self):
        with patch.object(recovery, "_target_runtime", return_value=self.runtime), \
             patch.object(recovery, "_checkpoint_codec", return_value=self.codec), \
             patch.object(data, "OFFICIAL_CHECKSUMS_SHA256", self.catalog_hash), redirect_stdout(io.StringIO()):
            yield

    def migrate(self):
        with self.target_runtime():
            return recovery.migrate(**self.kwargs())

    def snapshot(self, root):
        return {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob("*")
                if p.is_file() and p.name != "suite.lock"}

    def test_plan_readonly_and_18_3(self):
        before = self.snapshot(self.old)
        with patch.object(recovery, "_target_runtime") as runtime, patch.object(recovery, "_checkpoint_codec") as codec:
            plan = recovery.plan(**self.kwargs())
        self.assertEqual(len(plan["imported_job_ids"]), 18)
        self.assertEqual(len(plan["remaining_job_ids"]), 3)
        self.assertGreaterEqual(plan["headroom_bytes"], 8 * 1024**3)
        self.assertFalse(self.new.exists())
        self.assertFalse((self.old / "suite.lock").exists())
        self.assertEqual(before, self.snapshot(self.old))
        runtime.assert_not_called()
        codec.assert_not_called()
        self.assertNotIn("torch", sys.modules)

    def test_migration_invariant_idempotent_and_normal_validator_accepts(self):
        before = self.snapshot(self.old)
        target = self.migrate()
        self.assertEqual(target, self.new / "target_protocol.json")
        proof = read_json(self.new / "bootstrap_verification.json")
        self.assertEqual(len(proof["imported_job_ids"]), 18)
        self.assertEqual(len(proof["remaining_job_ids"]), 3)
        for job in (j for j in self.jobs if j["variant"]["source"] == "release500"):
            source, copied = self.old / "runs" / job["id"], self.new / "runs" / job["id"]
            binding = read_json(copied / "identity.json")
            self.codec.verify_completed(copied, binding)
            for name in ("history.json", "history.csv", "validation_predictions.npz", "val_thresholds.npy"):
                self.assertEqual((source / name).read_bytes(), (copied / name).read_bytes())
            for name in ("best.pt", "last.pt"):
                a, b = read_json(source / name), read_json(copied / name)
                a.pop("binding")
                b.pop("binding")
                self.assertEqual(a, b)
            a, b = read_json(source / "result.json"), read_json(copied / "result.json")
            for key in ("identity_hash", "run_dir"):
                a.pop(key)
                b.pop(key)
            self.assertEqual(a, b)
        ready_before = self.snapshot(self.new)
        self.migrate()
        self.assertEqual(ready_before, self.snapshot(self.new))
        self.assertEqual(before, self.snapshot(self.old))
        self.assertNotIn("torch", sys.modules)

    def test_unrelated_destination_and_test_evidence_refused(self):
        self.new.mkdir()
        (self.new / "keep.txt").write_text("retain")
        with self.assertRaisesRegex(ValueError, "destination|Destination|NEW"):
            self.migrate()
        self.assertEqual((self.new / "keep.txt").read_text(), "retain")
        self.new = self.root / "newer"
        for name, value in (("selection.json", {}), ("status.json", {"state": "completed"})):
            protocol.write_json(self.old / name, value)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "test|Test|completed"):
                self.migrate()
            (self.old / name).unlink()
        (self.old / "evaluation").mkdir()
        with self.assertRaisesRegex(ValueError, "test|Test|evaluation"):
            self.migrate()

    def test_identity_certificate_protocol_runtime_and_incident_rejected(self):
        paths = [(self.old / "identity.json", lambda x: x.update(code_fingerprint="0" * 64)),
                 (self.certificate, lambda x: x.update(new_data_sha256="0" * 64)),
                 (self.old / "target_protocol.json", lambda x: x["training"].update(epochs=2)),
                 (self.old / "errors.json", lambda x: x.clear())]
        for path, change in paths:
            original = path.read_bytes()
            value = json.loads(original)
            change(value)
            protocol.write_json(path, value)
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.migrate()
            path.write_bytes(original)
        self.runtime = {"fixture": "changed"}
        with self.assertRaisesRegex(ValueError, "runtime"):
            self.migrate()
        self.assertFalse((self.new / "bootstrap_verification.json").exists())

    def test_corrupt_artifact_missing_extra_and_reference_job_rejected(self):
        run = next((self.old / "runs").iterdir())
        artifact = run / "history.csv"
        original = artifact.read_bytes()
        artifact.write_bytes(original + b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum|SHA256"):
            self.migrate()
        artifact.write_bytes(original)
        marker = (run / "completed.json").read_bytes()
        (run / "completed.json").unlink()
        with self.assertRaisesRegex(ValueError, "completed|completion"):
            self.migrate()
        (run / "completed.json").write_bytes(marker)
        for name in ("unexpected", "reference100__seed42"):
            extra = self.old / "runs" / name
            extra.mkdir()
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.migrate()
            extra.rmdir()

    def test_interruption_retry_and_corrupt_destination_no_overwrite(self):
        original = self.codec.save
        calls = 0
        def interrupt(stream, payload):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise KeyboardInterrupt("fixture interrupted")
            original(stream, payload)
        with patch.object(self.codec, "save", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.migrate()
        self.assertFalse((self.new / "bootstrap_verification.json").exists())
        self.assertTrue((self.new / "bootstrap_error.json").exists())
        self.migrate()
        output = next((self.new / "runs").glob("*/history.csv"))
        output.write_bytes(b"unexpected replacement")
        with self.assertRaisesRegex(ValueError, "checksum|SHA256|changed"):
            self.migrate()
        self.assertEqual(output.read_bytes(), b"unexpected replacement")

    def test_lock_conflicts_make_no_diagnostic_writes(self):
        with SuiteLock(self.old / "suite.lock"), self.assertRaises(RuntimeError):
            self.migrate()
        self.assertFalse(self.new.exists())
        self.migrate()
        before = self.snapshot(self.new)
        with SuiteLock(self.new / "suite.lock"), self.assertRaises(RuntimeError):
            self.migrate()
        self.assertEqual(before, self.snapshot(self.new))

    def test_cohort_and_cache_label_drift_rejected_even_when_resealed(self):
        cache = next(iter(self.caches.values()))
        frame = pd.read_csv(cache / "validation.csv")
        frame.loc[0, "y_MI"] = 1 - frame.loc[0, "y_MI"]
        frame.to_csv(cache / "validation.csv", index=False)
        metadata = read_json(cache / "variant.json")
        metadata["splits"]["validation"]["manifest_sha256"] = file_hash(cache / "validation.csv")
        metadata["metadata_sha256"] = data._digest({k: v for k, v in metadata.items() if k != "metadata_sha256"})
        protocol.write_json(cache / "variant.json", metadata)
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.migrate()
        self.assertFalse((self.new / "bootstrap_verification.json").exists())

    def test_run_never_launches_after_failure_and_releases_locks(self):
        with self.target_runtime(), patch("ptbxl_experiments.suite.run_suite", return_value=0) as run:
            self.assertEqual(recovery.run(**self.kwargs()), 0)
            self.assertEqual(run.call_args.kwargs["device"], "cuda")
            with SuiteLock(self.new / "suite.lock"):
                pass
        protocol.write_json(self.certificate, dict(self.cert, new_code_fingerprint="0" * 64))
        with self.target_runtime(), patch("ptbxl_experiments.suite.run_suite") as run, self.assertRaises(ValueError):
            recovery.run(**self.kwargs())
        run.assert_not_called()

    def test_publication_interrupts_keep_refusal_until_durable_proof(self):
        freeze, publish = recovery.freeze_json, recovery._publish
        for stage in ("before_proof", "after_proof", "after_identity"):
            self.new = self.root / stage
            def interrupt_freeze(path, value):
                if Path(path).name == "bootstrap_verification.json" and stage == "before_proof":
                    raise KeyboardInterrupt("before proof")
                freeze(path, value)
            def interrupt_publish(work, journal, identity):
                if stage == "after_proof":
                    raise KeyboardInterrupt("after proof before identity")
                publish(work, journal, identity)
                if stage == "after_identity":
                    raise KeyboardInterrupt("after identity")
            with self.subTest(stage=stage), patch.object(recovery, "freeze_json", side_effect=interrupt_freeze), \
                 patch.object(recovery, "_publish", side_effect=interrupt_publish), self.assertRaises(KeyboardInterrupt):
                self.migrate()
            identity = dict(self.identity, code_fingerprint=self.cert["new_code_fingerprint"])
            if stage != "after_identity":
                with self.assertRaisesRegex(ValueError, "Frozen evidence changed"):
                    freeze(self.new / "identity.json", identity)
            proof_path = self.new / "bootstrap_verification.json"
            previous = proof_path.read_bytes() if proof_path.exists() else None
            self.migrate()
            self.assertEqual(read_json(self.new / "identity.json"), identity)
            if previous is not None:
                self.assertEqual(proof_path.read_bytes(), previous)

    def test_official100_prefetch_reused_and_corruption_refused(self):
        before = self.snapshot(self.old)
        self.assertEqual(recovery.plan(**self.kwargs())["official100_files_to_copy"], 12)
        self.migrate()
        proof = read_json(self.new / "bootstrap_verification.json")
        self.assertEqual(proof["official100_files_copied"], 12)
        self.assertEqual(self.snapshot(self.old / "cache/source100"), self.snapshot(self.new / "cache/source100"))
        self.assertEqual(before, self.snapshot(self.old))
        source = next((self.old / "cache/source100").rglob("*.dat"))
        source.write_bytes(b"corruption")
        self.new = self.root / "other destination"
        with self.assertRaisesRegex(ValueError, "checksum|SHA256"):
            self.migrate()
        self.assertFalse((self.new / "bootstrap_verification.json").exists())

    def test_payload_change_and_prediction_label_drift_fail_closed(self):
        original = self.codec.save
        def corrupt(stream, payload):
            changed = copy.deepcopy(payload)
            changed["model"]["weights"][0] = 999
            original(stream, changed)
        with patch.object(self.codec, "save", side_effect=corrupt), self.assertRaisesRegex(ValueError, "invariance"):
            self.migrate()
        self.assertFalse((self.new / "bootstrap_verification.json").exists())
        self.new = self.root / "newer destination"
        source = next((self.old / "runs").iterdir())
        with np.load(source / "validation_predictions.npz") as saved:
            prediction = {name: saved[name] for name in saved.files}
        prediction["y_true"][0, 0] = 1 - prediction["y_true"][0, 0]
        np.savez_compressed(source / "validation_predictions.npz", **prediction)
        self.reseal(source)
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.migrate()

    def test_threshold_no_positive_sentinel_is_accepted(self):
        directory = next((self.old / "runs").iterdir())
        np.save(directory / "val_thresholds.npy", np.full(5, np.nextafter(1., np.inf)), allow_pickle=False)
        recovery._validate_predictions(directory, self.release)
        np.save(directory / "val_thresholds.npy", np.full(5, 1.01), allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "threshold"):
            recovery._validate_predictions(directory, self.release)

    def test_completed_reference_and_test_evidence_preserved_only_after_publication(self):
        self.migrate()
        evidence = self.new / "evaluation/reference100/saved.json"
        protocol.write_json(evidence, {"keep": True})
        reference = self.new / "runs/reference100__seed42/result.json"
        protocol.write_json(reference, {"keep": "subsequent suite result"})
        protocol.write_json(self.new / "status.json", {"state": "completed"})
        before = self.snapshot(self.new)
        self.migrate()
        self.assertEqual(before, self.snapshot(self.new))

    def test_disk_budget_stops_before_any_copy_and_reuses_verified_files(self):
        with patch.object(recovery.shutil, "disk_usage", return_value=type("Disk", (), {"free": 0})()), \
             patch.object(recovery, "_copy") as copier, self.assertRaisesRegex(OSError, "disk"):
            self.migrate()
        copier.assert_not_called()
        self.assertFalse((self.new / "bootstrap_verification.json").exists())
        self.migrate()
        observed = []
        original = recovery._disk
        def check(*args):
            result = original(*args)
            observed.append(result)
            return result
        with patch.object(recovery, "_disk", side_effect=check):
            self.migrate()
        self.assertGreater(observed[0]["verified_existing_copy_bytes"], 0)

    def test_migration_disk_headroom_satisfies_unchanged_suite_boundary(self):
        from ptbxl_experiments import suite
        # Size metadata only: no multi-GB arrays or real cohort files are created.
        target = copy.deepcopy(self.base)
        target["source"]["expected_accepted"] = 21387
        sample_bytes = 21387 * 12 * 10 * 4
        imported_bytes = sample_bytes * sum(v["fs"] for v in target["variants"] if v["source"] == "release500")
        imported = SimpleNamespace(is_file=lambda: True, is_symlink=lambda: False,
                                   stat=lambda: SimpleNamespace(st_size=imported_bytes))
        (self.new / "cache").mkdir(parents=True)
        reserved = recovery._headroom({"protocol": target})
        with patch.object(Path, "rglob", return_value=[imported]), \
             patch.object(shutil, "disk_usage", return_value=SimpleNamespace(free=reserved)):
            budget = suite._disk_budget(target, self.new)
        self.assertGreaterEqual(reserved, budget["required_free_bytes_estimate"])
        self.assertEqual(budget["required_free_bytes_estimate"], 12846062016)

    def test_duplicate_cache_and_changed_cache_runtime_rejected(self):
        cache = next(iter(self.caches.values()))
        duplicate = cache.with_name("duplicate-cache")
        shutil.copytree(cache, duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.migrate()
        # Remove only the temporary fixture tree this test created.
        shutil.rmtree(duplicate)
        metadata = read_json(cache / "variant.json")
        metadata["provenance"]["runtime"]["python"] = "changed-runtime"
        metadata["fingerprint"] = data._digest(metadata["provenance"])
        metadata["metadata_sha256"] = data._digest({k: v for k, v in metadata.items() if k != "metadata_sha256"})
        protocol.write_json(cache / "variant.json", metadata)
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse((self.new / "bootstrap_verification.json").exists())

    def test_overlapping_roots_and_linked_files_rejected(self):
        kwargs = self.kwargs()
        kwargs["work_dir"] = self.old / "inside"
        with self.assertRaisesRegex(ValueError, "separate"):
            recovery.plan(**kwargs)
        self.assertFalse((self.old / "inside").exists())
        source = next((self.old / "runs").glob("*/last.pt"))
        linked = self.root / "hardlinked-checkpoint.pt"
        try:
            os.link(source, linked)
        except OSError as exc:
            self.skipTest(f"Filesystem does not support fixture hardlinks: {exc}")
        with self.assertRaisesRegex(ValueError, "unlinked"):
            self.migrate()

    def test_hardlinked_empty_suite_locks_are_rejected_before_writes(self):
        for directory in (self.old, self.new):
            directory.mkdir(exist_ok=True)
            lock = directory / "suite.lock"
            lock.write_bytes(b"")
            sibling = self.root / f"{directory.name}-lock-sibling"
            try:
                os.link(lock, sibling)
            except OSError as exc:
                self.skipTest(f"Filesystem does not support fixture hardlinks: {exc}")
            with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "unlinked"):
                self.migrate()
            self.assertEqual(lock.read_bytes(), b"")
            self.assertEqual(sibling.read_bytes(), b"")
            sibling.unlink()


if __name__ == "__main__":
    unittest.main()
