"""School bootstrap fixtures only; acquisition/stage calls are mocked, never network/model."""
from contextlib import contextmanager
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from ptbxl_pipeline import bootstrap
from test_experiment_data import bind, make_release, sha

ROOT = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.release = make_release(self.root)
        self.raw, self.cache, self.work = (self.root / n for n in ("raw", "source_cache", "work"))
        self.raw.mkdir()
        pipeline = json.loads((self.release / "pipeline.json").read_text())
        pipeline.update(implementation_sha256=sha(ROOT / "ptbxl_pipeline/core.py"),
                        checksum_catalog_sha256="c" * 64,
                        source_metadata_sha256={"ptbxl_database.csv": "d" * 64, "scp_statements.csv": "e" * 64})
        (self.release / "pipeline.json").write_text(json.dumps(pipeline))
        bind(self.release)
        self.contract = dict(schema_version=1, source_version="1.0.3", classes=pipeline["classes"],
                             leads=pipeline["leads"], source_records=6, accepted_records=6,
                             split_counts={"train": 2, "validation": 2, "test": 2},
                             checksum_catalog_sha256=pipeline["checksum_catalog_sha256"],
                             source_metadata_sha256=pipeline["source_metadata_sha256"],
                             producer_implementation_sha256=pipeline["implementation_sha256"],
                             cohort_sha256=bootstrap.cohort_digest(self.release / "records.csv"),
                             reference_release={"fingerprint": "b" * 64, "audit_sha256": "f" * 64})
        self.contract_path = self.root / "cohort.json"
        self.contract_path.write_text(json.dumps(self.contract))
        self.base = json.loads((ROOT / "configs/ptbxl_experiments.json").read_text())
        self.base["source"].update(expected_records=6, expected_accepted=6,
                                    expected_fingerprint="b" * 64, expected_audit_sha256="f" * 64)
        self.config = self.root / "base.json"
        self.config.write_text(json.dumps(self.base))

    def tearDown(self):
        self.temp.cleanup()

    @contextmanager
    def no_download(self):
        report = {"verified": True, "record_count": 6, "source_version": "1.0.3",
                  "checksum_manifest_sha256": self.contract["checksum_catalog_sha256"],
                  "file_hashes": self.contract["source_metadata_sha256"]}
        with patch.object(bootstrap, "_ensure_source", return_value=report) as source, \
             patch.object(bootstrap, "_check_disk", return_value={"fixture": True}), \
             patch.object(bootstrap, "_pipeline_stages") as stages:
            yield source, stages

    def prepare(self):
        return bootstrap.prepare_school(self.config, self.contract_path, self.raw, self.release,
                                        self.cache, self.work, allow_download=False)

    def test_canonical_digest_preserves_meaning_across_order_and_integer_format(self):
        path = self.release / "records.csv"
        original = bootstrap.cohort_digest(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            row["patient_id"] = str(int(float(row["patient_id"])))
            row["ecg_id"] += ".0"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(reversed(rows))
        self.assertEqual(original, bootstrap.cohort_digest(path))

    def test_canonical_rejects_duplicate_noninteger_and_nonbinary_values(self):
        frame = pd.read_csv(self.release / "records.csv")
        for field, value in (("ecg_id", 2), ("patient_id", 1.25), ("y_MI", 2), ("status", "unknown")):
            changed = frame.copy()
            changed[field] = changed[field].astype(object)
            changed.loc[0, field] = value
            path = self.root / "invalid.csv"
            changed.to_csv(path, index=False)
            with self.subTest(field=field), self.assertRaises(ValueError):
                bootstrap.cohort_digest(path)

    def test_semantic_digest_binds_label_patient_split_and_waveform_hash(self):
        frame = pd.read_csv(self.release / "records.csv")
        expected = bootstrap.cohort_digest(self.release / "records.csv")
        for field, value in (("y_MI", 1), ("patient_id", 12345), ("split", "validation"),
                             ("signal_sha256", "0" * 64)):
            changed = frame.copy()
            changed.loc[0, field] = value
            path = self.root / "changed.csv"
            changed.to_csv(path, index=False)
            with self.subTest(field=field):
                self.assertNotEqual(expected, bootstrap.cohort_digest(path))

    def test_equivalence_accepts_runtime_identity_change_but_rejects_contract_drift(self):
        result = bootstrap.verify_equivalent_release(self.release, self.contract)
        self.assertNotEqual(result["source_fingerprint"], self.contract["reference_release"]["fingerprint"])
        self.assertEqual(result["cohort_sha256"], self.contract["cohort_sha256"])
        for field in ("cohort_sha256", "checksum_catalog_sha256", "producer_implementation_sha256"):
            bad = dict(self.contract, **{field: "0" * 64})
            with self.subTest(field=field), self.assertRaises(ValueError):
                bootstrap.verify_equivalent_release(self.release, bad)

    def test_full_equivalence_checks_waveform_bytes(self):
        with (self.release / "signals/00001.npy").open("ab") as handle:
            handle.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum|SHA256"):
            bootstrap.verify_equivalent_release(self.release, self.contract)

    def test_ready_reuse_rebinds_only_source_anchors_and_is_idempotent(self):
        with self.no_download() as (source, stages):
            target = self.prepare()
            first = {name: (self.work / name).read_bytes() for name in (
                "target_protocol.json", "cohort_contract.json", "bootstrap_verification.json")}
            self.assertEqual(self.prepare(), target)
            stages.assert_not_called()
            self.assertEqual(source.call_count, 2)
            self.assertFalse(source.call_args.kwargs["download"])
        derived = json.loads(target.read_text())
        self.assertEqual(derived["source"]["expected_fingerprint"], "a" * 64)
        self.assertEqual(derived["source"]["expected_audit_sha256"], sha(self.release / "audit.json"))
        derived["source"] = self.base["source"]
        self.assertEqual(derived, self.base)
        for name, contents in first.items():
            self.assertEqual((self.work / name).read_bytes(), contents)
        self.assertNotIn("torch", sys.modules)

    def test_bad_ready_release_is_not_rebuilt_or_bound(self):
        with (self.release / "signals/00001.npy").open("ab") as handle:
            handle.write(b"bad")
        with self.no_download() as (_, stages):
            with self.assertRaises(ValueError):
                self.prepare()
            stages.assert_not_called()
        self.assertFalse((self.work / "target_protocol.json").exists())
        self.assertTrue((self.work / "bootstrap_error.json").exists())

    def test_changed_work_contract_refuses_before_acquisition(self):
        with self.no_download():
            self.prepare()
        changed = dict(self.contract, cohort_sha256="0" * 64)
        self.contract_path.write_text(json.dumps(changed))
        with self.no_download() as (source, _):
            with self.assertRaisesRegex(ValueError, "contract|evidence|Frozen"):
                self.prepare()
            source.assert_not_called()

    def test_changed_base_or_active_suite_identity_refuses_before_acquisition(self):
        with self.no_download():
            self.prepare()
        base = copy.deepcopy(self.base)
        base["training"]["epochs"] = 2
        self.config.write_text(json.dumps(base))
        with self.no_download() as (source, _):
            with self.assertRaises(ValueError):
                self.prepare()
            source.assert_not_called()

    def test_incomplete_release_runs_existing_stages_then_binds_after_audit(self):
        (self.release / "audit.json").unlink()
        pipeline = json.loads((self.release / "pipeline.json").read_text())
        pipeline["runtime"] = bootstrap._runtime()
        (self.release / "pipeline.json").write_text(json.dumps(pipeline))
        with self.no_download() as (_, stages):
            # Restore a successful fixture audit only after the mocked stage sequence.
            def finish(config):
                audit = {"ready": True, "checks_passed": True, "fixture": False, "errors": [],
                         "fingerprint": "a" * 64, "source_version": "1.0.3", "source_records": 6,
                         "accepted_records": 6, "split_counts": self.contract["split_counts"],
                         "cross_split_duplicates": []}
                (self.release / "audit.json").write_text(json.dumps(audit))
                bind(self.release)
            stages.side_effect = finish
            self.prepare()
            self.assertEqual(stages.call_count, 1)
        self.assertTrue((self.work / "target_protocol.json").exists())

    def test_incomplete_release_runtime_mismatch_refuses_before_acquisition(self):
        (self.release / "audit.json").unlink()
        with self.no_download() as (source, stages):
            with self.assertRaisesRegex(ValueError, "runtime|identity"):
                self.prepare()
            source.assert_not_called()
            stages.assert_not_called()

    def test_source_certificate_drift_or_incompleteness_blocks_release_binding(self):
        for field, value in (("verified", False), ("record_count", 5), ("source_version", "1.0.2"),
                             ("checksum_manifest_sha256", "0" * 64), ("file_hashes", {})):
            with self.subTest(field=field), self.no_download() as (source, stages):
                source.return_value[field] = value
                with self.assertRaises(ValueError):
                    self.prepare()
                stages.assert_not_called()
                self.assertFalse((self.work / "target_protocol.json").exists())

    def test_stage_failure_does_not_bind_or_continue(self):
        pipeline = json.loads((self.release / "pipeline.json").read_text())
        pipeline["runtime"] = bootstrap._runtime()
        (self.release / "pipeline.json").write_text(json.dumps(pipeline))
        (self.release / "audit.json").unlink()
        with self.no_download() as (_, stages):
            stages.side_effect = ValueError("stage fixture failed")
            with self.assertRaisesRegex(ValueError, "stage fixture"):
                self.prepare()
        self.assertFalse((self.work / "target_protocol.json").exists())

    def test_active_suite_identity_and_lock_are_not_overwritten(self):
        with self.no_download():
            target = self.prepare()
        protocol = json.loads(target.read_text())
        (self.work / "protocol.json").write_text(json.dumps(protocol))
        identity = {"protocol_hash": bootstrap.protocol_hash(protocol), "code_fingerprint": "0" * 64,
                    "source_fingerprint": protocol["source"]["expected_fingerprint"]}
        (self.work / "identity.json").write_text(json.dumps(identity))
        with self.no_download() as (source, _):
            with self.assertRaisesRegex(ValueError, "identity"):
                self.prepare()
            source.assert_not_called()
        before = (self.work / "bootstrap.log").read_bytes()
        with bootstrap.SuiteLock(self.work / "suite.lock"), self.no_download() as (source, _):
            with self.assertRaisesRegex(RuntimeError, "lock"):
                self.prepare()
            source.assert_not_called()
        self.assertEqual((self.work / "bootstrap.log").read_bytes(), before)

    def test_download_or_stage_failure_never_binds_protocol(self):
        with self.no_download() as (source, stages):
            source.side_effect = RuntimeError("interrupted source fixture")
            with self.assertRaisesRegex(RuntimeError, "interrupted source"):
                self.prepare()
            stages.assert_not_called()
        self.assertFalse((self.work / "target_protocol.json").exists())
        with self.no_download():
            self.prepare()
        self.assertTrue((self.work / "target_protocol.json").exists())

    def test_nonempty_unknown_release_and_overlap_are_rejected(self):
        (self.release / "pipeline.json").unlink()
        with self.no_download() as (source, _):
            with self.assertRaises(ValueError):
                self.prepare()
            source.assert_not_called()
        with self.assertRaises(ValueError):
            bootstrap.prepare_school(self.config, self.contract_path, self.raw, self.release,
                                     self.raw / "cache", self.work)

    def test_cli_plan_is_offline_and_run_does_not_follow_preparation_failure(self):
        spec = importlib.util.spec_from_file_location("bootstrap_cli", ROOT / "scripts/bootstrap_school.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = ["--config", str(self.config), "--contract", str(self.contract_path),
                "--raw-dir", str(self.raw), "--release-dir", str(self.release),
                "--cache-dir", str(self.cache), "--work-dir", str(self.work)]
        with patch.object(bootstrap, "prepare_school") as prepare:
            self.assertEqual(module.main(["plan", *args]), 0)
            prepare.assert_not_called()
            prepare.side_effect = ValueError("fixture preparation failure")
            with patch("ptbxl_experiments.suite.run_suite") as run:
                self.assertEqual(module.main(["run", *args]), 1)
                run.assert_not_called()
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
