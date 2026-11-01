"""Small synthetic fixtures only; never opens the real PTB-XL release."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import wfdb

from ptbxl_experiments import data


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def bind(release):
    """Rebind fixture envelopes so structural tests get past checksum checks."""
    pipeline = json.loads((release / "pipeline.json").read_text())
    stats = json.loads((release / "normalization.json").read_text())
    stats["records_sha256"] = sha(release / "records.csv")
    dump(release / "normalization.json", stats)
    pipeline.update(metadata_sha256=sha(release / "metadata.csv"),
                    records_sha256=sha(release / "records.csv"),
                    normalization_sha256=sha(release / "normalization.json"))
    dump(release / "pipeline.json", pipeline)
    audit = json.loads((release / "audit.json").read_text())
    audit["artifact_sha256"] = {name: sha(release / name) for name in data.ARTIFACTS}
    dump(release / "audit.json", audit)


def make_release(root):
    release = root / "release"
    release.mkdir()
    (release / "signals").mkdir()
    rows = []
    for index in range(6):
        split = ("train", "train", "validation", "validation", "test", "test")[index]
        t = np.arange(5000) / 500
        values = np.stack([(lead + 1) * np.sin(2 * np.pi * (1 + index / 10) * t)
                           + lead / 10 + (0 if split == "train" else 100)
                           for lead in range(12)]).astype(np.float32)
        npy_path = f"signals/{index + 1:05d}.npy"
        np.save(release / npy_path, values, allow_pickle=False)
        row = dict(ecg_id=index + 1, patient_id=float(index + 1),
                   split=split, strat_fold=1 if split == "train" else 9 if split == "validation" else 10,
                   filename_lr=f"records100/00000/{index + 1:05d}_lr",
                   filename_hr=f"records500/00000/{index + 1:05d}_hr",
                   baseline_drift="" if index % 2 else "I", static_noise="",
                   has_label=1, status="accepted", reason="", npy_path=npy_path,
                   signal_sha256=sha(release / npy_path), fs=500, n_samples=5000, units="mV")
        row.update({f"y_{name}": int(i == index % 5) for i, name in enumerate(data.CLASSES)})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(release / "records.csv", index=False)
    frame.drop(columns=["status", "reason", "npy_path", "signal_sha256", "fs", "n_samples", "units"]).to_csv(release / "metadata.csv", index=False)
    for split in data.SPLITS:
        frame[frame.split == split].to_csv(release / f"{split}.csv", index=False)
    fingerprint = "a" * 64
    dump(release / "pipeline.json", dict(fingerprint=fingerprint, classes=list(data.CLASSES),
         leads=list(data.LEADS), source_version="1.0.3", source_records=6,
         config={"target_fs": 500, "preprocessing": "minimal", "expected_records": 6},
         runtime={"python": "different-producer-version"}))
    dump(release / "normalization.json", dict(fingerprint=fingerprint, fit_split="train",
         train_records=2, classes=list(data.CLASSES), leads=list(data.LEADS), units="mV", ddof=0,
         mean=[0.] * 12, std=[1.] * 12, samples_per_lead=10000))
    dump(release / "audit.json", dict(ready=True, checks_passed=True, fixture=False, errors=[],
         fingerprint=fingerprint, source_version="1.0.3", source_records=6, accepted_records=6,
         split_counts={split: 2 for split in data.SPLITS}, cross_split_duplicates=[]))
    bind(release)
    return release


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.release = make_release(self.root)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.variant = {"id": "minimal500", "source": "release500", "fs": 500, "filter": "minimal"}

    def tearDown(self):
        self.tmp.cleanup()

    def edit_frames(self, callback):
        for name in ("records", "metadata", *data.SPLITS):
            path = self.release / f"{name}.csv"
            frame = pd.read_csv(path)
            callback(frame)
            frame.to_csv(path, index=False)
        bind(self.release)

    def test_portable_audit_accepts_changed_producer_runtime(self):
        result = data.verify_release(self.release, expected_fingerprint="a" * 64,
                                     expected_audit_sha256=sha(self.release / "audit.json"))
        self.assertEqual(result["accepted_records"], 6)
        self.assertEqual(result["split_counts"], dict(train=2, validation=2, test=2))

    def test_anchor_and_manifest_corruption_fail(self):
        with self.assertRaisesRegex(ValueError, "audit.*SHA256|audit.*checksum"):
            data.verify_release(self.release, expected_audit_sha256="0" * 64)
        with (self.release / "train.csv").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum|SHA256"):
            data.verify_release(self.release)

    def test_label_drift_and_row_order_fail_even_after_rebinding(self):
        frame = pd.read_csv(self.release / "train.csv")
        frame.loc[0, "y_MI"] = 1
        frame.to_csv(self.release / "train.csv", index=False)
        bind(self.release)
        with self.assertRaisesRegex(ValueError, "alignment|khớp"):
            data.verify_release(self.release)

    def test_patient_overlap_and_fold_mismatch_fail(self):
        self.edit_frames(lambda f: f.loc.__setitem__((f.ecg_id == 3, "patient_id"), 1))
        with self.assertRaisesRegex(ValueError, "patient|bệnh nhân"):
            data.verify_release(self.release)

    def test_fold_mapping_is_verified(self):
        self.edit_frames(lambda f: f.loc.__setitem__((f.ecg_id == 3, "strat_fold"), 1))
        with self.assertRaisesRegex(ValueError, "fold"):
            data.verify_release(self.release)

    def test_array_checksum_fail(self):
        path = self.release / "signals/00001.npy"
        values = np.load(path, allow_pickle=False)
        values[0, 0] += 1
        np.save(path, values, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "checksum|SHA256"):
            data.verify_release(self.release)

    def test_array_finiteness_checked_even_with_new_hash(self):
        path = self.release / "signals/00001.npy"
        values = np.load(path, allow_pickle=False)
        values[0, 0] = np.nan
        np.save(path, values, allow_pickle=False)
        for name in ("records", "train"):
            frame = pd.read_csv(self.release / f"{name}.csv")
            frame.loc[frame.ecg_id == 1, "signal_sha256"] = sha(path)
            frame.to_csv(self.release / f"{name}.csv", index=False)
        bind(self.release)
        with self.assertRaisesRegex(ValueError, "finite|NaN"):
            data.verify_release(self.release)

    def test_wrong_array_dtype_is_rejected_after_rebinding(self):
        path = self.release / "signals/00001.npy"
        values = np.load(path, allow_pickle=False).astype(np.float64)
        np.save(path, values, allow_pickle=False)
        for name in ("records", "train"):
            frame = pd.read_csv(self.release / f"{name}.csv")
            frame.loc[frame.ecg_id == 1, "signal_sha256"] = sha(path)
            frame.to_csv(self.release / f"{name}.csv", index=False)
        bind(self.release)
        with self.assertRaisesRegex(ValueError, "dtype"):
            data.verify_release(self.release)

    def test_duplicate_waveform_across_split_rejected(self):
        first = self.release / "signals/00001.npy"
        third = self.release / "signals/00003.npy"
        third.write_bytes(first.read_bytes())
        for name in ("records", "validation"):
            frame = pd.read_csv(self.release / f"{name}.csv")
            frame.loc[frame.ecg_id == 3, "signal_sha256"] = sha(third)
            frame.to_csv(self.release / f"{name}.csv", index=False)
        bind(self.release)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            data.verify_release(self.release)

    def test_path_traversal_rejected(self):
        for name in ("records", "train"):
            frame = pd.read_csv(self.release / f"{name}.csv")
            frame.loc[frame.ecg_id == 1, "npy_path"] = "../outside.npy"
            frame.to_csv(self.release / f"{name}.csv", index=False)
        bind(self.release)
        with self.assertRaisesRegex(ValueError, "path|đường dẫn|Đường dẫn"):
            data.verify_release(self.release)

    def test_train_only_global_and_per_lead_statistics(self):
        directory = data.prepare_variant(self.release, self.raw, self.root / "cache", self.variant)
        X, manifest, mean, std, metadata = data.load_prepared(directory, "train", "per_lead")
        expected = np.stack([np.load(self.release / f"signals/{i:05d}.npy") for i in (1, 2)])
        np.testing.assert_array_equal(X, expected)
        np.testing.assert_allclose(mean[:, 0], expected.mean(axis=(0, 2), dtype=np.float64), atol=1e-10)
        np.testing.assert_allclose(std[:, 0], expected.std(axis=(0, 2), dtype=np.float64), atol=1e-10)
        _, _, gmean, gstd, _ = data.load_prepared(directory, "train", "global")
        np.testing.assert_allclose(gmean, expected.mean(dtype=np.float64), atol=1e-10)
        np.testing.assert_allclose(gstd, expected.std(dtype=np.float64), atol=1e-10)
        self.assertLess(float(gmean[0, 0]), 1)
        self.assertEqual(manifest.ecg_id.astype(int).tolist(), [1, 2])
        self.assertIn("baseline_drift", manifest.columns)
        self.assertEqual(metadata["normalization"]["global"]["fit_split"], "train")
        self.assertEqual(data.verify_prepared(directory)["fingerprint"], metadata["fingerprint"])

    def test_cache_reuse_is_immutable_and_corruption_fails(self):
        directory = data.prepare_variant(self.release, self.raw, self.root / "cache", self.variant)
        stamp = (directory / "variant.json").stat().st_mtime_ns
        self.assertEqual(directory, data.prepare_variant(self.release, self.raw, self.root / "cache", self.variant))
        self.assertEqual(stamp, (directory / "variant.json").stat().st_mtime_ns)
        with (directory / "train.npy").open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum|SHA256"):
            data.prepare_variant(self.release, self.raw, self.root / "cache", self.variant)

    def test_normalization_metadata_tamper_and_no_ready_fail(self):
        directory = data.prepare_variant(self.release, self.raw, self.root / "cache", self.variant)
        meta = json.loads((directory / "variant.json").read_text())
        meta["normalization"]["global"]["mean"][0] += 10
        dump(directory / "variant.json", meta)
        with self.assertRaises(ValueError):
            data.load_prepared(directory, "train", "global")
        (directory / "variant.json").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            data.load_prepared(directory, "train", "global")

    def test_interruption_reuses_only_committed_splits(self):
        real_array = data._array
        def interrupted(path, expected_hash, fs):
            if path.name == "00003.npy":
                raise RuntimeError("synthetic interruption after train split")
            return real_array(path, expected_hash, fs)
        with patch.object(data, "_array", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                data.prepare_variant(self.release, self.raw, self.root / "cache", self.variant)
        directory = next((self.root / "cache").glob("minimal500-*"))
        self.assertFalse((directory / "variant.json").exists())
        stamp = (directory / "train.npy").stat().st_mtime_ns
        with patch.object(data, "_array", wraps=real_array) as source_read:
            resumed = data.prepare_variant(self.release, self.raw, self.root / "cache", self.variant)
            self.assertEqual(source_read.call_count, 4)
        self.assertEqual(resumed, directory)
        self.assertEqual(stamp, (directory / "train.npy").stat().st_mtime_ns)
        self.assertTrue(data.verify_prepared(directory)["ready"])

    def test_resample_axis_and_filter_full_record(self):
        for fs, filt in ((100, "minimal"), (200, "minimal"), (500, "bandpass_0.5_40")):
            with self.subTest(fs=fs, filter=filt):
                variant = dict(self.variant, id=f"case_{fs}_{filt}", fs=fs, filter=filt)
                directory = data.prepare_variant(self.release, self.raw, self.root / "cache", variant)
                X, _, _, _, _ = data.load_prepared(directory, "train", "global")
                self.assertEqual(X.shape, (2, 12, fs * 10))
                self.assertTrue(np.isfinite(X).all())

    def test_cache_cannot_be_inside_source(self):
        with self.assertRaises(ValueError):
            data.prepare_variant(self.release, self.raw, self.release / "cache", self.variant)

    def make_official100(self):
        source = self.root / "remote"
        subdir = source / "records100/00000"
        subdir.mkdir(parents=True)
        for i in range(1, 7):
            t = np.arange(1000) / 100
            values = np.stack([np.sin(2 * np.pi * (lead + 1) * t) + i / 10 for lead in range(12)], axis=1)
            wfdb.wrsamp(f"{i:05d}_lr", fs=100, units=["mV"] * 12,
                        sig_name=list(data.LEADS), p_signal=values, fmt=["16"] * 12, write_dir=str(subdir))
        paths = sorted(source.rglob("*.*"))
        catalog = "".join(f"{sha(path)}  {path.relative_to(source).as_posix()}\n" for path in paths)
        (self.raw / "SHA256SUMS.txt").write_text(catalog, encoding="utf-8")
        content = {path.relative_to(source).as_posix(): path.read_bytes() for path in paths}
        return sha(self.raw / "SHA256SUMS.txt"), content

    def test_official100_requires_permission_and_uses_verified_native_data(self):
        anchor, content = self.make_official100()
        variant = {"id": "official100", "source": "official100", "fs": 100, "filter": "minimal"}
        with patch.object(data, "OFFICIAL_CHECKSUMS_SHA256", anchor):
            with self.assertRaisesRegex(FileNotFoundError, "allow.download|download"):
                data.prepare_variant(self.release, self.raw, self.root / "cache", variant)
            def response(request, **kwargs):
                url = request.full_url if hasattr(request, "full_url") else request
                return io.BytesIO(content[url.removeprefix(data.OFFICIAL_BASE_URL)])
            with patch.object(data.urllib.request, "urlopen", side_effect=response) as mock:
                directory = data.prepare_variant(self.release, self.raw, self.root / "cache", variant, allow_download=True)
                self.assertEqual(mock.call_count, 12)
            X, _, _, _, metadata = data.load_prepared(directory, "train", "global")
            np.testing.assert_allclose(X[0, 0].mean(), .1, atol=.001)
            self.assertEqual(metadata["variant"]["source"], "official100")
            self.assertEqual(X.shape, (2, 12, 1000))

    def test_untrusted_official_catalog_rejected(self):
        self.make_official100()
        variant = {"id": "official100", "source": "official100", "fs": 100, "filter": "minimal"}
        with self.assertRaisesRegex(ValueError, "catalog|official|chính thức"):
            data.prepare_variant(self.release, self.raw, self.root / "cache", variant, allow_download=True)

    def test_existing_corrupt_official_file_is_not_replaced(self):
        anchor, content = self.make_official100()
        cache = self.root / "cache"
        existing = cache / "source100/records100/00000/00001_lr.hea"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"corrupt existing header")
        variant = {"id": "official100", "source": "official100", "fs": 100, "filter": "minimal"}
        with patch.object(data, "OFFICIAL_CHECKSUMS_SHA256", anchor):
            with patch.object(data.urllib.request, "urlopen") as network:
                with self.assertRaisesRegex(ValueError, "checksum|SHA256"):
                    data.prepare_variant(self.release, self.raw, cache, variant, allow_download=True)
                network.assert_not_called()
        self.assertEqual(existing.read_bytes(), b"corrupt existing header")

    def test_prefetch_is_bounded_and_resumes_verified_downloads(self):
        anchor, content = self.make_official100()
        cache = self.root / "cache"
        variant = {"id": "official100", "source": "official100", "fs": 100, "filter": "minimal"}
        active = 0
        maximum = 0
        lock = threading.Lock()
        def response(request, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(.02)
            try:
                return io.BytesIO(content[request.full_url.removeprefix(data.OFFICIAL_BASE_URL)])
            finally:
                with lock:
                    active -= 1
        with patch.object(data, "OFFICIAL_CHECKSUMS_SHA256", anchor):
            with patch.object(data.urllib.request, "urlopen", side_effect=response):
                directory = data.prepare_variant(self.release, self.raw, cache, variant, allow_download=True)
            self.assertGreater(maximum, 1)
            self.assertLessEqual(maximum, 8)
            # Mimic interruption after data transfer but before cache preparation.
            (directory / "variant.json").unlink()
            with patch.object(data.urllib.request, "urlopen") as network:
                data.prepare_variant(self.release, self.raw, cache, variant, allow_download=True)
                network.assert_not_called()

    def test_failed_transfer_keeps_valid_files_and_resume_fetches_only_missing(self):
        anchor, content = self.make_official100()
        cache = self.root / "cache"
        variant = {"id": "official100", "source": "official100", "fs": 100, "filter": "minimal"}
        failed = "records100/00000/00001_lr.dat"
        def interrupted(request, **kwargs):
            name = request.full_url.removeprefix(data.OFFICIAL_BASE_URL)
            return io.BytesIO(b"broken response" if name == failed else content[name])
        with patch.object(data, "OFFICIAL_CHECKSUMS_SHA256", anchor):
            with patch.object(data.urllib.request, "urlopen", side_effect=interrupted):
                with self.assertRaisesRegex(ValueError, "Downloaded SHA256"):
                    data.prepare_variant(self.release, self.raw, cache, variant, allow_download=True)
            self.assertFalse((cache / "source100" / failed).exists())
            self.assertFalse(any(cache.glob("*/variant.json")))
            existing = [path for path in (cache / "source100").rglob("*.*") if path.is_file()]
            existing_stamps = {path: path.stat().st_mtime_ns for path in existing}
            def response(request, **kwargs):
                return io.BytesIO(content[request.full_url.removeprefix(data.OFFICIAL_BASE_URL)])
            with patch.object(data.urllib.request, "urlopen", side_effect=response) as network:
                data.prepare_variant(self.release, self.raw, cache, variant, allow_download=True)
                self.assertEqual(network.call_count, len(content) - len(existing))
            for path, stamp in existing_stamps.items():
                self.assertEqual(path.stat().st_mtime_ns, stamp)


if __name__ == "__main__":
    unittest.main()
