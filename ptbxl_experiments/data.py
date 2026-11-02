"""Portable PTB-XL receiving audit and immutable physical-signal caches.

Intent: keep one audited cohort while changing only a declared signal transform.
Status: fixture-tested independently of the producing pipeline/runtime. Next:
execute the full audit/preparation on the school machine. Blockers: bad source
bytes, identity mismatch or missing official100 files fail explicitly. Confidence:
technical integrity does not imply label correctness or clinical suitability.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, resample_poly, sosfiltfilt
import wfdb

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
LABEL_COLUMNS = tuple(f"y_{name}" for name in CLASSES)
SPLITS = ("train", "validation", "test")
ARTIFACTS = ("metadata.csv", "normalization.json", "pipeline.json", "records.csv",
             "train.csv", "validation.csv", "test.csv")
OFFICIAL_CHECKSUMS_SHA256 = "b7224b92b341511ec3ceb13dc6652079b2c36a06504bcb49506f157f51dc695d"
OFFICIAL_BASE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_bytes(path: Path, value: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: dict):
    _atomic_bytes(path, _canonical(value) + b"\n")


def _safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe path: {relative!r}")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or any(p in ("", ".") for p in relative.split("/")):
        raise ValueError(f"Unsafe path: {relative!r}")
    if root.is_symlink():
        raise ValueError(f"Symlink root path not accepted: {root}")
    root = root.resolve()
    candidate = root.joinpath(*parts.parts)
    current = root
    for part in parts.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink path not accepted: {relative}")
    resolved = candidate.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Path escapes root: {relative}")
    return candidate


def _hash_matches(path: Path, expected: str):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Invalid SHA256 for {path.name}")
    if _sha(path) != expected:
        raise ValueError(f"SHA256 checksum mismatch: {path}")


def _frame(path: Path, *, accepted: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    required = {"ecg_id", "patient_id", "split", "strat_fold", *LABEL_COLUMNS}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing manifest fields in {path.name}: {sorted(required - set(frame.columns))}")
    if not len(frame):
        raise ValueError(f"Empty manifest: {path.name}")
    for name in ("ecg_id", "patient_id", "strat_fold"):
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all() or (values <= 0).any():
            raise ValueError(f"Invalid {name} in {path.name}")
        if name == "patient_id":
            frame[name] = [str(int(v)) for v in values]
        else:
            frame[name] = values.astype(np.int64)
    if frame.ecg_id.duplicated().any():
        raise ValueError(f"Duplicate ECG ID in {path.name}")
    y = frame[list(LABEL_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError(f"Invalid binary labels: {path.name}")
    if accepted and (y.sum(axis=1) == 0).any():
        raise ValueError(f"Accepted manifest has zero-label ECGs: {path.name}")
    folds = frame.strat_fold.to_numpy()
    if not np.isin(folds, np.arange(1, 11)).all():
        raise ValueError(f"Invalid strat_fold in {path.name}")
    expected = np.where(folds <= 8, "train", np.where(folds == 9, "validation", "test"))
    if not np.array_equal(expected, frame.split.to_numpy()):
        raise ValueError(f"Official fold/split mismatch: {path.name}")
    return frame


def _aligned(actual: pd.DataFrame, expected: pd.DataFrame, columns, context: str):
    try:
        pd.testing.assert_frame_equal(actual[list(columns)].reset_index(drop=True),
                                      expected[list(columns)].reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
    except (AssertionError, KeyError) as exc:
        raise ValueError(f"Manifest alignment mismatch: {context}") from exc


def _patient_disjoint(frames: dict[str, pd.DataFrame]):
    seen = {}
    for split, frame in frames.items():
        for patient in set(frame.patient_id):
            if patient in seen and seen[patient] != split:
                raise ValueError(f"patient leakage across {seen[patient]} and {split}: {patient}")
            seen[patient] = split


def _valid_stats(stats: dict, count: int, *, dimensions: int):
    if stats.get("fit_split") != "train" or stats.get("train_records") != count:
        raise ValueError("Normalization must be fitted on this accepted train cohort only")
    mean, std = np.asarray(stats.get("mean"), dtype=float), np.asarray(stats.get("std"), dtype=float)
    if mean.shape != (dimensions,) or std.shape != (dimensions,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid normalization mean/std")
    return mean, std


def _array(path: Path, expected_hash: str, fs: int):
    _hash_matches(path, expected_hash)
    values = np.load(path, allow_pickle=False, mmap_mode="r")
    if values.dtype != np.dtype("float32") or values.shape != (12, 10 * fs):
        raise ValueError(f"Signal shape/dtype mismatch: {path.name}")
    if not np.isfinite(values).all():
        raise ValueError(f"Signal is not finite (NaN/Inf): {path.name}")
    return values


def verify_release(release_dir: Path, *, expected_fingerprint: str | None = None,
                   expected_audit_sha256: str | None = None, full: bool = True) -> dict:
    """Verify portable bytes/invariants, without recreating producer context.

    full=False still checks all metadata/manifests; it does not certify array
    bytes. The suite calls full=True before any model execution. Preparation
    independently checks every source array it reads.
    """
    release = Path(release_dir).resolve()
    audit_path = _safe_path(release, "audit.json")
    audit_sha = _sha(audit_path)
    if expected_audit_sha256 is not None and audit_sha != expected_audit_sha256:
        raise ValueError("audit.json SHA256 anchor mismatch")
    audit = _json(audit_path)
    if audit.get("ready") is not True or audit.get("checks_passed") is not True or audit.get("fixture") is not False or audit.get("errors") != []:
        raise ValueError("Release audit is not a ready, non-fixture release")
    if audit.get("cross_split_duplicates") != [] or audit.get("source_version") != "1.0.3":
        raise ValueError("Release version or cross-split duplicate audit invalid")
    fingerprint = audit.get("fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch("[a-f0-9]{64}", fingerprint):
        raise ValueError("Invalid source fingerprint")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError("Source fingerprint mismatch")
    hashes = audit.get("artifact_sha256", {})
    if set(hashes) != set(ARTIFACTS):
        raise ValueError("Audit must bind all seven release artifacts")
    for name in ARTIFACTS:
        _hash_matches(_safe_path(release, name), hashes[name])
    pipeline = _json(release / "pipeline.json")
    stats = _json(release / "normalization.json")
    for document in (pipeline, stats):
        if document.get("fingerprint") != fingerprint or document.get("classes") != list(CLASSES) or document.get("leads") != list(LEADS):
            raise ValueError("Release fingerprint/classes/leads mismatch")
    if pipeline.get("source_version") != "1.0.3" or pipeline.get("config", {}).get("target_fs") != 500 or pipeline.get("config", {}).get("preprocessing") != "minimal":
        raise ValueError("Expected immutable minimal500 release")
    for field, artifact in (("metadata_sha256", "metadata.csv"), ("records_sha256", "records.csv"), ("normalization_sha256", "normalization.json")):
        if pipeline.get(field) != hashes[artifact]:
            raise ValueError(f"Pipeline checksum binding mismatch: {field}")
    if stats.get("records_sha256") != hashes["records.csv"] or stats.get("units") != "mV" or stats.get("ddof") != 0:
        raise ValueError("Original normalization provenance mismatch")
    records, metadata = _frame(release / "records.csv"), _frame(release / "metadata.csv")
    if not set(("status", "npy_path", "signal_sha256", "fs", "n_samples", "units")).issubset(records.columns):
        raise ValueError("Records missing signal/status fields")
    if not records.status.isin(("accepted", "no_label", "rejected")).all():
        raise ValueError("Unknown release record status")
    _aligned(records, metadata, metadata.columns, "records versus metadata")
    source_count = len(records)
    if source_count != audit.get("source_records") or source_count != pipeline.get("source_records"):
        raise ValueError("Source record count mismatch")
    accepted = records[records.status == "accepted"]
    if len(accepted) != audit.get("accepted_records") or accepted.empty:
        raise ValueError("Accepted record count mismatch")
    if accepted.npy_path.duplicated().any():
        raise ValueError("Multiple ECG IDs reference one signal path")
    if (accepted[list(LABEL_COLUMNS)].sum(axis=1) == 0).any():
        raise ValueError("Accepted cohort contains zero-label ECG")
    if not (accepted.fs.eq(500).all() and accepted.n_samples.eq(5000).all() and accepted.units.eq("mV").all()):
        raise ValueError("Accepted signal fs/length/units mismatch")
    frames = {}
    for split in SPLITS:
        frame = _frame(release / f"{split}.csv", accepted=True)
        _aligned(frame, accepted[accepted.split == split], records.columns, split)
        if len(frame) != audit.get("split_counts", {}).get(split):
            raise ValueError(f"Split count mismatch: {split}")
        frames[split] = frame
    _patient_disjoint({split: metadata[metadata.split == split] for split in SPLITS})
    _valid_stats(stats, len(frames["train"]), dimensions=12)
    if stats.get("samples_per_lead") != len(frames["train"]) * 5000:
        raise ValueError("Normalization sample count mismatch")
    duplicate_splits = accepted.groupby("signal_sha256").split.nunique()
    if (duplicate_splits > 1).any():
        raise ValueError("Exact waveform checksum duplicate across splits")
    for index, row in enumerate(accepted.itertuples(index=False), 1):
        path = _safe_path(release, row.npy_path)
        if not isinstance(row.signal_sha256, str) or not re.fullmatch("[a-f0-9]{64}", row.signal_sha256):
            raise ValueError("Invalid signal SHA256")
        if full:
            values = _array(path, row.signal_sha256, 500)
            if (np.ptp(values, axis=1) == 0).any():
                raise ValueError(f"Accepted constant lead: ECG {row.ecg_id}")
        if full and (index % 1000 == 0 or index == len(accepted)):
            print(f"Receiving audit: {index}/{len(accepted)} arrays verified", flush=True)
    return {"source_fingerprint": fingerprint, "accepted_records": len(accepted),
            "source_records": source_count, "split_counts": {s: len(f) for s, f in frames.items()},
            "audit_sha256": audit_sha, "classes": list(CLASSES), "leads": list(LEADS),
            "artifact_sha256": hashes, "full_array_verification": bool(full)}


def _runtime():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "wfdb": wfdb.__version__}


def _variant(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {"id", "source", "fs", "filter"}:
        raise ValueError("Variant requires id, source, fs and filter only")
    if not isinstance(value["id"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)*", value["id"]):
        raise ValueError("Unsafe variant ID")
    if value["source"] not in ("official100", "release500") or type(value["fs"]) is not int or value["fs"] not in (100, 200, 500) or value["filter"] not in ("minimal", "bandpass_0.5_40"):
        raise ValueError("Unsupported signal variant")
    if value["source"] == "official100" and (value["fs"] != 100 or value["filter"] != "minimal"):
        raise ValueError("official100 reference must remain native100/minimal")
    if value["filter"] == "bandpass_0.5_40" and value["fs"] != 500:
        raise ValueError("This protocol declares bandpass at500Hz only")
    return dict(value)


def _catalog(raw: Path) -> dict:
    path = _safe_path(raw, "SHA256SUMS.txt")
    if _sha(path) != OFFICIAL_CHECKSUMS_SHA256:
        raise ValueError("Untrusted official PTB-XL1.0.3 checksum catalog")
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not re.fullmatch(r"[a-fA-F0-9]{64}", fields[0]):
            raise ValueError("Invalid official checksum catalog syntax")
        relative = fields[1].lstrip("*").removeprefix("./")
        _safe_path(raw, relative)
        if relative in result:
            raise ValueError("Duplicate official checksum catalog path")
        result[relative] = fields[0].lower()
    return result


def _official_file(raw: Path, local: Path, relative: str, catalog: dict, allow_download: bool):
    if relative not in catalog:
        raise ValueError(f"Official catalog lacks {relative}")
    target = _safe_path(local, relative)
    if target.exists():
        _hash_matches(target, catalog[relative])
        return target
    original = _safe_path(raw, relative)
    if original.exists():
        _hash_matches(original, catalog[relative])
        _atomic_bytes(target, original.read_bytes())
        return target
    if not allow_download:
        raise FileNotFoundError(f"Missing official100 source: {relative}; enable --allow-download on school machine")
    # Individual WFDB files are small; retries never replace corrupt existing data.
    for attempt in range(3):
        try:
            request = urllib.request.Request(OFFICIAL_BASE_URL + relative, headers={"User-Agent": "PTBXL-school-benchmark/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                value = response.read(8 * 1024 * 1024 + 1)
            if len(value) > 8 * 1024 * 1024:
                raise ValueError(f"Unexpected official100 file size: {relative}")
            if hashlib.sha256(value).hexdigest() != catalog[relative]:
                raise ValueError(f"Downloaded SHA256 checksum mismatch: {relative}")
            _atomic_bytes(target, value)
            return target
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable official download state")


def _official_signal(row, raw, local, catalog, allow_download):
    relative = row.filename_lr
    if not isinstance(relative, str) or not re.fullmatch(r"records100/[0-9]{5}/[0-9]{5}_lr", relative):
        raise ValueError("Invalid official100 filename in accepted manifest")
    for extension in (".hea", ".dat"):
        _official_file(raw, local, relative + extension, catalog, allow_download)
    record_path = _safe_path(local, relative)
    header = wfdb.rdheader(str(record_path))
    # Official headers use AVR/AVL/AVF; ignore case while preserving lead order.
    if (header.fs != 100 or header.sig_len != 1000 or header.n_sig != 12 or
            [name.upper() for name in header.sig_name] != [name.upper() for name in LEADS] or header.units != ["mV"] * 12 or
            set(header.file_name) != {record_path.name + ".dat"}):
        raise ValueError(f"Official100 header contract mismatch: {relative}")
    record = wfdb.rdrecord(str(record_path), physical=True)
    values = np.asarray(record.p_signal.T, dtype=np.float32)
    if values.shape != (12, 1000) or not np.isfinite(values).all():
        raise ValueError(f"Official100 invalid signal: {relative}")
    return values


def _prefetch_official(frames, raw, local, catalog, allow_download):
    """Bound network concurrency and pending tasks; verify before decoding.

    Scan all existing files first so a corrupt local file or missing download
    permission fails before scheduling any network calls. Completed downloads
    are independently atomic/checksummed and reused after interruption.
    """
    relatives = []
    for frame in frames.values():
        for name in frame.filename_lr:
            if not isinstance(name, str) or not re.fullmatch(r"records100/[0-9]{5}/[0-9]{5}_lr", name):
                raise ValueError("Invalid official100 filename in accepted manifest")
            relatives.extend((name + ".hea", name + ".dat"))
    if len(set(relatives)) != len(relatives):
        raise ValueError("Multiple accepted ECG IDs reference one official100 record")
    pending = []
    for index, relative in enumerate(relatives, 1):
        if relative not in catalog:
            raise ValueError(f"Official catalog lacks {relative}")
        target, original = _safe_path(local, relative), _safe_path(raw, relative)
        if target.exists():
            _hash_matches(target, catalog[relative])
        else:
            if original.exists():
                _hash_matches(original, catalog[relative])
            elif not allow_download:
                raise FileNotFoundError(f"Missing official100 source: {relative}; enable --allow-download on school machine")
            pending.append(relative)
        if index % 1000 == 0:
            print(f"Official100 source inventory: {index}/{len(relatives)} files checked", flush=True)
    if not pending:
        print(f"Official100 source: all {len(relatives)} files already verified", flush=True)
        return
    print(f"Official100 source: obtain {len(pending)} files using at most8 concurrent transfers", flush=True)
    completed, last_progress = 0, time.monotonic()
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="ptbxl-source") as pool:
        # Only128 futures exist at once, rather than retaining42k requests.
        for start in range(0, len(pending), 128):
            futures = [pool.submit(_official_file, raw, local, relative, catalog, allow_download)
                       for relative in pending[start:start + 128]]
            try:
                for future in as_completed(futures):
                    future.result()
                    completed += 1
                    if completed % 250 == 0 or completed == len(pending) or time.monotonic() - last_progress >= 30:
                        print(f"Official100 source transfer: {completed}/{len(pending)} verified", flush=True)
                        last_progress = time.monotonic()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise


def _metadata(directory: Path) -> dict:
    metadata = _json(_safe_path(directory, "variant.json"))
    seal = metadata.get("metadata_sha256")
    content = {k: v for k, v in metadata.items() if k != "metadata_sha256"}
    if seal != _digest(content):
        raise ValueError("Prepared metadata checksum mismatch")
    if metadata.get("schema_version") != 1 or metadata.get("ready") is not True:
        raise ValueError("Prepared cache is incomplete or unsupported")
    variant = _variant(metadata.get("variant"))
    provenance = metadata.get("provenance", {})
    if _digest(provenance) != metadata.get("fingerprint") or provenance.get("variant") != variant or provenance.get("source_fingerprint") != metadata.get("source_fingerprint"):
        raise ValueError("Prepared identity fingerprint mismatch")
    if provenance.get("implementation_sha256") != _sha(Path(__file__)) or provenance.get("runtime") != _runtime():
        raise ValueError("Prepared cache implementation/runtime changed; prepare a new cache")
    if metadata.get("classes") != list(CLASSES) or metadata.get("leads") != list(LEADS) or metadata.get("fs") != variant["fs"]:
        raise ValueError("Prepared classes/leads/fs mismatch")
    splits = metadata.get("splits", {})
    if set(splits) != set(SPLITS):
        raise ValueError("Prepared split metadata missing")
    for split in SPLITS:
        item = splits[split]
        if item.get("signals") != f"{split}.npy" or item.get("manifest") != f"{split}.csv" or type(item.get("count")) is not int or item["count"] <= 0:
            raise ValueError("Prepared split filename/count invalid")
    normalizers = metadata.get("normalization", {})
    if set(normalizers) != {"global", "per_lead"}:
        raise ValueError("Prepared normalizers missing")
    _valid_stats(normalizers["global"], splits["train"]["count"], dimensions=1)
    _valid_stats(normalizers["per_lead"], splits["train"]["count"], dimensions=12)
    return metadata


def _prepared_split(directory, split, metadata, *, full):
    item = metadata["splits"][split]
    manifest_path = _safe_path(directory, item["manifest"])
    _hash_matches(manifest_path, item["manifest_sha256"])
    frame = _frame(manifest_path, accepted=True)
    if len(frame) != item["count"] or not frame.split.eq(split).all():
        raise ValueError(f"Prepared manifest split/count mismatch: {split}")
    signal_path = _safe_path(directory, item["signals"])
    if full:
        _hash_matches(signal_path, item["signals_sha256"])
    values = np.load(signal_path, mmap_mode="r", allow_pickle=False)
    if values.shape != (len(frame), 12, 10 * metadata["fs"]) or values.dtype != np.dtype("float32"):
        raise ValueError(f"Prepared signal shape/dtype mismatch: {split}")
    if full:
        for start in range(0, len(frame), 64):
            if not np.isfinite(values[start:start + 64]).all():
                raise ValueError(f"Prepared signal is not finite: {split}")
    return values, frame


def verify_prepared(directory: Path) -> dict:
    """Full cache check; reuse does not trust ready flags or file sizes alone."""
    directory = Path(directory).resolve()
    metadata = _metadata(directory)
    frames = {}
    for split in SPLITS:
        _, frames[split] = _prepared_split(directory, split, metadata, full=True)
        print(f"Prepared cache verified: {metadata['variant']['id']} / {split}", flush=True)
    joined = pd.concat(frames.values(), ignore_index=True)
    if joined.ecg_id.duplicated().any():
        raise ValueError("Prepared ECG ID duplicated across splits")
    _patient_disjoint(frames)
    return metadata


def load_prepared(directory: Path, split: str, normalization: str) -> tuple:
    """Read-only memmap; caller must full-verify once before starting training."""
    if split not in SPLITS or normalization not in ("global", "per_lead"):
        raise ValueError("Unknown split or normalization")
    directory = Path(directory).resolve()
    metadata = _metadata(directory)
    values, frame = _prepared_split(directory, split, metadata, full=False)
    mean, std = _valid_stats(metadata["normalization"][normalization],
                             metadata["splits"]["train"]["count"],
                             dimensions=1 if normalization == "global" else 12)
    mean = np.broadcast_to(mean.reshape(-1, 1), (12, 1)).copy()
    std = np.broadcast_to(std.reshape(-1, 1), (12, 1)).copy()
    return values, frame, mean, std, metadata


def _normalization(values: np.ndarray) -> dict:
    # Merge each full-record population using Chan/Welford moments in float64.
    count, mean, m2 = 0, np.zeros(12, dtype=np.float64), np.zeros(12, dtype=np.float64)
    for index in range(len(values)):
        block = np.asarray(values[index], dtype=np.float64)
        n = block.shape[1]
        bm = block.mean(axis=1)
        b2 = ((block - bm[:, None]) ** 2).sum(axis=1)
        delta = bm - mean
        total = count + n
        m2 += b2 + delta ** 2 * (count * n / total)
        mean += delta * n / total
        count = total
    variance = m2 / count
    global_mean = float(mean.mean())
    global_variance = float((variance + (mean - global_mean) ** 2).mean())
    result = {
        "global": {"mean": [global_mean], "std": [math.sqrt(global_variance)]},
        "per_lead": {"mean": mean.tolist(), "std": np.sqrt(variance).tolist()},
    }
    for value in result.values():
        value.update(fit_split="train", train_records=len(values), ddof=0,
                     samples_per_lead=count, units="mV")
    _valid_stats(result["global"], len(values), dimensions=1)
    _valid_stats(result["per_lead"], len(values), dimensions=12)
    return result


def prepare_variant(release_dir: Path, raw_dir: Path, cache_root: Path,
                    variant: dict, *, allow_download: bool = False) -> Path:
    """Prepare one shared physical cache; no fitting or model use of test labels.

    Each completed split has an atomic resume record. A killed partial split is
    rebuilt, while already verified splits are reused. variant.json is written
    last and seals normalization plus all split checksums. Existing corrupt ready
    caches fail rather than being overwritten; change work directory to rebuild.
    """
    release, raw, cache = (Path(p).resolve() for p in (release_dir, raw_dir, cache_root))
    for source in (release, raw):
        if cache == source or source in cache.parents or cache in source.parents:
            raise ValueError("Cache path must be outside and separate from source/release")
    variant = _variant(variant)
    audit = verify_release(release, full=False)
    catalog = _catalog(raw) if variant["source"] == "official100" else None
    provenance = {"schema_version": 1, "source_fingerprint": audit["source_fingerprint"],
                  "source_audit_sha256": audit["audit_sha256"],
                  "source_artifact_sha256": audit["artifact_sha256"], "variant": variant,
                  "runtime": _runtime(), "implementation_sha256": _sha(Path(__file__)),
                  "transform": {"duration_seconds": 10, "input_units": "mV", "output_dtype": "float32",
                                "filter_design_order": 4 if variant["filter"] != "minimal" else None,
                                "filter_application": "full-record SOS forward-backward" if variant["filter"] != "minimal" else "none",
                                "resampler": "scipy.signal.resample_poly/kaiser5/constant-pad" if variant["source"] == "release500" and variant["fs"] != 500 else "none"},
                  "official_catalog_sha256": OFFICIAL_CHECKSUMS_SHA256 if catalog is not None else None}
    fingerprint = _digest(provenance)
    cache.mkdir(parents=True, exist_ok=True)
    directory = _safe_path(cache, f"{variant['id']}-{fingerprint[:16]}")
    if (directory / "variant.json").exists():
        existing = verify_prepared(directory)
        if existing["fingerprint"] != fingerprint:
            raise ValueError("Prepared cache fingerprint collision/mismatch")
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    marker = _safe_path(directory, ".prepare.json")
    if marker.exists():
        if _json(marker) != provenance:
            raise ValueError("Incomplete cache belongs to another source/runtime/transform")
    else:
        _atomic_json(marker, provenance)
    frames = {split: _frame(release / f"{split}.csv", accepted=True) for split in SPLITS}
    if catalog is not None:
        _prefetch_official(frames, raw, _safe_path(cache, "source100"), catalog, allow_download)
    sos = butter(4, [0.5, 40], btype="bandpass", fs=500, output="sos") if variant["filter"] == "bandpass_0.5_40" else None
    split_metadata = {}
    for split in SPLITS:
        frame = frames[split]
        manifest_bytes = frame.to_csv(index=False).encode("utf-8")
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_path = _safe_path(directory, f"{split}.csv")
        array_path = _safe_path(directory, f"{split}.npy")
        done_path = _safe_path(directory, f".{split}.complete.json")
        if done_path.exists():
            item = _json(done_path)
            if item.get("fingerprint") != fingerprint or item.get("manifest_sha256") != manifest_hash or item.get("count") != len(frame):
                raise ValueError(f"Partial prepared split identity mismatch: {split}")
            _hash_matches(manifest_path, manifest_hash)
            _hash_matches(array_path, item["signals_sha256"])
            check = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if check.shape != (len(frame), 12, variant["fs"] * 10) or check.dtype != np.dtype("float32"):
                raise ValueError("Partial prepared split shape mismatch")
            del check
            split_metadata[split] = {k: v for k, v in item.items() if k != "fingerprint"}
            print(f"Preparation resume: {variant['id']} / {split} verified and reused", flush=True)
            continue
        temporary_path = _safe_path(directory, f".{split}.building.npy")
        output = np.lib.format.open_memmap(temporary_path, mode="w+", dtype=np.float32,
                                           shape=(len(frame), 12, variant["fs"] * 10))
        last_progress = time.monotonic()
        try:
            for index, row in enumerate(frame.itertuples(index=False)):
                if variant["source"] == "official100":
                    values = _official_signal(row, raw, _safe_path(cache, "source100"), catalog, False)
                else:
                    values = np.asarray(_array(_safe_path(release, row.npy_path), row.signal_sha256, 500), dtype=np.float64)
                    if sos is not None:
                        values = sosfiltfilt(sos, values, axis=1)
                    if variant["fs"] != 500:
                        divisor = math.gcd(variant["fs"], 500)
                        values = resample_poly(values, variant["fs"] // divisor, 500 // divisor,
                                               axis=1, window=("kaiser", 5.0), padtype="constant")
                    values = np.asarray(values, dtype=np.float32)
                if values.shape != (12, variant["fs"] * 10) or not np.isfinite(values).all():
                    raise ValueError(f"Transformed signal shape/finiteness invalid: ECG {row.ecg_id}")
                output[index] = values
                if (index + 1) % 250 == 0 or index + 1 == len(frame) or time.monotonic() - last_progress >= 30:
                    print(f"Prepare {variant['id']} / {split}: {index + 1}/{len(frame)}", flush=True)
                    last_progress = time.monotonic()
            output.flush()
        finally:
            # Close the mmap before replace; Windows refuses moving open maps.
            del output
        with temporary_path.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, array_path)
        _atomic_bytes(manifest_path, manifest_bytes)
        item = {"signals": array_path.name, "manifest": manifest_path.name, "count": len(frame),
                "signals_sha256": _sha(array_path), "manifest_sha256": manifest_hash}
        _atomic_json(done_path, {**item, "fingerprint": fingerprint})
        split_metadata[split] = item
    train = np.load(directory / "train.npy", mmap_mode="r", allow_pickle=False)
    normalizers = _normalization(train)
    del train
    metadata = {"schema_version": 1, "ready": True, "fingerprint": fingerprint,
                "source_fingerprint": audit["source_fingerprint"], "variant": variant,
                "fs": variant["fs"], "classes": list(CLASSES), "leads": list(LEADS),
                "normalization": normalizers, "splits": split_metadata, "provenance": provenance,
                "manifest_signal_fields": "fs/n_samples/npy_path/signal_sha256 preserve source500 release provenance; use variant.json fs/splits for this prepared cache"}
    metadata["metadata_sha256"] = _digest(metadata)
    _atomic_json(_safe_path(directory, "variant.json"), metadata)
    verify_prepared(directory)
    return directory
