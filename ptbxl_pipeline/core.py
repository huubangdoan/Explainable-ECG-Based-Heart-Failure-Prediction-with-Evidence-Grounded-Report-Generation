"""A reproducible, streaming PTB-XL diagnostic-superclass preparation pipeline.

Intent: preserve physical ECGs and the official patient folds. Status: implemented
with regression tests; full release readiness is decided by audit_release. Next:
consume immutable arrays through ECGDataset. Blockers: missing or corrupt source
fails explicitly. Confidence: a passed audit certifies this data contract, not a
clinical diagnosis or clinical suitability of any future model.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import tempfile
from typing import Iterator

import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, resample_poly, sosfiltfilt
import wfdb

from .acquire import OFFICIAL_CHECKSUMS_SHA256

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
LABEL_COLUMNS = tuple(f"y_{name}" for name in CLASSES)
SPLITS = ("train", "validation", "test")
VERSION = "1.0.3"


@dataclass(frozen=True)
class PipelineConfig:
    raw_dir: Path
    output_dir: Path
    target_fs: int = 500
    preprocessing: str = "minimal"
    expected_records: int | None = 21799

    def __post_init__(self):
        object.__setattr__(self, "raw_dir", Path(self.raw_dir).resolve())
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())
        if self.target_fs not in (100, 200, 250, 500):
            raise ValueError("target_fs phải là 100, 200, 250 hoặc 500 Hz")
        if self.preprocessing not in ("minimal", "bandpass_0.5_40"):
            raise ValueError("preprocessing phải là minimal hoặc bandpass_0.5_40")
        if self.expected_records is not None and self.expected_records <= 0:
            raise ValueError("expected_records phải dương hoặc None cho fixture")
        if self.output_dir == self.raw_dir or self.raw_dir in self.output_dir.parents:
            raise ValueError("output_dir phải nằm ngoài raw_dir để giữ nguồn nguyên vẹn")


def _sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _json_bytes(value) -> bytes:
    return (json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value):
    _atomic_bytes(path, _json_bytes(value))


def _atomic_csv(path: Path, frame: pd.DataFrame):
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_array(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_path(root: Path, relative: str) -> Path:
    relative = str(relative).replace("\\", "/")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or ":" in relative:
        raise ValueError(f"Đường dẫn nguồn không an toàn: {relative}")
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Đường dẫn vượt thư mục cho phép: {relative}")
    return path


def _checksum_catalog(raw_dir: Path) -> dict[str, str]:
    catalog = {}
    path = raw_dir / "SHA256SUMS.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing source checksum: {path}; chạy notebook 00")
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if not fields:
            continue
        if len(fields) != 2 or len(fields[0]) != 64 or any(c not in "0123456789abcdefABCDEF" for c in fields[0]):
            raise ValueError("SHA256SUMS.txt không đúng định dạng")
        relative = fields[1].lstrip("*")
        if relative.startswith("./"):
            relative = relative[2:]
        _safe_path(raw_dir, relative)
        if relative in catalog:
            raise ValueError(f"SHA256SUMS có đường dẫn trùng: {relative}")
        catalog[relative] = fields[0].lower()
    return catalog


def _verified_hash(raw_dir: Path, relative: str, catalog: dict) -> str:
    if relative not in catalog:
        raise ValueError(f"source thiếu checksum cho {relative}")
    path = _safe_path(raw_dir, relative)
    if not path.is_file():
        raise FileNotFoundError(f"Missing source: {path}")
    actual = _sha(path)
    if actual != catalog[relative]:
        raise ValueError(f"SHA256 checksum source không khớp: {relative}; chạy lại notebook 00")
    return actual


def _validated_source_catalog(config: PipelineConfig):
    catalog = _checksum_catalog(config.raw_dir)
    catalog_hash = _sha(config.raw_dir / "SHA256SUMS.txt")
    if config.expected_records is not None and catalog_hash != OFFICIAL_CHECKSUMS_SHA256:
        raise ValueError("SHA256SUMS.txt không khớp manifest official/chính thức PTB-XL v1.0.3")
    marker = config.raw_dir / "source_verification.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Missing source provenance: {marker}; chạy notebook 00")
    # The acquisition report is retained, but checksums below establish integrity.
    source_report = json.loads(marker.read_text(encoding="utf-8"))
    if source_report.get("verified") is not True or source_report.get("source_version") != VERSION:
        raise ValueError("source provenance chưa xác minh hoặc sai phiên bản; chạy notebook 00")
    if config.expected_records is not None and source_report.get("checksum_manifest_sha256") != catalog_hash:
        raise ValueError("source provenance marker không khớp checksum manifest hiện tại; chạy notebook 00")
    return catalog, catalog_hash


def _context(config: PipelineConfig, *, create=False):
    catalog, catalog_hash = _validated_source_catalog(config)
    metadata_hashes = {name: _verified_hash(config.raw_dir, name, catalog)
                       for name in ("ptbxl_database.csv", "scp_statements.csv")}
    payload = {
        "source_version": VERSION, "source_metadata_sha256": metadata_hashes,
        "checksum_catalog_sha256": catalog_hash,
        "implementation_sha256": _sha(Path(__file__)),
        "config": {"target_fs": config.target_fs,
                   "preprocessing": config.preprocessing, "expected_records": config.expected_records},
        "classes": list(CLASSES), "leads": list(LEADS),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scipy": scipy.__version__, "wfdb": wfdb.__version__},
    }
    fingerprint = hashlib.sha256(_json_bytes(payload)).hexdigest()
    identity_path = config.output_dir / "pipeline.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if identity.get("fingerprint") != fingerprint:
            raise ValueError("Pipeline identity/config/source/code đã thay đổi. Chọn output_dir mới; không tái sử dụng đầu ra cũ.")
    elif create:
        identity = {**payload, "fingerprint": fingerprint,
                    "source_root_at_creation": str(config.raw_dir),
                    "created_at": datetime.now(timezone.utc).isoformat()}
        _atomic_json(identity_path, identity)
    else:
        raise FileNotFoundError("Thiếu pipeline.json; chạy build_manifest / notebook 01 trước")
    return identity, catalog


def _read_frame(path: Path):
    return pd.read_csv(path, dtype={"patient_id": "string", "filename_hr": "string", "scp_codes": "string"}, keep_default_na=False)


def _patient_check(frame: pd.DataFrame):
    by_patient = frame.groupby("patient_id", dropna=False)["split"].nunique()
    invalid = by_patient[by_patient > 1]
    if len(invalid):
        raise ValueError(f"patient leakage: {len(invalid)} bệnh nhân có mặt ở nhiều tập: {invalid.index[:5].tolist()}")


def build_manifest(config: PipelineConfig) -> pd.DataFrame:
    """Map existing annotation keys, retain ambiguity, and use official folds."""
    catalog, _ = _validated_source_catalog(config)
    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        _verified_hash(config.raw_dir, name, catalog)
    frame = _read_frame(config.raw_dir / "ptbxl_database.csv")
    required = {"ecg_id", "patient_id", "strat_fold", "scp_codes", "filename_hr"}
    if not required.issubset(frame):
        raise ValueError(f"Metadata thiếu cột: {sorted(required - set(frame))}")
    if config.expected_records is not None and len(frame) != config.expected_records:
        raise ValueError(f"Sai source count: {len(frame)} thay vì {config.expected_records}")
    if frame.empty or frame.ecg_id.duplicated().any() or frame.filename_hr.duplicated().any():
        raise ValueError("Metadata rỗng hoặc có ecg_id/filename_hr trùng")
    for column in ("ecg_id", "strat_fold"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or (values != values.astype(int)).any():
            raise ValueError(f"{column} phải là số nguyên hữu hạn")
        frame[column] = values.astype(int)
    if (frame.ecg_id <= 0).any() or not frame.strat_fold.between(1, 10).all():
        raise ValueError("ecg_id phải dương, strat_fold phải thuộc 1..10")
    if frame.patient_id.str.strip().eq("").any() or frame.patient_id.str.lower().isin(["nan", "none"]).any():
        raise ValueError("Thiếu patient_id; không thể đảm bảo split bệnh nhân")
    frame["split"] = frame.strat_fold.map(lambda fold: "train" if fold <= 8 else "validation" if fold == 9 else "test")
    _patient_check(frame)
    scp = pd.read_csv(config.raw_dir / "scp_statements.csv", index_col=0)
    if not {"diagnostic", "diagnostic_class"}.issubset(scp):
        raise ValueError("scp_statements thiếu diagnostic/diagnostic_class")
    if scp.index.duplicated().any():
        raise ValueError("scp_statements có mã trùng")
    diagnostic = scp[pd.to_numeric(scp.diagnostic, errors="coerce") == 1]
    mapping = diagnostic.diagnostic_class.to_dict()
    if not set(mapping.values()).issubset(CLASSES):
        raise ValueError("Có diagnostic_class không thuộc 5 superclass chuẩn")
    annotations, classes, subclasses, unknown = [], [], [], []
    for row in frame.itertuples(index=False):
        try:
            parsed = ast.literal_eval(row.scp_codes)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"scp_codes không hợp lệ ở ECG {row.ecg_id}") from exc
        if not isinstance(parsed, dict) or not all(isinstance(k, str) and isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 100 for k, v in parsed.items()):
            raise ValueError(f"scp_codes phải là dictionary mã→likelihood 0..100: ECG {row.ecg_id}")
        names = {mapping[key] for key in parsed if key in mapping}
        annotations.append(json.dumps(parsed, sort_keys=True, ensure_ascii=False))
        classes.append(names)
        sub = sorted({str(diagnostic.loc[key, "diagnostic_subclass"]) for key in parsed if key in mapping and "diagnostic_subclass" in diagnostic and pd.notna(diagnostic.loc[key, "diagnostic_subclass"])})
        subclasses.append(json.dumps(sub))
        unknown.append(json.dumps(sorted(set(parsed) - set(scp.index))))
        if not str(row.filename_hr).startswith("records500/"):
            raise ValueError(f"filename_hr không thuộc records500: {row.filename_hr}")
        for extension in (".hea", ".dat"):
            relative = str(row.filename_hr) + extension
            if relative not in catalog or not _safe_path(config.raw_dir, relative).is_file():
                raise FileNotFoundError(f"Missing source hoặc checksum: {relative}")
    frame["scp_codes_json"] = annotations
    frame["diagnostic_subclasses_json"] = subclasses
    frame["unknown_scp_codes_json"] = unknown
    for name in CLASSES:
        frame[f"y_{name}"] = [int(name in labels) for labels in classes]
    frame["has_label"] = [int(bool(labels)) for labels in classes]
    frame["norm_with_abnormal"] = [int("NORM" in labels and len(labels) > 1) for labels in classes]
    frame["source_metadata_sha256"] = catalog["ptbxl_database.csv"]
    identity, _ = _context(config, create=True)
    _atomic_csv(config.output_dir / "metadata.csv", frame)
    identity["metadata_sha256"] = _sha(config.output_dir / "metadata.csv")
    identity["source_records"] = len(frame)
    _atomic_json(config.output_dir / "pipeline.json", identity)
    return frame


def _metadata(config, identity):
    path = config.output_dir / "metadata.csv"
    if not path.is_file() or _sha(path) != identity.get("metadata_sha256"):
        raise ValueError("metadata.csv thiếu hoặc đã bị sửa; chạy lại build_manifest")
    return _read_frame(path)


def _source_hashes(config, row, catalog):
    return {extension[1:]: _verified_hash(config.raw_dir, str(row["filename_hr"]) + extension, catalog)
            for extension in (".hea", ".dat")}


def _signal(config, row):
    signal, header = wfdb.rdsamp(str(_safe_path(config.raw_dir, str(row["filename_hr"]))))
    if header.get("fs") != 500:
        raise ValueError(f"source_fs:{header.get('fs')}")
    if signal.shape != (5000, 12):
        raise ValueError(f"shape:{signal.shape};expected:(5000,12)")
    names = [str(x).upper() for x in header.get("sig_name", [])]
    if len(names) != 12 or len(set(names)) != 12 or set(names) != {x.upper() for x in LEADS}:
        raise ValueError("missing_or_duplicate_lead")
    units = header.get("units", [])
    if len(units) != 12:
        raise ValueError("missing_units")
    factors = {"mv": 1.0, "uv": 0.001, "µv": 0.001, "μv": 0.001, "v": 1000.0}
    order = [names.index(lead.upper()) for lead in LEADS]
    values = np.asarray(signal[:, order], dtype=np.float64)
    for out_index, source_index in enumerate(order):
        unit = str(units[source_index]).strip().lower()
        if unit not in factors:
            raise ValueError(f"unsupported_units:{unit}")
        values[:, out_index] *= factors[unit]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite_source_signal")
    if np.any(np.ptp(values, axis=0) == 0):
        raise ValueError("constant_lead")
    if config.preprocessing == "bandpass_0.5_40":
        sos = butter(4, [0.5, 40], btype="bandpass", fs=500, output="sos")
        values = sosfiltfilt(sos, values, axis=0)
    if config.target_fs != 500:
        divisor = math.gcd(config.target_fs, 500)
        values = resample_poly(values, up=config.target_fs // divisor, down=500 // divisor, axis=0)
    values = np.asarray(values.T, dtype=np.float32, order="C")
    if values.shape != (12, config.target_fs * 10) or not np.isfinite(values).all():
        raise ValueError("invalid_output_shape_or_nonfinite")
    if np.any(np.ptp(values, axis=1) == 0):
        raise ValueError("constant_output_lead")
    return values


def _valid_sidecar(config, path, fingerprint, source_hashes):
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        if sidecar.get("fingerprint") != fingerprint or sidecar.get("source_hashes") != source_hashes:
            return None
        result = sidecar["result"]
        if result["status"] == "accepted":
            output = _safe_path(config.output_dir, result["npy_path"])
            if _sha(output) != result["signal_sha256"]:
                return None
            values = np.load(output, allow_pickle=False)
            if values.dtype != np.float32 or values.shape != (12, config.target_fs * 10) or not np.isfinite(values).all():
                return None
        elif result["status"] not in ("rejected", "no_label"):
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def process_signals(config: PipelineConfig) -> pd.DataFrame:
    """Verify every raw record; atomically save each ECG and recoverable metadata."""
    identity, catalog = _context(config)
    frame = _metadata(config, identity)
    rows = []
    for index, row in enumerate(frame.to_dict(orient="records"), 1):
        source_hashes = _source_hashes(config, row, catalog)  # corruption is fatal, not a QC exclusion
        sidecar_path = config.output_dir / "sidecars" / f"{int(row['ecg_id']):05d}.json"
        result = _valid_sidecar(config, sidecar_path, identity["fingerprint"], source_hashes) if sidecar_path.exists() else None
        if result is None:
            result = {"ecg_id": int(row["ecg_id"]), "status": "no_label", "reason": "no_diagnostic_superclass",
                      "npy_path": "", "signal_sha256": "", "fs": config.target_fs,
                      "n_samples": config.target_fs * 10, "units": "mV", "max_abs_mv": None}
            if row["has_label"]:
                try:
                    values = _signal(config, row)
                except (ValueError, IndexError) as exc:
                    result.update(status="rejected", reason=str(exc))
                else:
                    relative = f"signals/{int(row['ecg_id']):05d}.npy"
                    output = config.output_dir / relative
                    _atomic_array(output, values)
                    result.update(status="accepted", reason="", npy_path=relative,
                                  signal_sha256=_sha(output), max_abs_mv=float(np.max(np.abs(values))))
            _atomic_json(sidecar_path, {"fingerprint": identity["fingerprint"], "source_hashes": source_hashes, "result": result})
        rows.append({**row, **result})
        if index % 500 == 0 or index == len(frame):
            print(f"ECG: {index:,}/{len(frame):,}; metadata được lưu riêng cho từng bản ghi.", flush=True)
    records = pd.DataFrame(rows)
    _atomic_csv(config.output_dir / "records.csv", records)
    # A rerun invalidates derived release markers; all later stages recompute from records.
    identity["records_sha256"] = _sha(config.output_dir / "records.csv")
    _atomic_json(config.output_dir / "pipeline.json", identity)
    return records


def _records(config, identity):
    path = config.output_dir / "records.csv"
    if not path.is_file() or _sha(path) != identity.get("records_sha256"):
        raise ValueError("records.csv thiếu hoặc đã bị sửa; chạy process_signals")
    return _read_frame(path)


def _accepted_array(config, row):
    path = _safe_path(config.output_dir, row["npy_path"])
    if _sha(path) != row["signal_sha256"]:
        raise ValueError(f"Checksum output không khớp: ECG {row['ecg_id']}")
    x = np.load(path, allow_pickle=False)
    if x.shape != (12, config.target_fs * 10) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError(f"Array không hợp lệ: ECG {row['ecg_id']}")
    return x


def fit_normalizer(config: PipelineConfig) -> dict:
    """Population mean/std over train samples only, using stable batch Welford."""
    identity, _ = _context(config)
    records = _records(config, identity)
    train = records[(records.status == "accepted") & (records.split == "train")]
    if train.empty:
        raise ValueError("Không có train ECG accepted")
    count = 0
    mean, m2 = np.zeros(12, dtype=np.float64), np.zeros(12, dtype=np.float64)
    for row in train.to_dict(orient="records"):
        x = _accepted_array(config, row).astype(np.float64)
        n = x.shape[1]
        local_mean = x.mean(axis=1)
        local_m2 = ((x - local_mean[:, None]) ** 2).sum(axis=1)
        delta = local_mean - mean
        total = count + n
        mean += delta * n / total
        m2 += local_m2 + delta**2 * count * n / total
        count = total
    std = np.sqrt(m2 / count)
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Train standard deviation không hữu hạn hoặc bằng 0")
    positives = train[list(LABEL_COLUMNS)].sum(axis=0).to_numpy(dtype=int)
    if np.any(positives == 0):
        raise ValueError(f"Train có lớp không có positive: {[CLASSES[i] for i in np.where(positives == 0)[0]]}")
    negatives = len(train) - positives
    stats = {"fingerprint": identity["fingerprint"], "records_sha256": identity["records_sha256"],
             "classes": list(CLASSES), "leads": list(LEADS), "mean": mean.tolist(), "std": std.tolist(),
             "train_records": len(train), "samples_per_lead": count, "ddof": 0,
             "positive_counts": positives.tolist(), "negative_counts": negatives.tolist(),
             "pos_weight": (negatives / positives).tolist(), "fit_split": "train", "units": "mV"}
    _atomic_json(config.output_dir / "normalization.json", stats)
    identity["normalization_sha256"] = _sha(config.output_dir / "normalization.json")
    _atomic_json(config.output_dir / "pipeline.json", identity)
    return stats


def audit_release(config: PipelineConfig) -> dict:
    """Check every accepted sample and raw checksum; only full-source runs are ready."""
    errors = []
    identity, catalog = _context(config)
    metadata = _metadata(config, identity)
    records = _records(config, identity)
    if len(records) != len(metadata) or records.ecg_id.duplicated().any() or set(records.ecg_id) != set(metadata.ecg_id):
        errors.append("record/metadata alignment failed")
    try:
        _patient_check(records)
    except ValueError as exc:
        errors.append(str(exc))
    common = ["ecg_id", "patient_id", "strat_fold", "split", "filename_hr", "has_label", *LABEL_COLUMNS]
    expected = metadata[common].sort_values("ecg_id").reset_index(drop=True).astype(str)
    actual = records[common].sort_values("ecg_id").reset_index(drop=True).astype(str)
    if not expected.equals(actual):
        errors.append("record labels/splits differ from immutable metadata")
    duplicate_groups = {}
    for row in records.to_dict(orient="records"):
        try:
            source_hashes = _source_hashes(config, row, catalog)
            sidecar = _valid_sidecar(config, config.output_dir / "sidecars" / f"{int(row['ecg_id']):05d}.json", identity["fingerprint"], source_hashes)
            if sidecar is None or sidecar.get("ecg_id") != row["ecg_id"] or sidecar.get("status") != row["status"]:
                raise ValueError("missing/stale sidecar")
            if row["status"] == "accepted":
                x = _accepted_array(config, row)
                if not row["has_label"] or row["fs"] != config.target_fs or row["n_samples"] != config.target_fs * 10 or row["units"] != "mV":
                    raise ValueError("accepted metadata contract failed")
                if np.any(np.ptp(x, axis=1) == 0):
                    raise ValueError("constant output lead")
                duplicate_groups.setdefault(row["signal_sha256"], []).append({"ecg_id": int(row["ecg_id"]), "patient_id": str(row["patient_id"]), "split": row["split"]})
            elif row["status"] not in ("rejected", "no_label") or (row["status"] == "no_label" and row["has_label"]):
                raise ValueError("invalid status")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"ECG {row['ecg_id']}: {exc}")
    duplicates = [group for group in duplicate_groups.values() if len(group) > 1]
    cross_duplicates = [group for group in duplicates if len({item["split"] for item in group}) > 1]
    if cross_duplicates:
        errors.append(f"cross-split duplicate waveforms: {len(cross_duplicates)} groups")
    accepted = records[records.status == "accepted"]
    split_counts = {split: int((accepted.split == split).sum()) for split in SPLITS}
    if any(count == 0 for count in split_counts.values()):
        errors.append("at least one accepted split is empty")
    stats_path = config.output_dir / "normalization.json"
    try:
        if _sha(stats_path) != identity.get("normalization_sha256"):
            raise ValueError("normalization checksum mismatch; chạy fit_normalizer")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        train = accepted[accepted.split == "train"]
        if stats.get("fingerprint") != identity["fingerprint"] or stats.get("records_sha256") != identity["records_sha256"] or stats.get("fit_split") != "train" or stats.get("train_records") != len(train) or stats.get("classes") != list(CLASSES) or stats.get("leads") != list(LEADS):
            raise ValueError("normalizer provenance mismatch")
        if np.asarray(stats["mean"]).shape != (12,) or np.asarray(stats["std"]).shape != (12,) or not np.isfinite(stats["mean"]).all() or not np.isfinite(stats["std"]).all() or np.any(np.asarray(stats["std"]) <= 0):
            raise ValueError("normalizer numerical contract failed")
        if stats["positive_counts"] != train[list(LABEL_COLUMNS)].sum().astype(int).tolist():
            raise ValueError("normalizer class counts mismatch")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"normalization: {exc}")
    fixture = config.expected_records is None
    full_source = config.expected_records == 21799 and len(records) == 21799
    if not fixture and not full_source:
        errors.append("only all 21,799 source records constitute PTB-XL v1.0.3 release")
    report = {
        "ready": not errors and full_source, "checks_passed": not errors, "fixture": fixture,
        "source_version": VERSION, "fingerprint": identity["fingerprint"],
        "audited_at": datetime.now(timezone.utc).isoformat(), "source_records": len(metadata),
        "accepted_records": len(accepted), "status_counts": {str(k): int(v) for k, v in records.status.value_counts().items()},
        "split_counts": split_counts, "patient_counts": {split: int(accepted[accepted.split == split].patient_id.nunique()) for split in SPLITS},
        "label_counts": {split: {name: int(accepted[accepted.split == split][f"y_{name}"].sum()) for name in CLASSES} for split in SPLITS},
        "exclusions_by_label": {name: int(records[records.status != "accepted"][f"y_{name}"].sum()) for name in CLASSES},
        "exclusion_reasons": {str(k): int(v) for k, v in records[records.status != "accepted"].reason.value_counts().items()},
        "norm_with_abnormal": int(accepted.norm_with_abnormal.sum()),
        "duplicate_groups": duplicates, "cross_split_duplicates": cross_duplicates,
        "errors": errors, "artifact_sha256": {},
    }
    for split in SPLITS:
        path = config.output_dir / f"{split}.csv"
        _atomic_csv(path, accepted[accepted.split == split].reset_index(drop=True))
        report["artifact_sha256"][path.name] = _sha(path)
    for name in ("pipeline.json", "metadata.csv", "records.csv", "normalization.json"):
        path = config.output_dir / name
        if path.is_file():
            report["artifact_sha256"][name] = _sha(path)
    _atomic_json(config.output_dir / "audit.json", report)
    _write_card(config, report, identity)
    return report


def _write_card(config, report, identity):
    rows = "\n".join(f"| {split} | {report['split_counts'][split]} | {report['patient_counts'][split]} |" for split in SPLITS)
    text = f"""# Bộ dữ liệu PTB-XL v1.0.3 — Line

Trạng thái: {'SẴN SÀNG train/validation/test theo hợp đồng dữ liệu' if report['ready'] else 'CHƯA phải bản phát hành đầy đủ sẵn sàng'}. Audit: {report['audited_at']}.

Nguồn chính thức: https://physionet.org/content/ptb-xl/1.0.3/ ; giấy phép CC BY 4.0. Nguồn và tác giả gốc cần được trích dẫn khi sử dụng. Đây là nhãn ECG, không phải nhãn suy tim hay xác nhận sức khỏe lâm sàng.

Nguồn có {report['source_records']} ECG; giữ {report['accepted_records']}. Trạng thái từng ECG: {json.dumps(report['status_counts'], ensure_ascii=False)}.

| Tập | ECG được giữ | Bệnh nhân |
|---|---:|---:|
{rows}

Nhãn multi-hot theo thứ tự: {', '.join(CLASSES)}. Mỗi ECG có thể có nhiều nhãn. Mã SCP hiện diện được giữ kể cả likelihood = 0 (không biết độ tin cậy); không có diagnostic superclass thì loại khỏi supervised set. Cờ NORM đi cùng bất thường được giữ để kiểm tra: {report['norm_with_abnormal']} ECG.

Split chính thức theo bệnh nhân: folds 1–8 train, 9 validation, 10 test. Không fit chuẩn hóa hoặc class weights trên validation/test. Ngưỡng dự đoán phải chọn trên validation; test chỉ dùng khi chốt phương án.

Array float32 [12, {10 * config.target_fs}], {config.target_fs} Hz, 10 giây, thứ tự lead {', '.join(LEADS)}, đơn vị mV. Cấu hình `{config.preprocessing}`. Bản minimal không lọc bắt buộc, cắt biên độ, nội suy hay chuẩn hóa từng ECG. Chuẩn hóa bằng mean/std từng lead của train trong loader; file .npy giữ biên độ mV. Lọc hoặc giảm tần số phải tạo output_dir khác và được đánh giá bằng thí nghiệm riêng.

QC loại dữ liệu sai shape/lead/units, NaN/Inf và lead hằng; không dùng ngưỡng biên độ để suy diễn bệnh lý. Metadata nhiễu nguồn được giữ. Chi tiết loại theo nhãn, split và lý do có trong records.csv/audit.json. Trùng waveform qua tập: {len(report['cross_split_duplicates'])} nhóm; trùng nội bộ: {len(report['duplicate_groups']) - len(report['cross_split_duplicates'])} nhóm.

Provenance: fingerprint `{identity['fingerprint']}`; SHA256 raw theo manifest chính thức, SHA256 từng array trong records.csv, mã triển khai và runtime trong pipeline.json. Bộ fixture: {report['fixture']}. Lỗi audit: {json.dumps(report['errors'], ensure_ascii=False)}.

Giới hạn: đánh giá nội bộ PTB-XL không chứng minh hiệu quả trên bệnh viện, thiết bị hay quần thể khác. Nhãn có thể không đầy đủ; QC có thể gây lệch chọn mẫu. Audit cấu trúc không xác nhận chẩn đoán. Bộ nhãn này không được đổi tên thành suy tim.

Đánh giá model tiếp theo: sigmoid 5 đầu ra và BCE; báo AUROC, AUPRC, F1/recall/precision từng nhãn và macro; chọn threshold trên validation, giữ test độc lập. Kiểm tra subgroup khi có đủ mẫu. Không suy ra accuracy hay chất lượng model từ audit này.

Dung lượng array lý thuyết: {report['accepted_records'] * 12 * config.target_fs * 10 * 4 / 1e9:.3f} GB, chưa kể metadata/nguồn. Pipeline CPU; batch đọc giới hạn RAM, latency phụ thuộc ổ đĩa. Không có chi phí API. Hoàn tác bằng chọn thư mục output mới; nguồn raw bất biến.
"""
    _atomic_bytes(config.output_dir / "DATASET_CARD.vi.md", text.encode("utf-8"))


class ECGDataset:
    """NumPy dataset. Normalization always uses the saved training statistics."""

    def __init__(self, output_dir: Path, split: str = "train", normalize: bool = True):
        self.output_dir = Path(output_dir).resolve()
        if split not in SPLITS:
            raise ValueError(f"split phải thuộc {SPLITS}")
        audit = json.loads((self.output_dir / "audit.json").read_text(encoding="utf-8"))
        if not audit.get("checks_passed") or not (audit.get("ready") or audit.get("fixture")):
            raise ValueError("Audit chưa đạt; không nạp bộ dữ liệu train/test")
        for name in ("pipeline.json", "records.csv", "normalization.json", f"{split}.csv"):
            if _sha(self.output_dir / name) != audit["artifact_sha256"].get(name):
                raise ValueError(f"Artifact đã thay đổi sau audit: {name}")
        self.manifest = _read_frame(self.output_dir / f"{split}.csv")
        self.normalize = normalize
        self.stats = json.loads((self.output_dir / "normalization.json").read_text(encoding="utf-8"))
        self.mean = np.asarray(self.stats["mean"], dtype=np.float64)[:, None]
        self.std = np.asarray(self.stats["std"], dtype=np.float64)[:, None]
        identity = json.loads((self.output_dir / "pipeline.json").read_text(encoding="utf-8"))
        self.shape = (12, identity["config"]["target_fs"] * 10)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]
        path = _safe_path(self.output_dir, row.npy_path)
        if _sha(path) != row.signal_sha256:
            raise ValueError(f"Array đã thay đổi sau audit: {path.name}")
        x = np.load(path, allow_pickle=False)
        if x.shape != self.shape or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f"Array không hợp lệ: {path.name}")
        if self.normalize:
            x = ((x.astype(np.float64) - self.mean) / self.std).astype(np.float32)
        y = row[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
        if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Batch không hữu hạn hoặc nhãn ngoài 0/1")
        return x, y


def iter_batches(dataset: ECGDataset, batch_size: int = 32, shuffle: bool = False, seed: int = 42) -> Iterator:
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size phải là số nguyên dương")
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        examples = [dataset[int(index)] for index in indices[start:start + batch_size]]
        yield np.stack([x for x, _ in examples]), np.stack([y for _, y in examples])
