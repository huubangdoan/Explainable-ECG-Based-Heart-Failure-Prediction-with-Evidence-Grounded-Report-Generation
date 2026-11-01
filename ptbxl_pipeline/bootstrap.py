"""Portable source acquisition and cohort-equivalent school release preparation.

Intent: move acquisition/preparation to the target without changing the experiment.
Status: fixture-tested; real acquisition and CUDA runs remain target-only checks.
Next: inspect the target proof and results. Blockers: none known in offline fixtures.
Confidence: byte/cohort gates are explicit; no claim of model quality or target speed.
"""
from __future__ import annotations

import copy
import csv
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import traceback

from ptbxl_experiments.protocol import (CLASSES, LEADS, build_jobs, code_fingerprint,
                                        load_protocol, protocol_hash, write_json)
from ptbxl_experiments.suite import (SuiteLock, _Tee, collect_results, file_hash,
                                    freeze_json, utc_now)

COHORT_COLUMNS = (
    "ecg_id", "patient_id", "strat_fold", "split", "y_NORM", "y_MI", "y_STTC", "y_CD", "y_HYP",
    "has_label", "status", "reason", "filename_hr", "filename_lr", "npy_path", "signal_sha256",
    "fs", "n_samples", "units",
)
_NUMERIC_COLUMNS = {"ecg_id", "patient_id", "strat_fold", "has_label", "fs", "n_samples",
                    *(f"y_{name}" for name in CLASSES)}
_BINARY_COLUMNS = {"has_label", *(f"y_{name}" for name in CLASSES)}
_PACKAGES = {"numpy": "2.5.4", "pandas": "3.1.0", "scipy": "1.18.1", "wfdb": "4.3.1"}
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_SPLITS = {"train", "validation", "test"}
_CONTRACT_KEYS = {"schema_version", "source_version", "classes", "leads", "source_records",
                  "accepted_records", "split_counts", "checksum_catalog_sha256",
                  "source_metadata_sha256", "producer_implementation_sha256", "cohort_sha256",
                  "reference_release"}


def _read_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8-sig") as handle:
        result = json.load(handle, object_pairs_hook=unique_keys)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def _sha_value(value, name):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def validate_contract(contract: dict) -> dict:
    if not isinstance(contract, dict) or set(contract) != _CONTRACT_KEYS:
        raise ValueError("Cohort contract must have the specified schema keys")
    if type(contract["schema_version"]) is not int or contract["schema_version"] != 1 or contract["source_version"] != "1.0.3":
        raise ValueError("Unsupported cohort contract version")
    if contract["classes"] != list(CLASSES) or contract["leads"] != list(LEADS):
        raise ValueError("Cohort contract class/lead order mismatch")
    for name in ("source_records", "accepted_records"):
        if type(contract[name]) is not int or contract[name] <= 0:
            raise ValueError(f"Cohort contract {name} must be a positive integer")
    if contract["accepted_records"] > contract["source_records"]:
        raise ValueError("Cohort accepted count exceeds source count")
    counts = contract["split_counts"]
    if not isinstance(counts, dict) or set(counts) != _SPLITS or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError("Cohort contract needs positive train/validation/test counts")
    if sum(counts.values()) != contract["accepted_records"]:
        raise ValueError("Cohort contract split counts do not sum to accepted_records")
    for name in ("checksum_catalog_sha256", "producer_implementation_sha256", "cohort_sha256"):
        _sha_value(contract[name], name)
    for field, keys in (("source_metadata_sha256", {"ptbxl_database.csv", "scp_statements.csv"}),
                        ("reference_release", {"fingerprint", "audit_sha256"})):
        values = contract[field]
        if not isinstance(values, dict) or set(values) != keys:
            raise ValueError(f"Cohort contract {field} keys mismatch")
        for key, value in values.items():
            _sha_value(value, f"{field}.{key}")
    return contract


def _load_inputs(config_path, contract_path):
    base = load_protocol(config_path)
    contract = validate_contract(_read_json(contract_path))
    source = base["source"]
    expected = {"version": contract["source_version"], "expected_records": contract["source_records"],
                "expected_accepted": contract["accepted_records"],
                "expected_fingerprint": contract["reference_release"]["fingerprint"],
                "expected_audit_sha256": contract["reference_release"]["audit_sha256"]}
    if source != expected:
        raise ValueError("Base protocol source anchors/counts disagree with cohort contract")
    if file_hash(Path(__file__).with_name("core.py")) != contract["producer_implementation_sha256"]:
        raise ValueError("Producer implementation SHA256 differs from cohort contract; preserve source LF bytes")
    return base, contract


def cohort_digest(records_path: Path) -> str:
    """Canonical semantic identity of every row; never read waveform files."""
    rows, ids = [], set()
    with Path(records_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or not set(COHORT_COLUMNS).issubset(reader.fieldnames):
            raise ValueError("Records CSV lacks unique required cohort columns")
        for number, source in enumerate(reader, 2):
            if None in source or any(source.get(key) is None for key in COHORT_COLUMNS):
                raise ValueError(f"Malformed records CSV row {number}")
            row = []
            for key in COHORT_COLUMNS:
                value = source[key]
                if key in _NUMERIC_COLUMNS:
                    if not value.strip():
                        value = None
                    else:
                        try:
                            decimal = Decimal(value)
                        except InvalidOperation as exc:
                            raise ValueError(f"Invalid numeric {key} at CSV row {number}") from exc
                        if not decimal.is_finite() or decimal != decimal.to_integral_value():
                            raise ValueError(f"Nonfinite/noninteger {key} at CSV row {number}")
                        value = int(decimal)
                    if key in _BINARY_COLUMNS and value not in (0, 1):
                        raise ValueError(f"Nonbinary {key} at CSV row {number}")
                row.append(value)
            ecg_id = row[0]
            if ecg_id is None or ecg_id <= 0 or ecg_id in ids:
                raise ValueError(f"Missing, invalid, or duplicate ecg_id at CSV row {number}")
            if source["split"] not in _SPLITS or source["status"] not in {"accepted", "no_label", "rejected"}:
                raise ValueError(f"Unknown split/status at CSV row {number}")
            ids.add(ecg_id)
            rows.append(row)
    if not rows:
        raise ValueError("Empty cohort records")
    rows.sort(key=lambda row: row[0])
    return protocol_hash({"columns": list(COHORT_COLUMNS), "rows": rows})


def verify_equivalent_release(release_dir: Path, contract: dict, *, full: bool = True) -> dict:
    validate_contract(contract)
    from ptbxl_experiments.data import verify_release
    release = Path(release_dir)
    result = verify_release(release, full=full)
    for field in ("source_records", "accepted_records", "split_counts", "classes", "leads"):
        if result[field] != contract[field]:
            raise ValueError(f"Release {field} differs from cohort contract")
    pipeline = _read_json(release / "pipeline.json")
    for field in ("source_version", "checksum_catalog_sha256", "source_metadata_sha256"):
        if pipeline.get(field) != contract[field]:
            raise ValueError(f"Release {field} differs from cohort contract")
    if pipeline.get("config") != {"target_fs": 500, "preprocessing": "minimal", "expected_records": contract["source_records"]}:
        raise ValueError("Release producer configuration differs from minimal500 cohort contract")
    implementation = contract["producer_implementation_sha256"]
    if pipeline.get("implementation_sha256") != implementation or file_hash(Path(__file__).with_name("core.py")) != implementation:
        raise ValueError("Release/current producer implementation SHA256 differs from cohort contract")
    cohort = cohort_digest(release / "records.csv")
    if cohort != contract["cohort_sha256"]:
        raise ValueError("Release semantic cohort SHA256 differs from cohort contract")
    return {**result, "cohort_sha256": cohort}


def _locations(raw_dir, release_dir, cache_dir, work_dir):
    paths = []
    for value in (raw_dir, release_dir, cache_dir, work_dir):
        original = Path(value).absolute()
        for path in (original, *original.parents):
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ValueError(f"Bootstrap roots cannot traverse symlinks/junctions: {path}")
        if original.exists() and not original.is_dir():
            raise ValueError(f"Bootstrap root is not a directory: {original}")
        paths.append(original.resolve())
    for index, path in enumerate(paths):
        for other in paths[index + 1:]:
            if path == other or path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError("Raw, release, source-cache and work roots must be separate, not parent/child")
    return tuple(paths)


def plan(config_path, contract_path, raw_dir, release_dir, cache_dir, work_dir) -> dict:
    """Read-only, stdlib-only plan: no package imports, downloads, or preparation."""
    base, contract = _load_inputs(config_path, contract_path)
    paths = _locations(raw_dir, release_dir, cache_dir, work_dir)
    return {"suite_id": base["suite_id"], "base_protocol_sha256": protocol_hash(base),
            "cohort_contract_sha256": protocol_hash(contract), "cohort_sha256": contract["cohort_sha256"],
            "source_records": contract["source_records"], "accepted_records": contract["accepted_records"],
            "number_of_jobs": len(build_jobs(base)), "paths": dict(zip(("raw", "release", "cache", "work"), map(str, paths))),
            "requires_python": "3.12.x", "requires_packages": _PACKAGES,
            "network_execution": False, "data_preparation": False, "model_execution": False}


def _runtime():
    versions = {}
    for name, expected in _PACKAGES.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"Missing {name}; run Setup-School.ps1 in the target environment") from exc
        if actual != expected:
            raise ValueError(f"{name}=={expected} required; found {actual}. Use the pinned target environment")
        versions[name] = actual
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Python 3.12.x required for target preparation")
    return {"python": platform.python_version(), **versions}


def _check_disk(base, raw_dir, release_dir, cache_dir, work_dir):
    gib = 1024**3
    count, accepted = base["source"]["expected_records"], base["source"]["expected_accepted"]
    estimates = (int(count * 12 * 5000 * 2 * 1.2) + gib,
                 int(accepted * 12 * 5000 * 4 * 1.15) + gib, gib,
                 int(sum(accepted * 12 * v["fs"] * 10 * 4 for v in base["variants"]) * 1.15) + 9 * gib)
    volumes = {}
    for root, estimate in zip((raw_dir, release_dir, cache_dir, work_dir), estimates):
        ancestor = root
        while not ancestor.exists():
            ancestor = ancestor.parent
        # Existing regular files already consume the budget; linked files never count.
        existing = sum(p.stat().st_size for p in root.rglob("*")
                       if p.is_file() and not p.is_symlink()) if root.exists() else 0
        volume = volumes.setdefault(ancestor.stat().st_dev, {"path": str(ancestor), "required_free_bytes_estimate": 0,
                                                           "free_bytes": shutil.disk_usage(ancestor).free})
        volume["required_free_bytes_estimate"] += max(0, estimate - existing)
    for item in volumes.values():
        if item["free_bytes"] < item["required_free_bytes_estimate"]:
            raise OSError(f"Insufficient disk space at {item['path']}: {item['free_bytes'] / gib:.1f} GiB free; "
                          f"need about {item['required_free_bytes_estimate'] / gib:.1f} GiB more for source/release/caches/results")
    return list(volumes.values())


def _ensure_source(raw_dir, cache_dir, *, download):
    from .acquire import ensure_source
    return ensure_source(raw_dir, cache_dir, download=download)


def _pipeline_stages(config):
    from .core import audit_release, build_manifest, fit_normalizer, process_signals
    build_manifest(config)
    process_signals(config)
    fit_normalizer(config)
    audit = audit_release(config)
    if audit.get("ready") is not True:
        raise ValueError(f"Producer audit failed: {audit.get('errors', [])}")


def _guard_release(release, contract, runtime, work):
    identity_path, audit_path = release / "pipeline.json", release / "audit.json"
    if release.exists() and any(release.iterdir()) and not identity_path.is_file():
        raise ValueError("Nonempty release has no pipeline identity; choose a NEW release directory")
    audit = _read_json(audit_path) if audit_path.exists() else {}
    if audit.get("ready") is True:
        return True  # Ready releases use portable audit, never producer _context.
    if (work / "target_protocol.json").exists() or (work / "protocol.json").exists():
        raise ValueError("Previously bound release is no longer ready; keep evidence and use NEW directories")
    if identity_path.exists():
        identity = _read_json(identity_path)
        expected = {"source_version": contract["source_version"], "runtime": runtime,
                    "implementation_sha256": contract["producer_implementation_sha256"],
                    "source_metadata_sha256": contract["source_metadata_sha256"],
                    "checksum_catalog_sha256": contract["checksum_catalog_sha256"],
                    "classes": contract["classes"], "leads": contract["leads"],
                    "config": {"target_fs": 500, "preprocessing": "minimal", "expected_records": contract["source_records"]}}
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError("Incomplete release identity/runtime differs; choose a NEW release directory")
    return False


def _verify_source_certificate(source, contract):
    expected = {"verified": True, "record_count": contract["source_records"],
                "source_version": contract["source_version"],
                "checksum_manifest_sha256": contract["checksum_catalog_sha256"]}
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError("Source acquisition did not certify the required complete source/count/catalog")
    if any(source.get("file_hashes", {}).get(key) != value for key, value in contract["source_metadata_sha256"].items()):
        raise ValueError("Source acquisition metadata checksum differs from cohort contract")


def _scientific_protocol(value):
    result = copy.deepcopy(value)
    result["source"].pop("expected_fingerprint", None)
    result["source"].pop("expected_audit_sha256", None)
    return result


def _guard_work(work, base, contract, runtime):
    if (work / "cohort_contract.json").exists() and _read_json(work / "cohort_contract.json") != contract:
        raise ValueError("Frozen cohort contract changed; use a NEW work directory")
    for name in ("target_protocol.json", "protocol.json"):
        if (work / name).exists() and _scientific_protocol(load_protocol(work / name)) != _scientific_protocol(base):
            raise ValueError(f"Frozen scientific protocol changed: {name}; use a NEW work directory")
    expected = {"base_protocol_sha256": protocol_hash(base), "cohort_contract_sha256": protocol_hash(contract),
                "runtime": runtime, "bootstrap_implementation_sha256": file_hash(Path(__file__))}
    if (work / "bootstrap_verification.json").exists():
        proof = _read_json(work / "bootstrap_verification.json")
        if any(proof.get(k) != v for k, v in expected.items()):
            raise ValueError("Frozen bootstrap evidence/runtime changed; use a NEW work directory")
    if (work / "identity.json").exists():
        identity = _read_json(work / "identity.json")
        if not (work / "protocol.json").is_file():
            raise ValueError("Existing suite identity has no protocol; retain it and use a NEW work directory")
        existing = load_protocol(work / "protocol.json")
        wanted = {"protocol_hash": protocol_hash(existing), "code_fingerprint": code_fingerprint(),
                  "source_fingerprint": existing["source"]["expected_fingerprint"]}
        if identity != wanted:
            raise ValueError("Existing suite code/protocol identity differs; use a NEW work directory")
    elif any((work / name).exists() for name in ("runs", "selection.json", "evaluation")):
        raise ValueError("Run/test evidence without suite identity; retain it and use a NEW work directory")
    return expected


def _record_failure(work, exc):
    write_json(work / "bootstrap_error.json", {"state": "failed", "updated_at": utc_now(),
               "exception": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
               "action": "Keep existing evidence; fix the stated prerequisite, then repeat the same command. "
                         "Identity mismatches require new release/work directories."})
    try:
        print(f"Bootstrap diagnostics: {collect_results(work)}", flush=True)
    except Exception as diagnostic_error:
        print(f"Diagnostics ZIP unavailable: {diagnostic_error}; bootstrap_error.json/log retained", file=sys.stderr, flush=True)


def prepare_school(config_path: Path, contract_path: Path, raw_dir: Path, release_dir: Path,
                   cache_dir: Path, work_dir: Path, *, allow_download: bool = False) -> Path:
    raw, release, cache, work = _locations(raw_dir, release_dir, cache_dir, work_dir)
    work.mkdir(parents=True, exist_ok=True)
    # Own the same work lock as training; errors cannot mutate another live suite.
    with SuiteLock(work / "suite.lock"):
        with (work / "bootstrap.log").open("a", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
                try:
                    print(f"[{utc_now()}] Bootstrap data preparation; model execution disabled", flush=True)
                    base, contract = _load_inputs(config_path, contract_path)
                    runtime = _runtime()
                    proof_header = _guard_work(work, base, contract, runtime)
                    _guard_release(release, contract, runtime, work)
                    print(f"Disk budget: {json.dumps(_check_disk(base, raw, release, cache, work))}", flush=True)
                    # Same source uses the same OS lock across every destination work directory.
                    lock_id = hashlib.sha256(str(raw).casefold().encode("utf-8")).hexdigest()[:16]
                    release_lock_id = hashlib.sha256(str(release).casefold().encode("utf-8")).hexdigest()[:16]
                    with SuiteLock(raw.parent / f".ptbxl-bootstrap-{lock_id}.lock"), \
                         SuiteLock(release.parent / f".ptbxl-release-{release_lock_id}.lock"):
                        ready = _guard_release(release, contract, runtime, work)
                        print("Verifying/acquiring official 500 Hz source", flush=True)
                        source = _ensure_source(raw, cache, download=allow_download)
                        _verify_source_certificate(source, contract)
                        if not ready:
                            from .core import PipelineConfig
                            print("Building/resuming immutable minimal500 release", flush=True)
                            _pipeline_stages(PipelineConfig(raw, release, target_fs=500, preprocessing="minimal",
                                                           expected_records=contract["source_records"]))
                        print("Full receiving audit and exact cohort comparison", flush=True)
                        verified = verify_equivalent_release(release, contract, full=True)
                    target = copy.deepcopy(base)
                    current = {"fingerprint": verified["source_fingerprint"], "audit_sha256": verified["audit_sha256"]}
                    target["source"].update(expected_fingerprint=current["fingerprint"], expected_audit_sha256=current["audit_sha256"])
                    proof = {**proof_header, "schema_version": 1, "reference_release": contract["reference_release"],
                             "current_release": current, "verified_release": verified,
                             "identity_change_reason": "Producer runtime and audit timestamps can change identity; "
                                                       "all source/core anchors, cohort semantics and accepted array bytes must match."}
                    # Check all frozen evidence before writing any new binding document.
                    documents = {"cohort_contract.json": contract, "bootstrap_verification.json": proof,
                                 "target_protocol.json": target}
                    if (work / "protocol.json").exists() and load_protocol(work / "protocol.json") != target:
                        raise ValueError("Existing suite protocol anchors differ; refusing to rebind active suite")
                    for name, value in documents.items():
                        if (work / name).exists() and _read_json(work / name) != value:
                            raise ValueError(f"Frozen evidence changed: {name}; use a NEW work directory")
                    for name, value in documents.items():
                        freeze_json(work / name, value)
                    print(f"Equivalent release verified; target protocol: {work / 'target_protocol.json'}", flush=True)
                    return work / "target_protocol.json"
                except (Exception, KeyboardInterrupt) as exc:
                    print(f"Bootstrap failed: {exc}", file=sys.stderr, flush=True)
                    _record_failure(work, exc)
                    raise
