"""Certified, copy-only recovery of Line's closed-test official100 incident.

Intent: preserve 18 completed release500 jobs across the lead-case-only repair.
Status: synthetic tests; checkpoint/CUDA verification runs on the school target.
Next: inspect the immutable migration proof before interpreting target results.
Blockers: any code, cohort, runtime, payload or provenance mismatch fails closed.
Confidence: identity conversion is narrowly certified, not a model quality claim.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import traceback

from ptbxl_experiments.protocol import build_jobs, code_fingerprint, load_protocol, protocol_hash, write_json
from ptbxl_experiments.suite import SuiteLock, file_hash, freeze_json, utc_now
from . import bootstrap

ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ID = "official100-lead-case-v1"
BASE_COMMIT = "4369d39712b9f7412af7b7819524271f29155afb"
ARTIFACTS = ("result.json", "validation_predictions.npz", "val_thresholds.npy",
             "history.json", "history.csv", "best.pt", "last.pt")
_UNCHANGED = {f"ptbxl_experiments/{name}.py" for name in
              ("__init__", "metrics", "model", "protocol", "suite", "training")}
_CERTIFICATE_KEYS = {"schema_version", "recovery_id", "base_commit", "old_code_fingerprint",
                     "new_code_fingerprint", "old_data_sha256", "new_data_sha256", "unchanged_files",
                     "change_description"}
_JOURNAL = ".official100-recovery.json"
_PROOF = "bootstrap_verification.json"
_GIB = 1024**3


def _json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            value[key] = item
        return value
    _regular(path)
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle, object_pairs_hook=unique,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))


def _unlinked(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise ValueError(f"Recovery refuses symlink/junction: {item}")
    return path


def _regular(path):
    path = _unlinked(path)
    if not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"Expected regular, unlinked recovery file: {path}")
    return path


def _check_lock(path):
    # SuiteLock initializes an empty file with one byte. Reject links BEFORE
    # entering it, so that initialization cannot mutate a linked source artifact.
    path = _unlinked(path)
    if path.exists():
        _regular(path)


def _tree(root):
    """Inspect components before descent; never follow a Windows junction."""
    root = _unlinked(root)
    if not root.exists():
        return []
    pending, files = [root], []
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            _unlinked(path)
            if path.is_dir():
                pending.append(path)
            else:
                files.append(_regular(path))
    return sorted(files)


def _paths(old_work_dir, work_dir, release_dir, raw_dir):
    result = bootstrap._locations(old_work_dir, work_dir, release_dir, raw_dir)
    if not result[0].is_dir():
        raise ValueError("Original work directory must exist")
    return result


def _certificate(path):
    certificate = _json(path)
    if not isinstance(certificate, dict) or set(certificate) != _CERTIFICATE_KEYS:
        raise ValueError("Recovery certificate exact schema required")
    if (type(certificate["schema_version"]) is not int or certificate["schema_version"] != 1 or
            certificate["recovery_id"] != RECOVERY_ID or certificate["base_commit"] != BASE_COMMIT):
        raise ValueError("Unknown recovery certificate/version/base commit")
    for name in ("old_code_fingerprint", "new_code_fingerprint", "old_data_sha256", "new_data_sha256"):
        bootstrap._sha_value(certificate[name], name)
    if not isinstance(certificate["change_description"], str) or not certificate["change_description"].strip():
        raise ValueError("Certificate requires a change description")
    unchanged = certificate["unchanged_files"]
    if not isinstance(unchanged, dict) or set(unchanged) != _UNCHANGED:
        raise ValueError("Certificate must bind exactly the six unchanged experiment files")
    for name, digest in unchanged.items():
        bootstrap._sha_value(digest, name)
        if file_hash(_regular(ROOT / name)) != digest:
            raise ValueError(f"Certified unchanged source differs: {name}")
    data_path = ROOT / "ptbxl_experiments/data.py"
    if file_hash(_regular(data_path)) != certificate["new_data_sha256"] or code_fingerprint() != certificate["new_code_fingerprint"]:
        raise ValueError("Current code/data SHA256 differs from recovery certificate")
    # Reconstruct the predecessor bytes, not a spoofed runtime identity. This also
    # prevents a certificate from blessing an unrelated data.py compatibility edit.
    current = data_path.read_bytes()
    comment = b"    # Official headers use AVR/AVL/AVF; ignore case while preserving lead order.\n"
    new = b"[name.upper() for name in header.sig_name] != [name.upper() for name in LEADS]"
    old = b"header.sig_name != list(LEADS)"
    if current.count(comment) != 1 or current.count(new) != 1:
        raise ValueError("Certified lead-case repair shape differs")
    previous = current.replace(comment, b"", 1).replace(new, old, 1)
    if hashlib.sha256(previous).hexdigest() != certificate["old_data_sha256"]:
        raise ValueError("Certificate predecessor data SHA256 mismatch")
    digest = hashlib.sha256()
    for file in sorted(data_path.parent.glob("*.py")):
        digest.update(file.name.encode("utf-8") + b"\0")
        digest.update(previous if file == data_path else file.read_bytes())
        digest.update(b"\0")
    if digest.hexdigest() != certificate["old_code_fingerprint"]:
        raise ValueError("Certificate predecessor code fingerprint mismatch")
    return certificate


def _source_closed(old, files):
    if (old / "selection.json").exists() or (old / "evaluation").exists():
        raise ValueError("Source test/selection/evaluation evidence exists; recovery is not eligible")
    if (old / "status.json").exists() and _json(old / "status.json").get("state") == "completed":
        raise ValueError("Source suite already completed; test must remain closed")
    from ptbxl_experiments.suite import EVAL_ARTIFACTS
    if any(path.name in EVAL_ARTIFACTS for path in files):
        raise ValueError("Source contains test evaluation evidence")


def _incident(old, missing):
    errors = _json(old / "errors.json")
    if not isinstance(errors, list) or len(errors) != len(missing) + 1 or not all(isinstance(e, dict) for e in errors):
        raise ValueError("Source errors must identify only this official100 incident and its missing reference jobs")
    prepare = [e for e in errors if e.get("phase") == "prepare"]
    if (len(prepare) != 1 or prepare[0].get("variant") != missing[0]["variant"]["id"] or
            prepare[0].get("error") != "Official100 header contract mismatch: records100/00000/00001_lr"):
        raise ValueError("Source errors do not identify the certified official100 header contract failure")
    expected = {j["id"] for j in missing}
    train = [e for e in errors if e.get("phase") == "train"]
    if (len(train) != len(expected) or {e.get("job_id") for e in train} != expected or
            any(e.get("error") != "prepared variant unavailable" for e in train)):
        raise ValueError("Source missing reference-job errors do not match this incident")


def _old_metadata(path, variant, certificate, audit):
    metadata = _json(path)
    content = {k: v for k, v in metadata.items() if k != "metadata_sha256"}
    provenance = metadata.get("provenance", {})
    if (metadata.get("metadata_sha256") != protocol_hash(content) or
            metadata.get("fingerprint") != protocol_hash(provenance) or
            metadata.get("schema_version") != 1 or metadata.get("ready") is not True):
        raise ValueError(f"Old cache metadata checksum/fingerprint/completion mismatch: {path}")
    expected = {"schema_version": 1, "source_fingerprint": audit["fingerprint"],
                "source_audit_sha256": audit["_audit_sha256"], "source_artifact_sha256": audit["artifact_sha256"],
                "variant": variant, "runtime": provenance.get("runtime"),
                "implementation_sha256": certificate["old_data_sha256"],
                "transform": {"duration_seconds": 10, "input_units": "mV", "output_dtype": "float32",
                              "filter_design_order": 4 if variant["filter"] != "minimal" else None,
                              "filter_application": "full-record SOS forward-backward" if variant["filter"] != "minimal" else "none",
                              "resampler": "scipy.signal.resample_poly/kaiser5/constant-pad" if variant["fs"] != 500 else "none"},
                "official_catalog_sha256": None}
    if (provenance != expected or not isinstance(provenance.get("runtime"), dict) or
            metadata.get("variant") != variant or metadata.get("source_fingerprint") != audit["fingerprint"]):
        raise ValueError(f"Old prepared variant/source/provenance mismatch: {path}")
    if path.parent.name != f"{variant['id']}-{metadata['fingerprint'][:16]}":
        raise ValueError("Old cache folder does not match its fingerprint")
    splits = metadata.get("splits", {})
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Old cache split metadata missing")
    for split, entry in splits.items():
        if entry.get("signals") != f"{split}.npy" or entry.get("manifest") != f"{split}.csv":
            raise ValueError("Old cache split filename mismatch")
        for name in ("signals_sha256", "manifest_sha256"):
            bootstrap._sha_value(entry.get(name), name)
    return metadata


def _job_files(directory, binding, *, full):
    if _json(directory / "identity.json") != binding:
        raise ValueError(f"Job identity mismatch: {directory.name}")
    marker = _json(directory / "completed.json")
    if (set(marker) != {"schema_version", "identity_hash", "artifact_sha256", "completed_at"} or
            marker["schema_version"] != 1 or marker["identity_hash"] != protocol_hash(binding) or
            set(marker["artifact_sha256"]) != set(ARTIFACTS)):
        raise ValueError(f"Completed job identity/artifact manifest mismatch: {directory.name}")
    for name, digest in marker["artifact_sha256"].items():
        bootstrap._sha_value(digest, name)
        source = _regular(directory / name)
        if full and file_hash(source) != digest:
            raise ValueError(f"Completed artifact SHA256 checksum mismatch: {source}")
    result = _json(directory / "result.json")
    job = binding["job"]
    expected = {"status": "completed", "job_id": job["id"], "experiment_id": job["experiment_id"],
                "seed": job["seed"], "identity_hash": protocol_hash(binding),
                "epochs_completed": job["training"]["epochs"]}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Completed result job/identity/configuration mismatch: {directory.name}")
    return marker, result


def _inspect(old, work, release, raw, certificate_path, contract_path, config_path):
    certificate = _certificate(certificate_path)
    base, contract = bootstrap._load_inputs(config_path, contract_path)
    files = _tree(old)
    _source_closed(old, files)
    protocol = load_protocol(_regular(old / "protocol.json"))
    if load_protocol(_regular(old / "target_protocol.json")) != protocol:
        raise ValueError("Old protocol and target_protocol disagree")
    if bootstrap._scientific_protocol(protocol) != bootstrap._scientific_protocol(base):
        raise ValueError("Old scientific protocol differs from the certified base configuration")
    if _json(old / "cohort_contract.json") != contract:
        raise ValueError("Old cohort contract differs")
    identity = {"protocol_hash": protocol_hash(protocol), "source_fingerprint": protocol["source"]["expected_fingerprint"],
                "code_fingerprint": certificate["old_code_fingerprint"]}
    if _json(old / "identity.json") != identity:
        raise ValueError("Old suite identity does not match the certified source code/protocol")
    jobs = build_jobs(protocol)
    imported = [j for j in jobs if j["variant"]["source"] == "release500"]
    missing = [j for j in jobs if j["variant"]["source"] == "official100"]
    if len(imported) != 18 or len(missing) != 3 or any(j["experiment_id"] != protocol["reference_experiment"] for j in missing):
        raise ValueError("This recovery requires exactly 18 release500 and 3 reference jobs")
    _incident(old, missing)
    actual_runs = set(p.name for p in (old / "runs").iterdir()) if (old / "runs").is_dir() else set()
    if actual_runs != {j["id"] for j in imported}:
        raise ValueError("Source must contain exactly all 18 completed non-reference run IDs")
    audit = _json(release / "audit.json")
    audit["_audit_sha256"] = file_hash(release / "audit.json")
    if (audit.get("fingerprint") != identity["source_fingerprint"] or
            audit["_audit_sha256"] != protocol["source"]["expected_audit_sha256"]):
        raise ValueError("Release source/audit anchors differ from old protocol")
    caches = {}
    variants = {j["variant"]["id"]: j["variant"] for j in imported}
    for path in (old / "cache").glob("*/variant.json"):
        candidate = _json(path)
        variant_id = candidate.get("variant", {}).get("id")
        if variant_id in variants:
            if variant_id in caches:
                raise ValueError(f"Duplicate completed old cache for variant {variant_id}")
            metadata = _old_metadata(path, variants[variant_id], certificate, audit)
            snapshot = old / "prepared_metadata" / f"{variant_id}.json"
            if snapshot.exists() and _json(snapshot) != metadata:
                raise ValueError("Old prepared metadata snapshot disagrees with cache")
            caches[variant_id] = {"path": path.parent, "metadata": metadata}
        elif candidate.get("ready") is True:
            raise ValueError("Unexpected completed official/unknown cache in source")
    if set(caches) != set(variants):
        raise ValueError("Missing completed release500 cache variant")
    bindings, runtimes = {}, []
    for job in imported:
        directory = old / "runs" / job["id"]
        binding = _json(directory / "identity.json")
        expected = {"schema_version": 1, "job": job, "runtime": binding.get("runtime"),
                    "identity": dict(identity, prepared_fingerprint=caches[job["variant"]["id"]]["metadata"]["fingerprint"])}
        if binding != expected or not isinstance(binding["runtime"], dict):
            raise ValueError(f"Old job binding/configuration mismatch: {job['id']}")
        _job_files(directory, binding, full=False)
        bindings[job["id"]] = binding
        runtimes.append(binding["runtime"])
    if any(runtime != runtimes[0] for runtime in runtimes):
        raise ValueError("Source job runtimes disagree")
    original_proof = _json(old / _PROOF) if (old / _PROOF).exists() else None
    if original_proof is not None:
        expected = {"schema_version": 1, "base_protocol_sha256": protocol_hash(base),
                    "cohort_contract_sha256": protocol_hash(contract),
                    "current_release": {"fingerprint": identity["source_fingerprint"], "audit_sha256": audit["_audit_sha256"]}}
        if any(original_proof.get(k) != v for k, v in expected.items()):
            raise ValueError("Original bootstrap proof/configuration differs")
    return dict(old=old, work=work, release=release, raw=raw, certificate=certificate,
                certificate_sha256=file_hash(certificate_path), protocol=protocol, contract=contract,
                identity=identity, imported=imported, missing=missing, caches=caches,
                bindings=bindings, runtime=runtimes[0], original_bootstrap_proof=original_proof)


def _inventory(context, *, official=False):
    """Expected source bytes, before touching the destination; metadata only."""
    old = context["old"]
    inputs, copies = {}, {}
    def add(path, digest=None, destination=None):
        path = _regular(path)
        relative = path.relative_to(old).as_posix()
        inputs[relative] = {"sha256": digest or file_hash(path), "size": path.stat().st_size}
        if destination is not None:
            copies[destination] = relative
    for name in ("protocol.json", "target_protocol.json", "identity.json", "cohort_contract.json", "errors.json"):
        add(old / name)
    if (old / _PROOF).exists():
        add(old / _PROOF)
    for variant_id, cache in context["caches"].items():
        directory, metadata = cache["path"], cache["metadata"]
        new_metadata = copy.deepcopy(metadata)
        new_metadata["provenance"]["implementation_sha256"] = context["certificate"]["new_data_sha256"]
        new_metadata["fingerprint"] = protocol_hash(new_metadata["provenance"])
        new_metadata["metadata_sha256"] = protocol_hash({k: v for k, v in new_metadata.items() if k != "metadata_sha256"})
        cache["new_metadata"] = new_metadata
        cache["destination"] = f"cache/{variant_id}-{new_metadata['fingerprint'][:16]}"
        add(directory / "variant.json")
        snapshot = old / "prepared_metadata" / f"{variant_id}.json"
        if snapshot.exists():
            add(snapshot)
        for split, item in metadata["splits"].items():
            for field in ("signals", "manifest"):
                add(directory / item[field], item[field + "_sha256"], f"{cache['destination']}/{item[field]}")
    for job in context["imported"]:
        directory = old / "runs" / job["id"]
        marker = _json(directory / "completed.json")
        for name in ("identity.json", "completed.json"):
            add(directory / name)
        for name, digest in marker["artifact_sha256"].items():
            destination = f"runs/{job['id']}/{name}" if name not in {"best.pt", "last.pt", "result.json"} else None
            add(directory / name, digest, destination)
    official_count = 0
    source100 = old / "cache/source100"
    if source100.exists():
        # The catalog is pinned by data.py; plan reads metadata/catalog only.
        catalog_path = _regular(context["raw"] / "SHA256SUMS.txt")
        expected_catalog = context["contract"]["checksum_catalog_sha256"]
        if file_hash(catalog_path) != expected_catalog:
            raise ValueError("Official100 source catalog SHA256 mismatch")
        catalog = {}
        for line in catalog_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError("Invalid official catalog syntax")
            bootstrap._sha_value(fields[0].lower(), "catalog digest")
            name = fields[1].lstrip("*").removeprefix("./")
            if name in catalog:
                raise ValueError("Duplicate official catalog entry")
            catalog[name] = fields[0].lower()
        if official:
            from ptbxl_experiments.data import _catalog
            if _catalog(context["raw"]) != catalog:
                raise ValueError("Pinned official catalog differs")
        with (context["release"] / "records.csv").open(encoding="utf-8-sig", newline="") as handle:
            required = [row["filename_lr"] for row in csv.DictReader(handle) if row["status"] == "accepted"]
        for relative in required:
            if not re.fullmatch(r"records100/[0-9]{5}/[0-9]{5}_lr", relative):
                raise ValueError("Invalid official100 source path in release manifest")
            for extension in (".hea", ".dat"):
                name = relative + extension
                source = source100 / name
                if name not in catalog:
                    raise ValueError(f"Official catalog lacks required source: {name}")
                if source.exists():
                    add(source, catalog[name], f"cache/source100/{name}")
                    official_count += 1
    return inputs, copies, official_count


def _headroom(context):
    protocol = context["protocol"]
    sample_bytes = protocol["source"]["expected_accepted"] * 12 * 10 * 4
    all_cache_bytes = sample_bytes * sum(v["fs"] for v in protocol["variants"])
    imported_array_bytes = sample_bytes * sum(v["fs"] for v in protocol["variants"] if v["source"] == "release500")
    # Match unchanged suite._disk_budget after importing the release500 arrays.
    # Its 15% overhead covers ALL variants, including arrays already imported.
    # Ignoring CSV/metadata/source100 size here keeps this allowance conservative.
    return max(0, int(all_cache_bytes * 1.15 + _GIB) - imported_array_bytes) + 8 * _GIB


def _ownership_binding(context, inputs):
    return {"recovery_id": RECOVERY_ID, "schema_version": 1,
            **{f"{key}_dir": str(context[key]) for key in ("release", "raw")},
            "old_work_dir": str(context["old"]), "work_dir": str(context["work"]),
            "certificate_sha256": context["certificate_sha256"], "certificate": context["certificate"],
            "old_identity": context["identity"], "protocol": context["protocol"],
            "cohort_contract_sha256": protocol_hash(context["contract"]), "input_files": inputs}


def plan(*, old_work_dir, work_dir, release_dir, raw_dir, certificate_path, contract_path,
         config_path=ROOT / "configs/ptbxl_experiments.json"):
    """No locks, mkdir, Torch, array reads, downloads or checkpoint loading."""
    paths = _paths(old_work_dir, work_dir, release_dir, raw_dir)
    context = _inspect(*paths, certificate_path, contract_path, config_path)
    inputs, _, count = _inventory(context)
    owned = _guard_destination(context, _ownership_binding(context, inputs))
    copy_bytes = sum(item["size"] for name, item in inputs.items()
                     if name.startswith(("cache/", "runs/")))
    return {"recovery_id": RECOVERY_ID, "source": str(paths[0]), "destination": str(paths[1]),
            "imported_job_ids": [j["id"] for j in context["imported"]],
            "remaining_job_ids": [j["id"] for j in context["missing"]], "copy_bytes_estimate": copy_bytes,
            "headroom_bytes": _headroom(context), "extra_disk_bytes_estimate": copy_bytes + _headroom(context),
            "official100_files_to_copy": count, "certificate_sha256": context["certificate_sha256"],
            "old_identity": context["identity"], "new_code_fingerprint": context["certificate"]["new_code_fingerprint"],
            "destination_exists": paths[1].exists(), "destination_owned": owned is not None,
            "model_execution": False, "array_verification": False,
            "findings": ["18 completed job envelopes found; full byte/payload/runtime checks run during migrate",
                         "Source test remains closed", "Original files stay intact; suite.lock coordinates migration"]}


def _target_runtime():
    from ptbxl_experiments import training
    return training._runtime(training._device("cuda"))


class _TorchCodec:
    def __init__(self):
        import torch
        from ptbxl_experiments import training
        self.torch = torch
        self.verify_completed = training._verify_completed
        if tuple(training._ARTIFACTS) != ARTIFACTS:
            raise ValueError("Certified completed-artifact schema changed")

    def load(self, path):
        return self.torch.load(path, weights_only=True, map_location="cpu")

    def save(self, stream, value):
        self.torch.save(value, stream)

    def assert_equal(self, first, second):
        if type(first) is not type(second):
            raise ValueError("Checkpoint payload type changed")
        if isinstance(first, self.torch.Tensor):
            if (first.dtype != second.dtype or first.shape != second.shape or first.layout != second.layout or
                    first.requires_grad != second.requires_grad or not self.torch.equal(first, second)):
                raise ValueError("Checkpoint tensor invariance failed")
        elif isinstance(first, dict):
            if first.keys() != second.keys():
                raise ValueError("Checkpoint payload keys changed")
            for key in first:
                self.assert_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            if len(first) != len(second):
                raise ValueError("Checkpoint payload length changed")
            for a, b in zip(first, second):
                self.assert_equal(a, b)
        elif first != second:
            raise ValueError("Checkpoint payload value changed")


def _checkpoint_codec():
    return _TorchCodec()


def _atomic(path, writer):
    path = _unlinked(path)
    if path.exists():
        raise ValueError(f"Refusing overwrite of existing recovery output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.recovery-", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy(source, destination, expected):
    if file_hash(_regular(source)) != expected:
        raise ValueError(f"Source SHA256 checksum mismatch: {source}")
    if destination.exists():
        if file_hash(_regular(destination)) != expected:
            raise ValueError(f"Existing destination SHA256 checksum changed: {destination}")
        return
    def write(stream):
        with source.open("rb") as handle:
            shutil.copyfileobj(handle, stream, length=4 * 1024 * 1024)
    _atomic(destination, write)
    if file_hash(_regular(destination)) != expected:
        raise ValueError(f"Copied output SHA256 mismatch: {destination}")


def _sentinel(journal):
    return {"recovery_state": "incomplete-do-not-train", "recovery_id": RECOVERY_ID,
            "journal_sha256": protocol_hash(journal)}


def _new_identity(context):
    return dict(context["identity"], code_fingerprint=context["certificate"]["new_code_fingerprint"])


def _guard_destination(context, header):
    work = context["work"]
    files = _tree(work)
    if not files and (not work.exists() or not any(work.iterdir())):
        return None
    if not (work / _JOURNAL).is_file():
        raise ValueError("Nonempty destination has no recovery ownership journal; use a NEW directory")
    journal = _json(work / _JOURNAL)
    if set(journal) != {"binding", "created_at"} or journal["binding"] != header:
        raise ValueError("Destination recovery ownership/input binding changed; use a NEW directory")
    identity = _json(work / "identity.json") if (work / "identity.json").exists() else None
    if identity not in (None, _sentinel(journal), _new_identity(context)):
        raise ValueError("Destination identity differs from the owned migration")
    ready = (work / _PROOF).is_file()
    if identity == _new_identity(context) and not ready:
        raise ValueError("Destination published identity has no migration proof")
    if not ready or identity != _new_identity(context):
        if any((work / name).exists() for name in ("selection.json", "evaluation", "status.json")):
            raise ValueError("Unpublished recovery destination contains suite/test evidence")
        expected_jobs = {j["id"] for j in context["imported"]}
        if (work / "runs").exists() and not {p.name for p in (work / "runs").iterdir()}.issubset(expected_jobs):
            raise ValueError("Unpublished recovery contains unexpected/reference jobs")
    else:
        expected_jobs = {j["id"] for j in context["imported"] + context["missing"]}
        if (work / "runs").exists() and not {p.name for p in (work / "runs").iterdir()}.issubset(expected_jobs):
            raise ValueError("Destination contains an unknown run ID")
    return journal


def _disk(context, inputs, copies, codec):
    work = context["work"]
    total = sum(item["size"] for name, item in inputs.items() if name.startswith(("cache/", "runs/")))
    verified_existing = 0
    for destination, source in copies.items():
        path = work / destination
        if path.exists():
            if file_hash(_regular(path)) != inputs[source]["sha256"]:
                raise ValueError(f"Existing destination checksum changed: {path}")
            verified_existing += inputs[source]["size"]
    # Checkpoints change serialization bytes, so grant resume disk credit only
    # after comparing their complete rebound CPU payload against the source.
    for job in context["imported"]:
        binding = context["bindings"][job["id"]]
        cache = context["caches"][job["variant"]["id"]]
        new_binding = copy.deepcopy(binding)
        new_binding["identity"].update(code_fingerprint=context["certificate"]["new_code_fingerprint"],
                                       prepared_fingerprint=cache["new_metadata"]["fingerprint"])
        for name in ("best.pt", "last.pt"):
            relative = f"runs/{job['id']}/{name}"
            destination = work / relative
            if destination.exists():
                source = codec.load(_regular(context["old"] / relative))
                _checkpoint_schema(source, binding, name)
                copied = codec.load(_regular(destination))
                _checkpoint_schema(copied, new_binding, name)
                codec.assert_equal(dict(source, binding=new_binding), copied)
                verified_existing += inputs[relative]["size"]
    ancestor = work
    while not ancestor.exists():
        ancestor = ancestor.parent
    required = max(0, total - verified_existing) + _headroom(context)
    free = shutil.disk_usage(ancestor).free
    if free < required:
        raise OSError(f"Insufficient target disk: {free / _GIB:.2f} GiB free; need {required / _GIB:.2f} GiB "
                      "including official100 cache and at least 8 GiB checkpoint/results headroom")
    return {"required_free_bytes": required, "verified_existing_copy_bytes": verified_existing}


def _cache(context, cache, inputs, copies):
    from ptbxl_experiments import data
    destination = context["work"] / cache["destination"]
    for output, source in copies.items():
        if output.startswith(cache["destination"] + "/"):
            _copy(context["old"] / source, context["work"] / output, inputs[source]["sha256"])
    freeze_json(destination / "variant.json", cache["new_metadata"])
    metadata = data.verify_prepared(destination)
    for split in data.SPLITS:
        actual = data._frame(destination / f"{split}.csv", accepted=True)
        expected = data._frame(context["release"] / f"{split}.csv", accepted=True)
        if set(actual.columns) != set(expected.columns):
            raise ValueError("Prepared/release manifest columns differ")
        data._aligned(actual, expected, list(expected.columns), f"recovery {split}")
    freeze_json(context["work"] / "prepared_metadata" / f"{metadata['variant']['id']}.json", metadata)
    return {"old_path": str(cache["path"]), "new_path": str(destination),
            "old_metadata": cache["metadata"], "new_metadata": metadata,
            "old_metadata_sha256": file_hash(cache["path"] / "variant.json"),
            "new_metadata_sha256": file_hash(destination / "variant.json")}


def _validate_predictions(directory, cache_path):
    import numpy as np
    from ptbxl_experiments import data
    from ptbxl_experiments.suite import load_predictions
    prediction = load_predictions(directory / "validation_predictions.npz")
    frame = data._frame(cache_path / "validation.csv", accepted=True)
    expected = {"ecg_id": frame.ecg_id.to_numpy(), "patient_id": frame.patient_id.to_numpy(dtype=str),
                "y_true": frame[list(data.LABEL_COLUMNS)].to_numpy()}
    if any(not np.array_equal(prediction[key], value) for key, value in expected.items()):
        raise ValueError("Validation prediction label/patient/cohort alignment mismatch")
    threshold = np.load(directory / "val_thresholds.npy", allow_pickle=False)
    if (threshold.shape != (5,) or not np.isfinite(threshold).all() or
            ((threshold < 0) | (threshold > np.nextafter(1., np.inf))).any()):
        raise ValueError("Validation thresholds invalid")


def _checkpoint_schema(payload, binding, name):
    keys = ({"schema_version", "binding", "epoch", "score", "model"} if name == "best.pt" else
            {"schema_version", "binding", "epoch", "model", "optimizer", "scheduler", "scaler", "rng",
             "history", "best", "elapsed_seconds", "peak_gpu_memory_bytes"})
    if (not isinstance(payload, dict) or set(payload) != keys or payload["schema_version"] != 1 or
            payload["binding"] != binding):
        raise ValueError(f"Checkpoint schema/binding mismatch: {name}")


def _import_job(context, job, codec, inputs, copies):
    old = context["old"] / "runs" / job["id"]
    destination = context["work"] / "runs" / job["id"]
    cache = context["caches"][job["variant"]["id"]]
    binding = context["bindings"][job["id"]]
    marker, result = _job_files(old, binding, full=True)
    codec.verify_completed(old, binding)
    _validate_predictions(old, context["work"] / cache["destination"])
    new_binding = copy.deepcopy(binding)
    new_binding["identity"].update(code_fingerprint=context["certificate"]["new_code_fingerprint"],
                                   prepared_fingerprint=cache["new_metadata"]["fingerprint"])
    freeze_json(destination / "identity.json", new_binding)
    for output, source in copies.items():
        if output.startswith(f"runs/{job['id']}/"):
            _copy(context["old"] / source, context["work"] / output, inputs[source]["sha256"])
    last = codec.load(old / "last.pt")
    best = codec.load(old / "best.pt")
    _checkpoint_schema(last, binding, "last.pt")
    _checkpoint_schema(best, binding, "best.pt")
    if (last["epoch"] != job["training"]["epochs"] or result["best_epoch"] != best["epoch"] or
            not isinstance(last["best"], dict) or set(last["best"]) != {"epoch", "score", "model"} or
            not 1 <= best["epoch"] <= last["epoch"]):
        raise ValueError("Completed checkpoint epoch/best-state mismatch")
    codec.assert_equal({k: best[k] for k in ("epoch", "score", "model")}, last["best"])
    codec.assert_equal(last["history"], _json(old / "history.json"))
    for name, payload in (("best.pt", best), ("last.pt", last)):
        rebound = dict(payload, binding=new_binding)
        path = destination / name
        if not path.exists():
            _atomic(path, lambda stream: codec.save(stream, rebound))
        reloaded = codec.load(_regular(path))
        _checkpoint_schema(reloaded, new_binding, name)
        codec.assert_equal(rebound, reloaded)
        # Replacing only binding is deliberately explicit and checked after reload.
        codec.assert_equal({k: v for k, v in payload.items() if k != "binding"},
                           {k: v for k, v in reloaded.items() if k != "binding"})
    new_result = dict(result, identity_hash=protocol_hash(new_binding), run_dir=str(destination))
    freeze_json(destination / "result.json", new_result)
    new_marker = dict(marker, identity_hash=protocol_hash(new_binding),
                      artifact_sha256={name: file_hash(_regular(destination / name)) for name in ARTIFACTS})
    freeze_json(destination / "completed.json", new_marker)
    _job_files(destination, new_binding, full=True)
    if codec.verify_completed(destination, new_binding) != new_result:
        raise ValueError("Normal completed-artifact validation did not accept imported job")
    return {"old_binding": binding, "new_binding": new_binding,
            "old_artifact_sha256": marker["artifact_sha256"], "new_artifact_sha256": new_marker["artifact_sha256"],
            "old_completed_sha256": file_hash(old / "completed.json"),
            "new_completed_sha256": file_hash(destination / "completed.json"),
            "checkpoint_payload_invariance": {"best.pt": True, "last.pt": True},
            "original_completed_at": marker["completed_at"]}


def _publish(work, journal, identity):
    current = _json(work / "identity.json")
    if current == identity:
        return
    if current != _sentinel(journal) or not (work / _PROOF).is_file():
        raise ValueError("Recovery identity is not the owned publication sentinel")
    write_json(work / "identity.json", identity)


def migrate(*, old_work_dir, work_dir, release_dir, raw_dir, certificate_path, contract_path,
            config_path=ROOT / "configs/ptbxl_experiments.json"):
    """School-only CPU checkpoint conversion, with exact CUDA runtime comparison.

    No model construction, optimizer step, inference, acquisition or preparation.
    The only source write is the standard suite lock. Existing outputs are never
    overwritten, except the owned refusal identity at final publication.
    """
    old, work, release, raw = _paths(old_work_dir, work_dir, release_dir, raw_dir)
    _check_lock(old / "suite.lock")
    with SuiteLock(old / "suite.lock"):
        context = _inspect(old, work, release, raw, certificate_path, contract_path, config_path)
        inputs, copies, official_count = _inventory(context, official=True)
        header = _ownership_binding(context, inputs)
        prior = _guard_destination(context, header)
        # Lock conflicts occur outside the diagnostic handler: another owner's
        # directory/logs must never be changed by a failed lock contender.
        work.mkdir(parents=True, exist_ok=True)
        _check_lock(work / "suite.lock")
        with SuiteLock(work / "suite.lock"):
            # Recheck after acquiring the lock; preserve an already owned journal.
            if prior is not None:
                prior = _guard_destination(context, header)
            elif any(path.name != "suite.lock" for path in work.iterdir()):
                raise ValueError("Destination changed before its lock was acquired; use a NEW directory")
            journal = prior or {"binding": header, "created_at": utc_now()}
            freeze_json(work / _JOURNAL, journal)
            if not (work / "identity.json").exists():
                freeze_json(work / "identity.json", _sentinel(journal))
            try:
                print("Recovery: verifying target runtime, all source bytes and release cohort", flush=True)
                if _target_runtime() != context["runtime"]:
                    raise ValueError("Target CUDA/dependency runtime differs from source jobs")
                from ptbxl_experiments import data
                for cache in context["caches"].values():
                    if cache["metadata"]["provenance"]["runtime"] != data._runtime():
                        raise ValueError("Prepared-cache dependency runtime differs from target")
                if context["original_bootstrap_proof"] is not None:
                    if context["original_bootstrap_proof"].get("runtime") != bootstrap._runtime():
                        raise ValueError("Original bootstrap runtime differs from target")
                codec = _checkpoint_codec()
                _disk(context, inputs, copies, codec)
                for name, expected in inputs.items():
                    if file_hash(_regular(old / name)) != expected["sha256"]:
                        raise ValueError(f"Source artifact SHA256 checksum mismatch: {name}")
                verified_release = bootstrap.verify_equivalent_release(release, context["contract"], full=True)
                if (verified_release["source_fingerprint"] != context["identity"]["source_fingerprint"] or
                        verified_release["audit_sha256"] != context["protocol"]["source"]["expected_audit_sha256"]):
                    raise ValueError("Full release verification differs from protocol anchors")
                cache_reports = {key: _cache(context, value, inputs, copies) for key, value in context["caches"].items()}
                for output, source in copies.items():
                    if output.startswith("cache/source100/"):
                        _copy(old / source, work / output, inputs[source]["sha256"])
                job_reports = {}
                for number, job in enumerate(context["imported"], 1):
                    print(f"Recovery: verify/import {number}/18 {job['id']}", flush=True)
                    job_reports[job["id"]] = _import_job(context, job, codec, inputs, copies)
                # Input changes mid-copy cannot yield a ready proof.
                for name, expected in inputs.items():
                    if file_hash(_regular(old / name)) != expected["sha256"]:
                        raise ValueError(f"Source changed during migration: {name}")
                _certificate(certificate_path)
                for name, value in (("target_protocol.json", context["protocol"]), ("protocol.json", context["protocol"]),
                                    ("cohort_contract.json", context["contract"])):
                    freeze_json(work / name, value)
                proof = {"schema_version": 1, "kind": "certified-official100-recovery", "recovery_id": RECOVERY_ID,
                         "created_at": journal["created_at"], "journal_sha256": protocol_hash(journal),
                         "certificate": context["certificate"], "certificate_sha256": context["certificate_sha256"],
                         "old_identity": context["identity"], "new_identity": _new_identity(context),
                         "original_bootstrap_proof": context["original_bootstrap_proof"],
                         "verified_release": verified_release, "caches": cache_reports, "jobs": job_reports,
                         "copied_files": {output: inputs[source] for output, source in copies.items()},
                         "official100_files_copied": official_count,
                         "imported_job_ids": [j["id"] for j in context["imported"]],
                         "remaining_job_ids": [j["id"] for j in context["missing"]]}
                freeze_json(work / _PROOF, proof)
                _publish(work, journal, _new_identity(context))
                return work / "target_protocol.json"
            except (Exception, KeyboardInterrupt) as exc:
                write_json(work / "bootstrap_error.json", {"state": "failed", "recovery_id": RECOVERY_ID,
                           "updated_at": utc_now(), "exception": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc(),
                           "action": "Keep source and destination evidence. Correct the stated prerequisite and repeat recovery; "
                                     "changed identities or corrupt outputs require a NEW destination."})
                raise


def run(*, allow_download=False, **kwargs):
    target = migrate(**kwargs)
    from ptbxl_experiments.suite import run_suite
    return run_suite(target, kwargs["release_dir"], kwargs["raw_dir"], kwargs["work_dir"],
                     device="cuda", allow_download=allow_download)
