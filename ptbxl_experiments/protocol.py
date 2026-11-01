"""Frozen protocol and validation-only selection; standard library only.

Intent: make experiment changes explicit and prevent partial-suite test selection.
Status: offline fixture validation; target run is a separate gate.
Next: inspect target evidence, not test-tune this protocol. Blockers: none known.
Confidence: invariants are fixture-tested; model quality remains unmeasured.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
_ID = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        actual = set(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{name}: exact keys required {expected}; got {actual}")


def _integer(value, minimum, name):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, minimum, maximum, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite in [{minimum}, {maximum}]")


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"Unsafe or invalid identifier: {value!r}")


def _unique(items, name):
    if len(items) != len(set(items)):
        raise ValueError(f"Duplicate {name}")


def validate_protocol(config):
    _keys(config, "schema_version suite_id source seeds reference_experiment variants experiments training evaluation", "protocol")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported schema_version")
    _identifier(config["suite_id"])
    src = config["source"]
    _keys(src, "version expected_fingerprint expected_audit_sha256 expected_records expected_accepted", "source")
    if src["version"] != "1.0.3":
        raise ValueError("Only the specified PTB-XL1.0.3 source is supported")
    for key in ("expected_fingerprint", "expected_audit_sha256"):
        if not isinstance(src[key], str) or not _SHA.fullmatch(src[key]):
            raise ValueError(f"source.{key} must be SHA256")
    for key in ("expected_records", "expected_accepted"):
        _integer(src[key], 1, key)
    if src["expected_accepted"] > src["expected_records"]:
        raise ValueError("Accepted count exceeds source")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a nonempty list")
    for seed in seeds:
        _integer(seed, 0, "seed")
        if seed >= 2**32:
            raise ValueError("seed must fit uint32")
    _unique(seeds, "seeds")
    variants = config["variants"]
    if not isinstance(variants, list) or not variants:
        raise ValueError("variants must be nonempty list")
    for v in variants:
        _keys(v, "id source fs filter", "variant")
        _identifier(v["id"])
        if v["source"] not in ("official100", "release500") or type(v["fs"]) is not int or v["fs"] not in (100, 200, 500):
            raise ValueError("Unsupported signal source/rate")
        if v["filter"] not in ("minimal", "bandpass_0.5_40"):
            raise ValueError("Unsupported filter")
        if v["source"] == "official100" and (v["fs"] != 100 or v["filter"] != "minimal"):
            raise ValueError("official100 must be100Hz/minimal")
        if v["filter"] != "minimal" and v["fs"] != 500:
            raise ValueError("This protocol defines bandpass at500Hz only")
    _unique([v["id"] for v in variants], "variant IDs")
    _unique([(v["source"], v["fs"], v["filter"]) for v in variants], "physical variants")
    variant_map = {v["id"]: v for v in variants}
    experiments = config["experiments"]
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments must be nonempty list")
    for e in experiments:
        _keys(e, "id variant normalization", "experiment")
        _identifier(e["id"])
        if e["variant"] not in variant_map or e["normalization"] not in ("global", "per_lead"):
            raise ValueError("Unknown variant or normalization")
    _unique([e["id"] for e in experiments], "experiment IDs")
    _unique([(e["variant"], e["normalization"]) for e in experiments], "experiment conditions")
    if set(variant_map) != {e["variant"] for e in experiments}:
        raise ValueError("Every variant must be used")
    reference = next((e for e in experiments if e["id"] == config["reference_experiment"]), None)
    if reference is None or reference["normalization"] != "global" or variant_map[reference["variant"]]["source"] != "official100":
        raise ValueError("Reference must be official100 with global normalization")
    tr = config["training"]
    _keys(tr, "architecture epochs batch_size learning_rate weight_decay crop_seconds eval_stride_seconds aggregation workers amp scheduler kernel_policy", "training")
    for key, value in {"architecture": "resnet1d_wang", "aggregation": "max", "scheduler": "onecycle", "kernel_policy": "scale_odd"}.items():
        if tr[key] != value:
            raise ValueError(f"Unsupported training.{key}")
    _integer(tr["epochs"], 2, "epochs")
    _integer(tr["batch_size"], 2, "batch_size")
    _integer(tr["workers"], 0, "workers")
    if type(tr["amp"]) is not bool:
        raise ValueError("amp must be boolean")
    _number(tr["learning_rate"], 1e-8, 1, "learning_rate")
    _number(tr["weight_decay"], 0, 1, "weight_decay")
    _number(tr["crop_seconds"], .5, 10, "crop_seconds")
    _number(tr["eval_stride_seconds"], .01, tr["crop_seconds"], "eval_stride_seconds")
    for v in variants:
        for field in ("crop_seconds", "eval_stride_seconds"):
            samples = tr[field] * v["fs"]
            if not math.isclose(samples, round(samples), abs_tol=1e-9):
                raise ValueError(f"{field} must be integer samples at each rate")
    _keys(config["evaluation"], "bootstrap_replicates bootstrap_seed", "evaluation")
    _integer(config["evaluation"]["bootstrap_replicates"], 1, "bootstrap_replicates")
    _integer(config["evaluation"]["bootstrap_seed"], 0, "bootstrap_seed")
    return config


def load_protocol(path: Path) -> dict:
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8-sig") as handle:
        config = json.load(handle, object_pairs_hook=reject_duplicate_keys)
    return validate_protocol(config)


def protocol_hash(protocol: dict) -> str:
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_jobs(protocol: dict) -> list[dict]:
    validate_protocol(protocol)
    variants = {v["id"]: v for v in protocol["variants"]}
    return [dict(id=f"{e['id']}__seed{seed}", experiment_id=e["id"], seed=seed,
                 variant=copy.deepcopy(variants[e["variant"]]), normalization=e["normalization"],
                 training=copy.deepcopy(protocol["training"]))
            for e in protocol["experiments"] for seed in protocol["seeds"]]


def select_experiment(protocol, results: list[dict]) -> dict:
    jobs = {j["id"]: j for j in build_jobs(protocol)}
    ids = [r.get("job_id") for r in results]
    if len(ids) != len(set(ids)) or set(ids) != set(jobs):
        raise ValueError("Selection requires exactly one result for every planned job")
    grouped = {e["id"]: [] for e in protocol["experiments"]}
    for r in results:
        j = jobs[r["job_id"]]
        if r.get("status") != "completed" or r.get("experiment_id") != j["experiment_id"] or r.get("seed") != j["seed"]:
            raise ValueError(f"Incomplete or mismatched result: {j['id']}")
        vals = r.get("val_metrics", {})
        if vals.get("valid_auroc_classes", len(CLASSES)) != len(CLASSES):
            raise ValueError(f"Selection needs defined AUROC for all five classes: {j['id']}")
        for metric in ("macro_auroc", "macro_average_precision"):
            _number(vals.get(metric), 0, 1, f"{j['id']}.{metric}")
        grouped[j["experiment_id"]].append({"seed": j["seed"], "job_id": j["id"],
                                            "macro_auroc": vals["macro_auroc"],
                                            "macro_average_precision": vals["macro_average_precision"]})
    rows = []
    for exp, values in grouped.items():
        auc = [v["macro_auroc"] for v in values]
        ap = [v["macro_average_precision"] for v in values]
        rows.append(dict(experiment_id=exp, mean_macro_auroc=statistics.mean(auc),
                         std_macro_auroc=statistics.stdev(auc) if len(auc) > 1 else 0.,
                         mean_macro_average_precision=statistics.mean(ap),
                         std_macro_average_precision=statistics.stdev(ap) if len(ap) > 1 else 0.,
                         seeds=sorted(values, key=lambda x: x["seed"])))
    rows.sort(key=lambda x: (-x["mean_macro_auroc"], -x["mean_macro_average_precision"], x["experiment_id"]))
    return {"selected_experiment": rows[0]["experiment_id"], "reference_experiment": protocol["reference_experiment"],
            "selection_rule": "mean_validation_macro_auroc_then_mean_macro_AP_then_lexical_ID",
            "all_jobs_completed": len(jobs), "rows": rows}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temp = Path(handle.name)
            json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
