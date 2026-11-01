"""Target suite orchestration with offline plan/summary/collection support.

Intent: freeze validation evidence before opening test; preserve resumable failures.
Status: fixture reviewed; CUDA is verified only by target preflight.
Next: bring back the results ZIP. Blockers: target-only checks cannot run on laptop.
Confidence: invariants have offline tests; end-to-end GPU execution is not yet verified.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile

from .protocol import (CLASSES, LEADS, build_jobs, code_fingerprint, load_protocol,
                       protocol_hash, select_experiment, write_json)

ROOT = Path(__file__).resolve().parents[1]
RUN_ARTIFACTS = {"identity.json", "status.json", "error.json", "history.json", "history.csv",
                 "result.json", "validation_predictions.npz", "val_thresholds.npy", "last.pt",
                 "best.pt", "completed.json"}
SUITE_ARTIFACTS = {"protocol.json", "identity.json", "environment.json", "preflight.json", "status.json",
                   "error.json", "errors.json", "heartbeat.json", "selection.json", "validation_comparison.json",
                   "validation_comparison.csv", "naive_validation.json", "SUMMARY.vi.md", "suite.log",
                   "target-tests.log", "target-tests.json", "target_protocol.json", "cohort_contract.json",
                   "bootstrap_verification.json", "bootstrap_error.json", "bootstrap.log"}
EVAL_ARTIFACTS = {"test_predictions.npz", "prediction_binding.json", "test_metrics.json",
                  "ensemble_test_predictions.npz", "ensemble_validation_predictions.npz",
                  "ensemble_metrics.json", "ensemble_thresholds.json"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_json(path, value):
    """Refuse changed evidence in a directory that may already have exposed test."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen evidence changed: {path}. Use a NEW work directory/protocol.")
    else:
        write_json(path, value)


def snapshot_prepared_metadata(work_dir, variant_id, metadata):
    """Retain train scalers and transform provenance with the portable checkpoints."""
    if not isinstance(variant_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", variant_id):
        raise ValueError("Invalid prepared metadata snapshot name")
    path = Path(work_dir) / "prepared_metadata" / f"{variant_id}.json"
    freeze_json(path, metadata)
    return path


class SuiteLock:
    """OS-owned byte/flock lock, released by the OS after process death; never unlink."""
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("Lock path cannot be a symlink")
        self.handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"Another process owns suite lock: {self.path}") from exc
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            try:
                self.handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()
                self.handle = None


class _Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile
        self.lock = threading.RLock()

    def write(self, text):
        with self.lock:
            self.terminal.write(text)
            self.logfile.write(text)
            self.logfile.flush()
        return len(text)

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.logfile.flush()

    def isatty(self):
        return False


class _Heartbeat:
    def __init__(self, work_dir):
        self.path = Path(work_dir) / "heartbeat.json"
        self.stage = "starting"
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="suite-heartbeat", daemon=True)

    def _loop(self):
        while not self.stop.wait(30):
            elapsed = time.monotonic() - self.started
            write_json(self.path, {"updated_at": utc_now(), "pid": os.getpid(), "stage": self.stage,
                                   "elapsed_seconds": elapsed})
            print(f"[heartbeat] {self.stage}; elapsed {elapsed / 60:.1f} min", flush=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        write_json(self.path, {"updated_at": utc_now(), "pid": os.getpid(), "stage": "stopped",
                               "elapsed_seconds": time.monotonic() - self.started})


def environment():
    packages = {}
    for name in ("numpy", "pandas", "scipy", "wfdb", "torch", "matplotlib", "requests", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def plan(config_path):
    protocol = load_protocol(config_path)
    return {"suite_id": protocol["suite_id"], "protocol_hash": protocol_hash(protocol),
            "jobs": build_jobs(protocol), "number_of_jobs": len(build_jobs(protocol)),
            "test_policy": "Only reference plus validation-selected condition, after all jobs complete",
            "model_execution": False}


def _validate_locations(release_dir, raw_dir, work_dir):
    release_dir, raw_dir, work_dir = (Path(p).resolve() for p in (release_dir, raw_dir, work_dir))
    for original in (release_dir, raw_dir):
        if work_dir == original or work_dir.is_relative_to(original) or original.is_relative_to(work_dir):
            raise ValueError("Work directory must be separate from raw/release roots, not parent/child")
    return release_dir, raw_dir, work_dir


def _disk_budget(protocol, work_dir):
    # Conservative allowance includes all caches, official100 source and checkpoints/results ZIP.
    full_caches = sum(protocol["source"]["expected_accepted"] * 12 * v["fs"] * 10 * 4
                      for v in protocol["variants"])
    existing = 0
    cache = Path(work_dir) / "cache"
    if cache.exists():
        existing = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file() and not p.is_symlink())
    remaining = max(0, int(full_caches * 1.15 + 1024**3) - existing)
    required = remaining + 8 * 1024**3
    free = shutil.disk_usage(work_dir).free
    if free < required:
        raise OSError(f"Insufficient work-volume space: {free / 1024**3:.1f} GiB free; "
                      f"need at least {required / 1024**3:.1f} GiB for this run (estimate).")
    return {"free_bytes": free, "required_free_bytes_estimate": required,
            "note": "8GiB checkpoint/result headroom plus remaining caches; verify actual target growth"}


def preflight(protocol, release_dir, raw_dir, work_dir, *, device="cuda"):
    from .data import verify_release
    if device != "cuda":
        raise ValueError("School suite requires CUDA; CPU fallback is intentionally disabled")
    release_dir, raw_dir, work_dir = _validate_locations(release_dir, raw_dir, work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use the Python3.12 .venv-train created by Setup-School.ps1")
    expected = {"numpy": "2.5.4", "pandas": "3.1.0", "scipy": "1.18.1", "wfdb": "4.3.1",
                "matplotlib": "3.11.2", "requests": "2.34.2", "pytest": "9.1.1"}
    mismatches = {k: {"expected": v, "found": env["packages"].get(k)} for k, v in expected.items()
                  if env["packages"].get(k) != v}
    if env["packages"].get("torch") != "2.14.1+cu126":
        mismatches["torch"] = {"expected": "2.14.1+cu126", "found": env["packages"].get("torch")}
    if mismatches:
        raise RuntimeError(f"Dependency mismatch; run Setup-School.ps1: {mismatches}")
    disk = _disk_budget(protocol, work_dir)
    if not (raw_dir / "SHA256SUMS.txt").is_file():
        raise FileNotFoundError("Missing raw SHA256SUMS.txt; transfer raw source with the release")
    print("[preflight] Full read-only receiving audit ...", flush=True)
    src = protocol["source"]
    verified = verify_release(release_dir, expected_fingerprint=src["expected_fingerprint"],
                              expected_audit_sha256=src["expected_audit_sha256"], full=True)
    if verified["accepted_records"] != src["expected_accepted"] or verified["source_records"] != src["expected_records"]:
        raise ValueError("Receiving audit cohort counts differ from frozen protocol")
    if tuple(verified["classes"]) != CLASSES or tuple(verified["leads"]) != LEADS:
        raise ValueError("Receiving classes/leads differ from protocol")
    from .training import target_smoke
    print("[preflight] CUDA synthetic shapes, largest batch/backward/checkpoint ...", flush=True)
    smoke = target_smoke(protocol["training"], device=device)
    report = {"checked_at": utc_now(), "environment": env, "release": verified, "disk": disk,
              "target_smoke": smoke}
    write_json(work_dir / "preflight.json", report)
    write_json(work_dir / "environment.json", env)
    return report


def _target_tests(work_dir):
    """Run gated tests in an isolated child; parent environment is never modified."""
    env = dict(os.environ)
    env["ECG_TARGET_TESTS"] = "1"
    env["PYTHONUTF8"] = "1"
    command = [sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", "tests",
               "-p", "test_experiment_target.py", "-v"]
    test_path = ROOT / "tests" / "test_experiment_target.py"
    if not test_path.is_file():
        raise FileNotFoundError("Target regression tests missing from bundle")
    print("[preflight] Running target-only model/crop/checkpoint-resume regression tests ...", flush=True)
    with (Path(work_dir) / "target-tests.log").open("w", encoding="utf-8") as logfile:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                logfile.write(line)
                logfile.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
    write_json(Path(work_dir) / "target-tests.json", {"exit_code": code, "checked_at": utc_now(),
                                                      "test_file_sha256": file_hash(test_path)})
    if code:
        raise RuntimeError(f"Target tests failed with exit code{code}; see target-tests.log. No training started.")


def load_predictions(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"ecg_id", "patient_id", "y_true", "probabilities"}:
            raise ValueError(f"Unexpected prediction schema: {path}")
        result = {name: saved[name] for name in saved.files}
    return _validate_predictions(result)


def _validate_predictions(pred):
    import numpy as np
    expected = {"ecg_id", "patient_id", "y_true", "probabilities"}
    if set(pred) != expected:
        raise ValueError("Prediction keys must carry IDs, labels and probabilities")
    pred = {k: np.asarray(v) for k, v in pred.items()}
    n = len(pred["ecg_id"])
    if n == 0 or pred["ecg_id"].shape != (n,) or pred["patient_id"].shape != (n,):
        raise ValueError("Invalid prediction IDs")
    if len(np.unique(pred["ecg_id"])) != n:
        raise ValueError("Duplicate ECG prediction IDs")
    if pred["y_true"].shape != (n, len(CLASSES)) or pred["probabilities"].shape != (n, len(CLASSES)):
        raise ValueError("Prediction label/probability shape mismatch")
    if not np.isin(pred["y_true"], [0, 1]).all() or not np.isfinite(pred["probabilities"]).all():
        raise ValueError("Nonbinary labels or nonfinite probabilities")
    if ((pred["probabilities"] < 0) | (pred["probabilities"] > 1)).any():
        raise ValueError("Probabilities outside [0,1]")
    if any(v.dtype.hasobject for v in pred.values()):
        raise ValueError("Object arrays are forbidden in predictions")
    return pred


def save_predictions(path, pred):
    import numpy as np
    pred = _validate_predictions(pred)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **pred)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def ensemble_predictions(predictions):
    import numpy as np
    if not predictions:
        raise ValueError("Empty ensemble")
    predictions = [_validate_predictions(p) for p in predictions]
    first = predictions[0]
    for pred in predictions[1:]:
        for key in ("ecg_id", "patient_id", "y_true"):
            if not np.array_equal(first[key], pred[key]):
                raise ValueError(f"Ensemble alignment mismatch: {key}")
    return dict(first, probabilities=np.mean([p["probabilities"] for p in predictions], axis=0))


def _check_common_cohort(prepared):
    from .data import load_prepared
    import numpy as np
    baseline = {}
    for variant, directory in prepared.items():
        for split in ("train", "validation", "test"):
            x, manifest, _, _, _ = load_prepared(directory, split, "global")
            del x
            aligned = manifest[["ecg_id", "patient_id", "split", *[f"y_{c}" for c in CLASSES]]].astype(str).to_numpy()
            if split in baseline and not np.array_equal(baseline[split], aligned):
                raise ValueError(f"Cohort/order/labels differ for {variant}:{split}")
            baseline.setdefault(split, aligned)


def _naive_validation(directory):
    from .data import load_prepared
    from .metrics import multilabel_metrics
    import numpy as np
    x, train, _, _, _ = load_prepared(directory, "train", "global")
    del x
    x, val, _, _, _ = load_prepared(directory, "validation", "global")
    del x
    labels = [f"y_{c}" for c in CLASSES]
    prevalence = train[labels].to_numpy().mean(axis=0)
    truth = val[labels].to_numpy()
    return {"fit_split": "train", "class_order": CLASSES, "prevalence": prevalence,
            "validation_metrics": multilabel_metrics(truth, np.broadcast_to(prevalence, truth.shape))}


def _validation_comparison(selection):
    rows = {r["experiment_id"]: r for r in selection["rows"]}
    planned = [("derived100", "reference100"), ("derived200", "derived100"),
               ("minimal500_global", "derived100"), ("minimal500_global", "derived200"),
               ("minimal500_per_lead", "minimal500_global"),
               ("bandpass500_global", "minimal500_global"),
               ("bandpass500_per_lead", "minimal500_per_lead")]
    contrasts = []
    for candidate, baseline in planned:
        if candidate not in rows or baseline not in rows:
            continue
        left = {r["seed"]: r for r in rows[candidate]["seeds"]}
        right = {r["seed"]: r for r in rows[baseline]["seeds"]}
        deltas = [{"seed": seed, "macro_auroc_delta": left[seed]["macro_auroc"] - right[seed]["macro_auroc"],
                   "macro_AP_delta": left[seed]["macro_average_precision"] - right[seed]["macro_average_precision"]}
                  for seed in sorted(left)]
        contrasts.append({"candidate": candidate, "baseline": baseline, "paired_seeds": deltas,
                          "mean_macro_auroc_delta": sum(d["macro_auroc_delta"] for d in deltas)/len(deltas),
                          "mean_macro_AP_delta": sum(d["macro_AP_delta"] for d in deltas)/len(deltas)})
    return {"ranking": selection["rows"], "planned_contrasts": contrasts,
            "interpretation": "Descriptive paired-seed validation differences; not significance tests. "
                              "Frequency comparisons also change scaled kernels/parameters/temporal stride."}


def _prediction_binding(selection, job, run_dir):
    if code_fingerprint() != selection["identity"]["code_fingerprint"]:
        raise ValueError("Source changed after validation selection; refusing test prediction")
    return {"selection_sha256": protocol_hash(selection), "job_id": job["id"],
            "best_checkpoint_sha256": file_hash(Path(run_dir) / "best.pt")}


def _evaluate(protocol, jobs, prepared, work_dir, selection, device):
    from .training import predict_run
    from .metrics import fit_thresholds, multilabel_metrics, patient_bootstrap
    experiments = list(dict.fromkeys([selection["reference_experiment"], selection["selected_experiment"]]))
    for experiment in experiments:
        val_seeds, test_seeds = [], []
        experiment_dir = work_dir / "evaluation" / experiment
        for job in [j for j in jobs if j["experiment_id"] == experiment]:
            run_dir = work_dir / "runs" / job["id"]
            # Cached test predictions still require the exact frozen validation evidence.
            frozen = selection["validation_evidence_sha256"][job["id"]]
            for name in ("result.json", "validation_predictions.npz", "best.pt"):
                if frozen.get(name) != file_hash(run_dir / name):
                    raise ValueError(f"Changed test evidence: frozen validation/checkpoint {job['id']}/{name}")
            val = load_predictions(run_dir / "validation_predictions.npz")
            eval_dir = experiment_dir / f"seed{job['seed']}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            binding = _prediction_binding(selection, job, run_dir)
            binding_path = eval_dir / "prediction_binding.json"
            pred_path = eval_dir / "test_predictions.npz"
            if binding_path.exists():
                saved = read_json(binding_path)
                if saved.get("identity") != binding or saved.get("prediction_sha256") != file_hash(pred_path):
                    raise ValueError(f"Changed test evidence: {eval_dir}")
                test = load_predictions(pred_path)
            else:
                print(f"[test] Frozen selection: predict {job['id']}", flush=True)
                test = predict_run(job, prepared[job["variant"]["id"]], run_dir, "test", device=device)
                save_predictions(pred_path, test)
                write_json(binding_path, {"identity": binding, "prediction_sha256": file_hash(pred_path)})
            threshold = fit_thresholds(val["y_true"], val["probabilities"])
            write_json(eval_dir / "test_metrics.json", {
                "threshold_fit_split": "validation", "thresholds": threshold,
                "metrics": multilabel_metrics(test["y_true"], test["probabilities"], threshold)})
            val_seeds.append(val)
            test_seeds.append(test)
        val = ensemble_predictions(val_seeds)
        test = ensemble_predictions(test_seeds)
        threshold = fit_thresholds(val["y_true"], val["probabilities"])
        save_predictions(experiment_dir / "ensemble_validation_predictions.npz", val)
        save_predictions(experiment_dir / "ensemble_test_predictions.npz", test)
        write_json(experiment_dir / "ensemble_thresholds.json", {"fit_split": "validation", "thresholds": threshold})
        metrics = multilabel_metrics(test["y_true"], test["probabilities"], threshold)
        print(f"[test] Patient-cluster bootstrap for {experiment}", flush=True)
        intervals = patient_bootstrap(test["y_true"], test["probabilities"], test["patient_id"], thresholds=threshold,
                                      n_bootstrap=protocol["evaluation"]["bootstrap_replicates"],
                                      seed=protocol["evaluation"]["bootstrap_seed"])
        write_json(experiment_dir / "ensemble_metrics.json", {
            "experiment_id": experiment, "seeds": protocol["seeds"], "aggregation": "mean_seed_probabilities",
            "selection_identity": protocol_hash(selection), "metrics": metrics, "patient_bootstrap": intervals,
            "interpretation": "CI conditional on trained ensemble; not retraining/selection uncertainty"})


def _selection_evidence(results, work_dir):
    evidence = {}
    for result in results:
        run_dir = work_dir / "runs" / result["job_id"]
        evidence[result["job_id"]] = {name: file_hash(run_dir / name)
                                      for name in ("result.json", "validation_predictions.npz", "best.pt")}
    return evidence


def run_suite(config_path, release_dir, raw_dir, work_dir, *, device="cuda", allow_download=False):
    protocol = load_protocol(config_path)
    jobs = build_jobs(protocol)
    release_dir, raw_dir, work_dir = _validate_locations(release_dir, raw_dir, work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    exit_code = 1
    with SuiteLock(work_dir / "suite.lock"):
        with (work_dir / "suite.log").open("a", encoding="utf-8", buffering=1) as logfile:
            tee = _Tee(sys.stdout, logfile)
            with redirect_stdout(tee), redirect_stderr(tee):
                try:
                    with _Heartbeat(work_dir) as pulse:
                        print(f"\n[{utc_now()}] Starting/resuming {protocol['suite_id']}", flush=True)
                        identity = {"protocol_hash": protocol_hash(protocol), "code_fingerprint": code_fingerprint(),
                                    "source_fingerprint": protocol["source"]["expected_fingerprint"]}
                        freeze_json(work_dir / "protocol.json", protocol)
                        freeze_json(work_dir / "identity.json", identity)
                        write_json(work_dir / "status.json", {"state": "running", "updated_at": utc_now()})
                        pulse.stage = "receiving audit and CUDA preflight"
                        preflight(protocol, release_dir, raw_dir, work_dir, device=device)
                        _target_tests(work_dir)
                        from .data import prepare_variant, verify_prepared
                        from .training import train_run
                        prepared, metadata, errors = {}, {}, []
                        for variant in protocol["variants"]:
                            pulse.stage = f"prepare {variant['id']}"
                            try:
                                path = prepare_variant(release_dir, raw_dir, work_dir / "cache", variant,
                                                       allow_download=allow_download)
                                metadata[variant["id"]] = verify_prepared(path)
                                snapshot_prepared_metadata(work_dir, variant["id"], metadata[variant["id"]])
                                prepared[variant["id"]] = path
                            except Exception as exc:
                                detail = {"phase": "prepare", "variant": variant["id"], "error": str(exc),
                                          "traceback": traceback.format_exc(), "at": utc_now()}
                                errors.append(detail)
                                write_json(work_dir / "errors.json", errors)
                                print(detail["traceback"], flush=True)
                        _check_common_cohort(prepared)
                        if prepared:
                            write_json(work_dir / "naive_validation.json", _naive_validation(next(iter(prepared.values()))))
                        results = []
                        for index, job in enumerate(jobs, 1):
                            pulse.stage = f"job {index}/{len(jobs)} {job['id']}"
                            print(f"[{pulse.stage}]", flush=True)
                            if job["variant"]["id"] not in prepared:
                                errors.append({"job_id": job["id"], "phase": "train", "error": "prepared variant unavailable"})
                                continue
                            try:
                                if code_fingerprint() != identity["code_fingerprint"]:
                                    raise ValueError("Source changed during suite")
                                job_identity = dict(identity, prepared_fingerprint=metadata[job["variant"]["id"]]["fingerprint"])
                                results.append(train_run(job, prepared[job["variant"]["id"]], work_dir / "runs" / job["id"],
                                                         identity=job_identity, device=device, resume=True))
                            except Exception as exc:
                                detail = {"job_id": job["id"], "phase": "train", "error": str(exc),
                                          "traceback": traceback.format_exc(), "at": utc_now()}
                                errors.append(detail)
                                write_json(work_dir / "errors.json", errors)
                                print(detail["traceback"], flush=True)
                        write_json(work_dir / "errors.json", errors)
                        if errors or len(results) != len(jobs):
                            raise RuntimeError(f"Suite incomplete ({len(results)}/{len(jobs)} jobs). Test remains closed; see errors.json.")
                        if code_fingerprint() != identity["code_fingerprint"]:
                            raise ValueError("Source changed during suite; refusing selection")
                        selection = select_experiment(protocol, results)
                        selection["identity"] = identity
                        selection["validation_evidence_sha256"] = _selection_evidence(results, work_dir)
                        freeze_json(work_dir / "selection.json", selection)
                        comparison = _validation_comparison(selection)
                        write_json(work_dir / "validation_comparison.json", comparison)
                        import csv
                        with (work_dir / "validation_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
                            writer = csv.DictWriter(handle, fieldnames=["experiment_id", "mean_macro_auroc", "std_macro_auroc",
                                                                       "mean_macro_average_precision", "std_macro_average_precision"])
                            writer.writeheader()
                            writer.writerows({k: v for k, v in row.items() if k != "seeds"} for row in selection["rows"])
                        pulse.stage = "test reference and validation-selected ensemble"
                        _evaluate(protocol, jobs, prepared, work_dir, selection, device)
                        write_json(work_dir / "status.json", {"state": "completed", "updated_at": utc_now(),
                                                              "jobs_completed": len(results),
                                                              "selection_sha256": protocol_hash(selection)})
                        exit_code = 0
                except BaseException as exc:
                    failure = {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
                               "updated_at": utc_now()}
                    write_json(work_dir / "error.json", failure)
                    write_json(work_dir / "status.json", {"state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                                          "updated_at": utc_now(), "error": str(exc)})
                    print(failure["traceback"], flush=True)
                finally:
                    try:
                        summarize(work_dir)
                        bundle = collect_results(work_dir)
                        print(f"Results/diagnostics bundle: {bundle}", flush=True)
                    except Exception:
                        exit_code = 1
                        print("Result collection failed; run collect manually after resolving disk/access error.", flush=True)
                        traceback.print_exc()
    return exit_code


def summarize(work_dir):
    work_dir = Path(work_dir)
    state = read_json(work_dir / "status.json") if (work_dir / "status.json").exists() else {"state": "not_started"}
    lines = ["# Kết quả PTB-XL cho Line", "", f"Trạng thái: **{state['state']}**.", "",
             "Đây là phân loại 5 nhóm chẩn đoán ECG, không phải nhãn suy tim.",
             "Baseline bám benchmark gốc nhưng dùng PTB-XL1.0.3, cohort21387 và PyTorch hiện đại; không phải tái lập nguyên xi.", ""]
    selection_path = work_dir / "selection.json"
    if selection_path.exists():
        selection = read_json(selection_path)
        lines.extend([f"Phương án được chọn bằng validation: **{selection['selected_experiment']}**.",
                      f"Mốc tham chiếu: **{selection['reference_experiment']}**.", "",
                      "| Phương án | Validation macro AUROC (mean ± SD seed) | Validation macro AP |", "|---|---:|---:|"])
        for row in selection["rows"]:
            lines.append(f"| {row['experiment_id']} | {row['mean_macro_auroc']:.5f} ± {row['std_macro_auroc']:.5f} | {row['mean_macro_average_precision']:.5f} |")
        lines.append("")
        for path in sorted((work_dir / "evaluation").glob("*/ensemble_metrics.json")):
            evaluation = read_json(path)
            metrics = evaluation["metrics"]
            lines.append(f"Test ensemble {evaluation['experiment_id']}: macro AUROC={metrics.get('macro_auroc')}; macro AP={metrics.get('macro_average_precision')}; macro F1={metrics.get('macro_f1')}.")
        lines.extend(["", "Xem ensemble_metrics.json để đọc CI95% bootstrap theo patient và số bootstrap hợp lệ; CI có điều kiện trên ensemble đã train, không bao gồm bất định chọn phương án/retrain."])
    else:
        completed = sum(1 for _ in (work_dir / "runs").glob("*/completed.json"))
        lines.append(f"Có {completed} completion marker; chưa có selection.json. Không được kết luận winner/test từ lượt chạy thiếu.")
    lines.extend(["", "AP là average precision không nội suy, được dùng làm AUPRC; các lớp không xác định được metric được ghi null cùng số lớp hợp lệ.",
                  "Các contrast được định trước trong validation_comparison.json. Chênh lệch3seed là mô tả, không tự động chứng minh ý nghĩa thống kê.",
                  "So tần số có scale_odd kernel nên số tham số/bước thời gian cũng đổi; không suy ra tác động thuần của thông tin tần số.",
                  "", "Nếu failed/interrupted: đọc error.json, errors.json và suite.log; sửa môi trường/đường dẫn rồi chạy lại cùng lệnh. Không sửa config/code trong cùng work directory.",
                  "Bộ kết quả không chứa waveform thô/cache. Giữ work directory ở máy trường để resume.", ""])
    (work_dir / "SUMMARY.vi.md").write_text("\n".join(lines), encoding="utf-8")
    return work_dir / "SUMMARY.vi.md"


def collect_results(work_dir, *, include_source=True):
    work_dir = Path(work_dir).resolve()
    if not work_dir.is_dir():
        raise FileNotFoundError(work_dir)
    files = []
    for name in SUITE_ARTIFACTS:
        path = work_dir / name
        if path.is_file():
            files.append((path, name))
    for folder, names in (("runs", RUN_ARTIFACTS), ("evaluation", EVAL_ARTIFACTS)):
        for path in (work_dir / folder).rglob("*") if (work_dir / folder).exists() else []:
            if path.is_file() and path.name in names:
                files.append((path, path.relative_to(work_dir).as_posix()))
    if (work_dir / "protocol.json").is_file():
        protocol = load_protocol(work_dir / "protocol.json")
        for variant in protocol["variants"]:
            path = work_dir / "prepared_metadata" / f"{variant['id']}.json"
            if path.is_file():
                files.append((path, path.relative_to(work_dir).as_posix()))
    if include_source:
        files.extend((p, f"source/ptbxl_experiments/{p.name}") for p in (ROOT / "ptbxl_experiments").glob("*.py"))
    for path, name in files:
        allowed_root = ROOT if name.startswith("source/") else work_dir
        if not path.resolve().is_relative_to(allowed_root.resolve()):
            raise ValueError(f"Results source escapes approved root: {path}")
        cursor = path
        while cursor != allowed_root:
            if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
                raise ValueError(f"Refusing linked path in results: {path}")
            cursor = cursor.parent
    outbound = work_dir / "outbound"
    outbound.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = outbound / f"results-{stamp}-{uuid.uuid4().hex[:8]}.zip"
    partial = output.with_suffix(".zip.partial")
    manifest = {"created_at": utc_now(), "files": {}, "excludes": ["raw", "cache", "venv", "secrets"]}
    with zipfile.ZipFile(partial, "x", allowZip64=True) as archive:
        for path, name in sorted(files, key=lambda x: x[1]):
            digest = hashlib.sha256()
            size = 0
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED if path.suffix in (".pt", ".npz", ".npy") else zipfile.ZIP_DEFLATED
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as destination:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            manifest["files"][name] = {"sha256": digest.hexdigest(), "size": size}
        archive.writestr("RESULTS_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    os.replace(partial, output)
    output.with_suffix(".zip.sha256").write_text(f"{file_hash(output)}  {output.name}\n", encoding="ascii")
    return output
