"""Target-only training engine for the frozen PTB-XL experiment protocol.

Intent: reproducible record-level evaluation and crash-safe epoch-boundary resume.
Status: CUDA/model execution is reserved for the school machine, never import
this module from plan/collect/summarize. The suite owns the test-selection gate.
Recovery: last.pt is the authoritative epoch commit and contains the best model,
so a crash before best.pt/history writes can be recovered without losing a best
epoch or duplicating history. Completed runs additionally require file hashes.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
import traceback

# Must precede the first CUDA context; required for deterministic CUDA GEMM.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .metrics import CLASSES, fit_thresholds, multilabel_metrics
from .model import build_model


LABEL_COLUMNS = [f"y_{name}" for name in CLASSES]
_ARTIFACTS = (
    "result.json", "validation_predictions.npz", "val_thresholds.npy",
    "history.json", "history.csv", "best.pt", "last.pt",
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _object_hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    _atomic_write(path, lambda stream: stream.write(text.encode("utf-8")))


def _atomic_torch_save(path, payload):
    _atomic_write(path, lambda stream: torch.save(payload, stream))


def _write_history(directory, history):
    import csv
    import io
    _write_json(Path(directory) / "history.json", history)
    stream = io.StringIO(newline="")
    fields = ("epoch", "train_loss", "validation_bce", "validation_macro_auroc",
              "validation_macro_average_precision", "learning_rate", "amp_skipped_steps",
              "elapsed_seconds")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in history)
    _atomic_write(Path(directory) / "history.csv", lambda output: output.write(stream.getvalue().encode("utf-8")))


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return copy.deepcopy(value)


def _rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"name": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state):
    random.setstate(state["python"])
    saved = state["numpy"]
    np.random.set_state((saved["name"], np.asarray(saved["keys"], dtype=np.uint32),
                         saved["position"], saved["has_gauss"], saved["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def _seed(seed):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before creating the CUDA context")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _device(value):
    device = torch.device(value)
    if device.type != "cuda":
        raise ValueError("This experiment package requires CUDA; CPU fallback is disabled")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check school GPU driver and the pinned CUDA PyTorch installation")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    return device


def _runtime(device):
    return {
        "python": platform.python_version(), "platform": sys.platform,
        "torch": str(torch.__version__), "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "pandas": importlib.metadata.version("pandas"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "deterministic_algorithms": True, "cudnn_benchmark": False, "tf32": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _binding(job, identity, device):
    required = {"protocol_hash", "code_fingerprint", "source_fingerprint", "prepared_fingerprint"}
    if set(identity) != required or any(not isinstance(value, str) or not value for value in identity.values()):
        raise ValueError("Run identity must contain the four nonempty protocol/code/source/prepared hashes")
    return {"schema_version": 1, "job": copy.deepcopy(job), "identity": dict(identity), "runtime": _runtime(device)}


def _load_prepared(directory, split, normalization):
    from .data import load_prepared
    return load_prepared(directory, split, normalization)


def _window_starts(samples, crop, stride):
    if crop < 1 or stride < 1 or samples < crop:
        raise ValueError("Invalid crop/stride or signal shorter than the crop")
    starts = list(range(0, samples - crop + 1, stride))
    if starts[-1] != samples - crop:
        starts.append(samples - crop)
    return starts


class ECGCropDataset(Dataset):
    """Memory-mapped, Windows-spawn-safe dataset with deterministic per-ECG crops."""
    def __init__(self, directory, split, normalization, crop, *, seed, epoch, stride):
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        self.directory, self.split, self.normalization = Path(directory), split, normalization
        self.crop, self.stride, self.seed, self.epoch = int(crop), int(stride), int(seed), int(epoch)
        self._open()
        self.record_count = len(self.manifest)
        self.starts = _window_starts(self.signals.shape[-1], self.crop, self.stride)

    def _open(self):
        self.signals, self.manifest, self.mean, self.std, self.metadata = _load_prepared(
            self.directory, self.split, self.normalization)
        if self.signals.shape[:2] != (len(self.manifest), 12):
            raise ValueError("Prepared data/manifest alignment mismatch")
        if self.manifest["ecg_id"].isna().any() or self.manifest["ecg_id"].duplicated().any():
            raise ValueError("ECG IDs must be present and unique within each split")
        if self.manifest["patient_id"].isna().any():
            raise ValueError("Missing patient ID")
        self.labels = self.manifest[LABEL_COLUMNS].to_numpy(dtype=np.float32)
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("Prepared labels must be binary")
        self.mean, self.std = np.asarray(self.mean), np.asarray(self.std)
        if self.mean.shape != (12, 1) or self.std.shape != (12, 1):
            raise ValueError("Normalization must broadcast as [12,1]")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("Invalid normalization statistics")

    def __getstate__(self):
        state = self.__dict__.copy()
        # Never pickle a multi-GB memmap into spawned Windows workers.
        state["signals"] = None
        return state

    def __len__(self):
        return self.record_count if self.split == "train" else self.record_count * len(self.starts)

    def __getitem__(self, index):
        if self.signals is None:
            self._open()
        if self.split == "train":
            record = index
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, record]))
            start = int(rng.integers(0, self.signals.shape[-1] - self.crop + 1))
        else:
            record, window = divmod(index, len(self.starts))
            start = self.starts[window]
        signal = np.asarray((self.signals[record, :, start:start + self.crop] - self.mean) / self.std,
                            dtype=np.float32)
        if not np.isfinite(signal).all():
            raise ValueError(f"Nonfinite normalized input for ECG row {record}")
        return torch.from_numpy(signal.copy()), torch.from_numpy(self.labels[record].copy()), record


def _batch_indices(count, batch_size, seed, epoch):
    if count < 2 or batch_size < 2:
        raise ValueError("BatchNorm training needs at least two records and batch_size>=2")
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 7919])).permutation(count).tolist()
    batches = [order[start:start + batch_size] for start in range(0, count, batch_size)]
    if len(batches[-1]) == 1:
        if len(batches) < 2 or len(batches[-2]) < 3:
            raise ValueError("Singleton final training batch; use batch_size>=3 for this cohort")
        batches[-1].insert(0, batches[-2].pop())
    return batches


def _train_loader(dataset, settings, seed, epoch):
    generator = torch.Generator().manual_seed(seed + epoch)
    return DataLoader(dataset, batch_sampler=_batch_indices(len(dataset), settings["batch_size"], seed, epoch),
                      num_workers=settings["workers"], pin_memory=True, generator=generator,
                      persistent_workers=False)


def _predict_dataset(model, dataset, *, batch_size, device, workers, amp):
    if dataset.split == "train":
        raise ValueError("Record-level evaluation requires validation/test window enumeration")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=False, generator=torch.Generator().manual_seed(0))
    probabilities = np.full((dataset.record_count, len(CLASSES)), -np.inf, dtype=np.float64)
    counts = np.zeros(dataset.record_count, dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for signal, _, indices in loader:
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(signal.to(device, non_blocking=True))
            batch = torch.sigmoid(logits.float()).cpu().numpy()
            if not np.isfinite(batch).all():
                raise FloatingPointError("Nonfinite evaluation probabilities")
            rows = indices.numpy()
            np.maximum.at(probabilities, rows, batch)
            np.add.at(counts, rows, 1)
    if not np.all(counts == len(dataset.starts)) or not np.isfinite(probabilities).all():
        raise ValueError("Missing/repeated evaluation windows or nonfinite aggregate")
    return {
        "ecg_id": dataset.manifest["ecg_id"].to_numpy(copy=True),
        "patient_id": dataset.manifest["patient_id"].astype(str).to_numpy(dtype=str),
        "y_true": dataset.labels.copy(), "probabilities": probabilities,
    }


def _validate_prepared(job, metadata, identity):
    if metadata["fingerprint"] != identity["prepared_fingerprint"]:
        raise ValueError("Prepared fingerprint differs from run identity")
    if metadata["source_fingerprint"] != identity["source_fingerprint"]:
        raise ValueError("Source fingerprint differs from run identity")
    if metadata["fs"] != job["variant"]["fs"] or metadata.get("variant") != job["variant"]:
        raise ValueError("Prepared variant differs from job")


def _verify_completed(directory, binding):
    path = Path(directory) / "completed.json"
    if not path.is_file():
        return None
    marker = json.loads(path.read_text(encoding="utf-8"))
    if marker.get("identity_hash") != _object_hash(binding):
        raise ValueError("Completed run identity mismatch; use a new run directory")
    if set(marker.get("artifact_sha256", {})) != set(_ARTIFACTS):
        raise ValueError("Completed artifact hash manifest is incomplete")
    for name, expected in marker["artifact_sha256"].items():
        file = Path(directory) / name
        if not file.is_file() or _sha256(file) != expected:
            raise ValueError(f"Completed artifact checksum mismatch: {name}")
    result = json.loads((Path(directory) / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "completed" or result.get("job_id") != binding["job"]["id"]:
        raise ValueError("Completed result identity/status mismatch")
    return result


def _best_checkpoint(last):
    return {"schema_version": 1, "binding": last["binding"], "epoch": last["best"]["epoch"],
            "score": last["best"]["score"], "model": last["best"]["model"]}


def _load_checkpoint(path, binding):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("binding") != binding:
        raise ValueError("Checkpoint identity mismatch; resume requires identical job/source/code/runtime")
    return checkpoint


def _record_bce(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def train_run(job: dict, prepared_dir: Path, run_dir: Path, *, identity: dict,
              device: str = "cuda", resume: bool = True) -> dict:
    """Train one fixed job; partial epochs rerun from the last atomic epoch commit."""
    run_dir, prepared_dir = Path(run_dir), Path(prepared_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    current_device = _device(device)
    binding = _binding(job, identity, current_device)
    identity_file = run_dir / "identity.json"
    if identity_file.exists():
        if json.loads(identity_file.read_text(encoding="utf-8")) != binding:
            raise ValueError("Run identity mismatch; keep old results and use a new run directory")
        if not resume:
            raise ValueError("Run directory exists but resume=False")
    elif any(run_dir.iterdir()):
        raise ValueError("Nonempty run directory lacks identity; refusing to overwrite")
    else:
        _write_json(identity_file, binding)
    complete = _verify_completed(run_dir, binding)
    if complete is not None:
        print(f"[{job['id']}] verified completed artifacts; skip training", flush=True)
        return complete
    started = time.monotonic()
    settings, seed = job["training"], int(job["seed"])
    committed_epoch = 0
    try:
        if settings["architecture"] != "resnet1d_wang" or settings["aggregation"] != "max":
            raise ValueError("Unsupported architecture/aggregation")
        if settings["scheduler"] != "onecycle" or settings["kernel_policy"] != "scale_odd":
            raise ValueError("Unsupported scheduler/kernel policy")
        _seed(seed)
        torch.cuda.reset_peak_memory_stats(current_device)
        fs = int(job["variant"]["fs"])
        crop, stride = round(settings["crop_seconds"] * fs), round(settings["eval_stride_seconds"] * fs)
        train = ECGCropDataset(prepared_dir, "train", job["normalization"], crop, seed=seed, epoch=0, stride=stride)
        validation = ECGCropDataset(prepared_dir, "validation", job["normalization"], crop,
                                    seed=seed, epoch=0, stride=stride)
        _validate_prepared(job, train.metadata, identity)
        _validate_prepared(job, validation.metadata, identity)
        model = build_model(fs).to(current_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                      weight_decay=settings["weight_decay"])
        steps = len(_batch_indices(len(train), settings["batch_size"], seed, 0))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=settings["learning_rate"], epochs=settings["epochs"], steps_per_epoch=steps,
            pct_start=0.3, div_factor=25, final_div_factor=1e4)
        scaler = torch.amp.GradScaler("cuda", enabled=settings["amp"])
        history, best, elapsed_before, peak_before = [], None, 0.0, 0
        if (run_dir / "last.pt").is_file():
            last = _load_checkpoint(run_dir / "last.pt", binding)
            committed_epoch = int(last["epoch"])
            if not 1 <= committed_epoch <= settings["epochs"]:
                raise ValueError("Checkpoint epoch outside frozen training schedule")
            history = last["history"]
            if [row["epoch"] for row in history] != list(range(1, committed_epoch + 1)):
                raise ValueError("Checkpoint history is incomplete or has duplicate epochs")
            model.load_state_dict(last["model"], strict=True)
            optimizer.load_state_dict(last["optimizer"])
            scheduler.load_state_dict(last["scheduler"])
            scaler.load_state_dict(last["scaler"])
            best = last["best"]
            elapsed_before, peak_before = last["elapsed_seconds"], last["peak_gpu_memory_bytes"]
            # Recover any best/history write interrupted after the last.pt commit.
            _atomic_torch_save(run_dir / "best.pt", _best_checkpoint(last))
            _write_history(run_dir, history)
            _restore_rng(last["rng"])
            print(f"[{job['id']}] resume at epoch {committed_epoch + 1}; partial epoch is rerun", flush=True)
        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(committed_epoch + 1, settings["epochs"] + 1):
            train.epoch = epoch
            loader = _train_loader(train, settings, seed, epoch)
            model.train()
            running_loss, seen, skipped = 0.0, 0, 0
            heartbeat = time.monotonic()
            _write_json(run_dir / "status.json", {"status": "training", "job_id": job["id"],
                         "epoch": epoch, "last_committed_epoch": committed_epoch, "updated_at": _now()})
            for step, (signal, targets, _) in enumerate(loader, 1):
                optimizer.zero_grad(set_to_none=True)
                signal = signal.to(current_device, non_blocking=True)
                targets = targets.to(current_device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=settings["amp"]):
                    logits = model(signal)
                    loss = criterion(logits, targets)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Nonfinite training loss in epoch {epoch}, batch {step}")
                scaler.scale(loss).backward()
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < old_scale)
                # The frozen OneCycle schedule advances per scheduled minibatch;
                # AMP overflow skips are explicitly recorded, never hidden.
                scheduler.step()
                running_loss += float(loss.detach().item()) * len(signal)
                seen += len(signal)
                if time.monotonic() - heartbeat >= 30:
                    print(f"[{job['id']}] epoch {epoch}/{settings['epochs']} batch {step}/{len(loader)} loss={running_loss/seen:.5f}", flush=True)
                    _write_json(run_dir / "status.json", {"status": "training", "job_id": job["id"],
                                "epoch": epoch, "batch": step, "batches": len(loader),
                                "last_committed_epoch": committed_epoch, "updated_at": _now()})
                    heartbeat = time.monotonic()
            prediction = _predict_dataset(model, validation, batch_size=settings["batch_size"],
                                          device=current_device, workers=settings["workers"], amp=settings["amp"])
            val_metrics = multilabel_metrics(prediction["y_true"], prediction["probabilities"])
            score = val_metrics["macro_auroc"]
            if score is None or not math.isfinite(score):
                raise ValueError("Validation macro AUROC undefined; cannot choose a checkpoint")
            if best is None or score > best["score"]:
                best = {"epoch": epoch, "score": score, "model": _cpu(model.state_dict())}
            elapsed = elapsed_before + time.monotonic() - started
            history.append({"epoch": epoch, "train_loss": running_loss / seen,
                            "validation_bce": _record_bce(prediction["y_true"], prediction["probabilities"]),
                            "validation_macro_auroc": score,
                            "validation_macro_average_precision": val_metrics["macro_average_precision"],
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "amp_skipped_steps": skipped, "elapsed_seconds": elapsed})
            last = {"schema_version": 1, "binding": binding, "epoch": epoch,
                    "model": _cpu(model.state_dict()), "optimizer": _cpu(optimizer.state_dict()),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "rng": _rng_state(), "history": copy.deepcopy(history), "best": best,
                    "elapsed_seconds": elapsed,
                    "peak_gpu_memory_bytes": max(peak_before, torch.cuda.max_memory_allocated(current_device))}
            _atomic_torch_save(run_dir / "last.pt", last)
            committed_epoch = epoch
            _atomic_torch_save(run_dir / "best.pt", _best_checkpoint(last))
            _write_history(run_dir, history)
            print(f"[{job['id']}] committed epoch {epoch}; val macro AUROC={score:.6f}; best={best['epoch']}", flush=True)
        if best is None:
            raise ValueError("Training finished without a best checkpoint")
        model.load_state_dict(best["model"], strict=True)
        prediction = _predict_dataset(model, validation, batch_size=settings["batch_size"],
                                      device=current_device, workers=settings["workers"], amp=settings["amp"])
        thresholds = fit_thresholds(prediction["y_true"], prediction["probabilities"])
        val_metrics = multilabel_metrics(prediction["y_true"], prediction["probabilities"], thresholds)
        _atomic_write(run_dir / "validation_predictions.npz", lambda stream: np.savez_compressed(stream, **prediction))
        _atomic_write(run_dir / "val_thresholds.npy", lambda stream: np.save(stream, thresholds, allow_pickle=False))
        result = {
            "job_id": job["id"], "experiment_id": job["experiment_id"], "seed": seed,
            "status": "completed", "val_metrics": val_metrics, "run_dir": str(run_dir),
            "best_epoch": best["epoch"], "epochs_completed": settings["epochs"],
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
            "peak_gpu_memory_bytes": max(peak_before, torch.cuda.max_memory_allocated(current_device)),
            "identity_hash": _object_hash(binding), "architecture": model.architecture_details,
            "threshold_fit_split": "validation",
            "validation_f1_note": "thresholds optimized on this validation set; not an unbiased test score",
            "amp_skipped_steps": sum(row["amp_skipped_steps"] for row in history),
            "benchmark_status": "modern benchmark-aligned adaptation; not bit-exact historical reproduction",
            "completed_at": _now(),
        }
        _write_json(run_dir / "result.json", result)
        _write_json(run_dir / "completed.json", {"schema_version": 1, "identity_hash": _object_hash(binding),
                    "artifact_sha256": {name: _sha256(run_dir / name) for name in _ARTIFACTS}, "completed_at": _now()})
        _write_json(run_dir / "status.json", {"status": "completed", "job_id": job["id"],
                    "last_committed_epoch": committed_epoch, "updated_at": _now()})
        return result
    except Exception as error:
        details = {"status": "failed", "job_id": job["id"], "last_committed_epoch": committed_epoch,
                   "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
                   "updated_at": _now(), "batch_size_unchanged": job["training"]["batch_size"]}
        if isinstance(error, torch.cuda.OutOfMemoryError):
            details["action"] = "No automatic batch reduction. Review protocol and restart in a NEW suite directory."
        _write_json(run_dir / "error.json", details)
        _write_json(run_dir / "status.json", {key: value for key, value in details.items() if key != "traceback"})
        raise
    finally:
        # Release the previous model before the suite starts another GPU job.
        if "model" in locals():
            del model
        if "optimizer" in locals():
            del optimizer
        torch.cuda.empty_cache()


def predict_run(job: dict, prepared_dir: Path, run_dir: Path, split: str, *, device: str = "cuda") -> dict:
    """Predict with verified completed best weights. Suite must authorize test first."""
    if split not in {"validation", "test"}:
        raise ValueError("predict_run only accepts validation/test")
    run_dir = Path(run_dir)
    current_device = _device(device)
    saved = json.loads((run_dir / "identity.json").read_text(encoding="utf-8"))
    binding = _binding(job, saved["identity"], current_device)
    if saved != binding:
        raise ValueError("Prediction job/runtime identity mismatch")
    if _verify_completed(run_dir, binding) is None:
        raise ValueError("Prediction requires a verified completed training run")
    _seed(int(job["seed"]))
    settings, fs = job["training"], int(job["variant"]["fs"])
    dataset = ECGCropDataset(Path(prepared_dir), split, job["normalization"], round(settings["crop_seconds"] * fs),
                             seed=job["seed"], epoch=0, stride=round(settings["eval_stride_seconds"] * fs))
    _validate_prepared(job, dataset.metadata, binding["identity"])
    model = build_model(fs).to(current_device)
    try:
        checkpoint = _load_checkpoint(run_dir / "best.pt", binding)
        model.load_state_dict(checkpoint["model"], strict=True)
        return _predict_dataset(model, dataset, batch_size=settings["batch_size"], device=current_device,
                                workers=settings["workers"], amp=settings["amp"])
    finally:
        del model
        torch.cuda.empty_cache()


def target_smoke(training: dict, *, device: str = "cuda") -> dict:
    """Target-only maximum-input/backward and safe checkpoint optimizer-resume smoke.

    All inputs are synthetic. Failure stops the long suite without changing the
    batch, architecture or scientific protocol. Full epoch resume equivalence is
    additionally covered by gated tests/test_experiment_target.py on the target.
    """
    current_device = _device(device)
    _seed(42)
    torch.cuda.reset_peak_memory_stats(current_device)
    started = time.monotonic()
    details = []
    try:
        for fs in (100, 200, 500):
            model = build_model(fs).to(current_device)
            model.train()
            count = training["batch_size"] if fs == 500 else 2
            signal = torch.randn(count, 12, round(fs * training["crop_seconds"]), device=current_device)
            labels = torch.zeros(count, 5, device=current_device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                          weight_decay=training["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=training["amp"])
            with torch.amp.autocast("cuda", enabled=training["amp"]):
                logits = model(signal)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if logits.shape != (count, 5) or not torch.isfinite(loss).item():
                raise RuntimeError(f"Invalid synthetic forward/loss at {fs} Hz")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            details.append({**model.architecture_details, "tested_batch_size": count,
                            "synthetic_loss": float(loss.detach().item())})
            with tempfile.TemporaryDirectory(prefix="ecg-checkpoint-smoke-") as temporary:
                path = Path(temporary) / "checkpoint.pt"
                payload = {"model": _cpu(model.state_dict()), "optimizer": _cpu(optimizer.state_dict()),
                           "scaler": scaler.state_dict(), "rng": _rng_state()}
                _atomic_torch_save(path, payload)
                restored = torch.load(path, map_location="cpu", weights_only=True)
                model.load_state_dict(restored["model"], strict=True)
                optimizer.load_state_dict(restored["optimizer"])
                scaler.load_state_dict(restored["scaler"])
                _restore_rng(restored["rng"])
                # A real update after load catches optimizer device-state errors.
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=training["amp"]):
                    resumed_loss = nn.functional.binary_cross_entropy_with_logits(model(signal), labels)
                if not torch.isfinite(resumed_loss).item():
                    raise RuntimeError("Nonfinite loss after safe checkpoint load")
                scaler.scale(resumed_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            del model, optimizer, scaler, signal, labels, logits, loss, resumed_loss
            torch.cuda.empty_cache()
        torch.cuda.synchronize(current_device)
        return {"status": "passed", "runtime": _runtime(current_device), "architectures": details,
                "elapsed_seconds": time.monotonic() - started,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(current_device),
                "scope": "synthetic forward/backward and safe optimizer checkpoint load; no real ECG training"}
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError("CUDA preflight OOM at frozen batch/input. No automatic batch reduction; review protocol before long training.") from error
    finally:
        torch.cuda.empty_cache()
