"""Record-level multilabel metrics without a Torch/scikit-learn dependency.

Intent: keep the statistical definition auditable and offline summaries portable.
Status: NumPy fixture coverage; clinical/model validity is a separate question.
AP is non-interpolated average precision at distinct score thresholds (not the
trapezoidal area under an interpolated PR curve). Undefined scores stay None.
"""

from __future__ import annotations

import numpy as np


CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def _arrays(y_true, probabilities):
    y, p = np.asarray(y_true), np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 2 or y.shape != p.shape or y.shape[1] != len(CLASSES) or not len(y):
        raise ValueError("Expected nonempty matching [records, 5] labels/probabilities")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be finite binary indicators")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite in [0, 1]")
    return y.astype(np.int8, copy=False), p


def _threshold_array(thresholds):
    value = np.asarray(thresholds, dtype=np.float64)
    if value.shape != (len(CLASSES),) or not np.isfinite(value).all():
        raise ValueError("Expected five finite thresholds")
    return value


def _binary_ranking(y, p):
    """Tied-score groups share a threshold, giving half-credit in AUROC."""
    order = np.argsort(-p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(sorted_p) != 0), len(y) - 1]
    tp = np.cumsum(sorted_y, dtype=np.int64)[last]
    predicted = last + 1
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    # ROC trapezoids are exactly rank AUROC, including half-credit for ties.
    if positives and negatives:
        tpr = np.r_[0.0, tp / positives]
        fpr = np.r_[0.0, (predicted - tp) / negatives]
        auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))
    else:
        auc = None
    ap = float(np.sum(np.diff(np.r_[0, tp]) / positives * (tp / predicted))) if positives else None
    return auc, ap, sorted_p[last], tp, predicted


def _defined_mean(values):
    defined = [value for value in values if value is not None]
    return (float(np.mean(defined)) if defined else None), len(defined)


def multilabel_metrics(y_true, probabilities, thresholds=None) -> dict:
    """Calculate five-label record-level metrics; thresholds must come from validation.

    Precision is undefined with no predicted positive; recall with no true
    positive; F1 only when both true and predicted positives are absent. The
    explicit valid-class counts make changing denominators visible.
    """
    y, p = _arrays(y_true, probabilities)
    threshold = _threshold_array(thresholds) if thresholds is not None else None
    per_class = {}
    for i, name in enumerate(CLASSES):
        auc, ap, *_ = _binary_ranking(y[:, i], p[:, i])
        positive = int(y[:, i].sum())
        item = {
            "positive_support": positive,
            "negative_support": len(y) - positive,
            "auroc": auc,
            "average_precision": ap,
            "auroc_defined": auc is not None,
            "average_precision_defined": ap is not None,
            "brier_score": float(np.mean((p[:, i] - y[:, i]) ** 2)),
        }
        if threshold is not None:
            calls = p[:, i] >= threshold[i]
            tp = int(np.count_nonzero(calls & (y[:, i] == 1)))
            predicted = int(calls.sum())
            item.update({
                "threshold": float(threshold[i]), "predicted_positive": predicted,
                "true_positive": tp, "false_positive": predicted - tp,
                "false_negative": positive - tp,
                "true_negative": len(y) - positive - (predicted - tp),
                "precision": tp / predicted if predicted else None,
                "recall": tp / positive if positive else None,
                "f1": 2 * tp / (predicted + positive) if predicted + positive else None,
            })
        per_class[name] = item
    result = {
        "n_records": len(y), "per_class": per_class,
        "brier_score": float(np.mean((p - y) ** 2)),
        "average_precision_definition": "non-interpolated average precision; tied scores grouped",
        "macro_definition": "mean over defined classes; see valid_*_classes",
    }
    for key in ("auroc", "average_precision") + (("f1", "precision", "recall") if threshold is not None else ()):
        result[f"macro_{key}"], result[f"valid_{key}_classes"] = _defined_mean(
            [value[key] for value in per_class.values()]
        )
    return result


def fit_thresholds(y_true, probabilities) -> np.ndarray:
    """Maximize classwise validation F1; exact ties prefer the largest threshold.

    For a validation class with no positives, choose nextafter(1,+inf), i.e.
    predict none even when the input probability is exactly 1. This decision
    is deterministic and does not read test labels.
    """
    y, p = _arrays(y_true, probabilities)
    thresholds = np.full(len(CLASSES), np.nextafter(1.0, np.inf), dtype=np.float64)
    for i in range(len(CLASSES)):
        positives = int(y[:, i].sum())
        if not positives:
            continue
        _, _, candidates, tp, predicted = _binary_ranking(y[:, i], p[:, i])
        f1 = 2 * tp / (positives + predicted)
        # candidates descend, so the first maximizing value is the largest.
        thresholds[i] = candidates[int(np.argmax(f1))]
    return thresholds


def patient_bootstrap(y_true, probabilities, patient_ids, *, thresholds=None,
                      n_bootstrap=1000, seed=42) -> dict:
    """Percentile 95% CIs by sampling patients with replacement, keeping all ECGs.

    Within a replicate, an undefined-class metric is omitted from the macro as
    in multilabel_metrics. Counts of full-five-class replicates and undefined
    per-class scores are reported explicitly; these are conditional intervals
    for the fitted model, not training-seed or external-population uncertainty.
    """
    y, p = _arrays(y_true, probabilities)
    ids = np.asarray(patient_ids)
    if ids.ndim != 1 or len(ids) != len(y):
        raise ValueError("patient_ids must align one-to-one with prediction rows")
    if any(value is None or str(value).strip().lower() in {"", "nan", "none"} for value in ids):
        raise ValueError("patient_ids must not be missing")
    if not isinstance(n_bootstrap, (int, np.integer)) or isinstance(n_bootstrap, bool) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    threshold = _threshold_array(thresholds) if thresholds is not None else None
    unique, inverse = np.unique(ids.astype(str), return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    names = ["auroc", "average_precision"] + (["f1"] if threshold is not None else [])
    values = {name: [] for name in names}
    full_classes = {name: 0 for name in names}
    class_valid = {name: {label: 0 for label in CLASSES} for name in names}
    for _ in range(int(n_bootstrap)):
        chosen = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[i] for i in chosen])
        metrics = multilabel_metrics(y[rows], p[rows], threshold)
        for name in names:
            value = metrics[f"macro_{name}"]
            if value is not None:
                values[name].append(value)
            full_classes[name] += metrics[f"valid_{name}_classes"] == len(CLASSES)
            for label in CLASSES:
                class_valid[name][label] += metrics["per_class"][label][name] is not None
    intervals = {}
    for name in names:
        bounds = np.quantile(values[name], [0.025, 0.975]) if values[name] else [None, None]
        intervals[f"macro_{name}"] = {
            "low": float(bounds[0]) if bounds[0] is not None else None,
            "high": float(bounds[1]) if bounds[1] is not None else None,
            "valid_replicates": len(values[name]),
            "all_five_classes_defined_replicates": int(full_classes[name]),
            "per_class_valid_replicates": class_valid[name],
        }
    return {
        "method": "patient-cluster percentile bootstrap", "confidence_level": 0.95,
        "n_bootstrap": int(n_bootstrap), "seed": int(seed), "n_patients": len(unique),
        "n_records": len(y), "intervals": intervals,
        "undefined_policy": "macro over defined classes; undefined macro draws excluded with counts",
        "scope": "conditional on fitted predictions; excludes training-seed and external-population uncertainty",
    }
