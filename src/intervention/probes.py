"""
Linear probes for Gate 0 — ported from Paper4/PhaseB/analysis/stage3_2_reference_arm_baseline.py
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TEST_SIZE = 0.3


def linear_probe(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    test_size: float = TEST_SIZE,
) -> tuple[dict, np.ndarray, np.ndarray, object]:
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=test_size, random_state=seed, stratify=targets
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    proba = clf.predict_proba(x_test)
    metrics = {
        "seed": seed,
        "accuracy": float(accuracy_score(y_test, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "n_test": int(len(y_test)),
    }
    return metrics, proba, y_test, clf


def ece_from_proba(proba: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(labels), 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        ece += (float(mask.sum()) / n) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)
