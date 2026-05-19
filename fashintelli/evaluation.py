from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib


matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    brier_score_loss,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    brier: Optional[float]

    #auto calculates all evaluation metrics
    @classmethod
    def from_predictions(cls, y_true, y_pred, y_proba: Optional[np.ndarray] = None) -> "ClassificationMetrics":
        roc_auc = None
        brier = None
        if y_proba is not None:
            try:
                roc_auc = float(roc_auc_score(y_true, y_proba))
            except Exception:
                roc_auc = None
            try:
                brier = float(brier_score_loss(y_true, y_proba))
            except Exception:
                brier = None

        return cls(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            roc_auc=roc_auc,
            brier=brier,
        )


def plot_confusion(y_true, y_pred, *, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_roc(y_true, y_proba, *, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pr(y_true, y_proba, *, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_calibration(y_true, y_proba, *, title: str, out_path: Path, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def infer_group_from_onehot(df: pd.DataFrame, prefix: str, default: str = "unknown") -> pd.Series:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return pd.Series([default] * len(df), index=df.index)
    mat = df[cols].copy()
    for c in cols:
        if mat[c].dtype == bool:
            mat[c] = mat[c].astype(int)
    idx = mat.values.argmax(axis=1)
    labels = [cols[i].replace(prefix, "", 1) for i in idx]
    return pd.Series(labels, index=df.index)

# perform subgroup-based performance analysis
def subgroup_report(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    group_series: pd.Series,
    min_group_size: int = 20,
) -> pd.DataFrame:
    """Compute subgroup metrics safely.

    Important: `y_true` typically preserves the original row index from the source
    dataframe, while `y_pred` is often a NumPy array (positional). When we slice by
    subgroup indices (e.g., 349, 512, ...), NumPy positional indexing will crash.

    We fix this by converting `y_pred` to a pandas Series aligned to the same index
    as `y_true` (or `group_series`).
    """

    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)

    # Align group labels to y_true index
    if not isinstance(group_series, pd.Series):
        group_series = pd.Series(group_series, index=y_true.index)
    else:
        group_series = group_series.reindex(y_true.index)

    # Align predictions to y_true index
    if isinstance(y_pred, pd.Series):
        y_pred_s = y_pred.reindex(y_true.index)
    else:
        y_pred_arr = np.asarray(y_pred)
        if len(y_pred_arr) == len(y_true):
            y_pred_s = pd.Series(y_pred_arr, index=y_true.index)
        elif len(y_pred_arr) == len(group_series):
            y_pred_s = pd.Series(y_pred_arr, index=group_series.index)
        else:
            # Fallback: best-effort positional alignment
            y_pred_s = pd.Series(y_pred_arr)

    # Drop rows where group label is missing
    valid = group_series.notna()
    y_true = y_true.loc[valid]
    y_pred_s = y_pred_s.loc[valid]
    group_series = group_series.loc[valid]

    rows = []
    for g, idx in group_series.groupby(group_series).groups.items():
        if len(idx) < int(min_group_size):
            continue

        yt = y_true.loc[idx]
        yp = y_pred_s.loc[idx]

        rows.append(
            {
                "group": str(g),
                "n": int(len(idx)),
                "accuracy": float(accuracy_score(yt, yp)),
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["group", "n", "accuracy", "precision", "recall", "f1"])
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
