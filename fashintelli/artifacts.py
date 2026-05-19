from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from .utils import load_json, save_json

# Manage saving and loading of trained ML models, metrics, leaderboard files,
# SHAP importance, permutation importance, and other pipeline artifact outputs.
@dataclass
class ArtifactPaths:
    model_path: Path
    metrics_path: Path
    leaderboard_path: Path
    permutation_importance_path: Path
    shap_importance_path: Path

    @classmethod
    def for_task(cls, artifacts_dir: Path, task: str) -> "ArtifactPaths":
        return cls(
            model_path=artifacts_dir / f"{task}_model.joblib",
            metrics_path=artifacts_dir / f"{task}_metrics.json",
            leaderboard_path=artifacts_dir / f"{task}_leaderboard.csv",
            permutation_importance_path=artifacts_dir / f"{task}_perm_importance.csv",
            shap_importance_path=artifacts_dir / f"{task}_shap_importance.csv",
        )


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    return joblib.load(path)


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    save_json(metrics, path)


def load_metrics(path: Path) -> Dict[str, Any]:
    return load_json(path)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
