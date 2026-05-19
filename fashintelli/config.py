from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Manages project folder paths and machine learning configuration settings.
@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def sample_data_dir(self) -> Path:
        return self.data_dir / "sample"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.outputs_dir / "logs"


@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    test_size: float = 0.2

    baseline_cv_folds: int = 2
    tune_cv_folds: int = 2
    tune_top_k: int = 2
    search_iter: int = 3
    n_jobs: int = 1

    primary_metric: str = "f1"
    secondary_metric: str = "roc_auc"
