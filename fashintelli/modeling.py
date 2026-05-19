from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from .config import ModelConfig
from .utils import get_logger, seed_everything


@dataclass
class TrainResult:
    best_model_name: str
    best_estimator: BaseEstimator
    leaderboard: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

# automatically detect feature types
def infer_column_groups(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str], List[str]]:
    X = df.drop(columns=[target_col])

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) or X[c].dtype == bool]
    cat_cols: List[str] = []
    text_cols: List[str] = []

    for c in X.columns:
        if c in numeric_cols:
            continue
        if pd.api.types.is_string_dtype(X[c]):
            avg_len = X[c].dropna().astype(str).map(len).mean() if X[c].notna().any() else 0
            if avg_len >= 30:
                text_cols.append(c)
            else:
                cat_cols.append(c)
        else:
            cat_cols.append(c)

    return numeric_cols, cat_cols, text_cols


def to_1d_text_array(x):
    """Convert 2D single-column text input into a 1D string array.

    This helper is defined at module level so that pipelines using it
    are picklable with joblib (Streamlit + artifact saving).
    """
    # Pandas inputs (preferred)
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            s = x.iloc[:, 0]
        else:
            # If multiple columns are accidentally passed, concatenate them.
            s = x.astype(str).agg(" ".join, axis=1)
        return s.fillna("").astype(str).values

    if isinstance(x, pd.Series):
        return x.fillna("").astype(str).values

    # Numpy / list-like inputs
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    return pd.Series(arr).fillna("").astype(str).values


def build_preprocessor(
    *,
    numeric_cols: List[str],
    cat_cols: List[str],
    text_cols: List[str],
    max_tfidf_features: int = 4000,
) -> ColumnTransformer:
    transformers = []

    if numeric_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("num", num_pipe, numeric_cols))

    if cat_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    if text_cols:
        tcol = text_cols[0]

        text_pipe = Pipeline(
            steps=[
                ("to1d", FunctionTransformer(to_1d_text_array, validate=False)),
                ("tfidf", TfidfVectorizer(max_features=max_tfidf_features, ngram_range=(1, 2))),
            ]
        )
        transformers.append(("text", text_pipe, tcol))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

# Define Models - dummy,logreg,xgboost,randomforest
def candidate_models(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    models: Dict[str, Dict[str, Any]] = {}

    models["dummy"] = {
        "estimator": DummyClassifier(strategy="most_frequent"),
        "params": {},
    }

    models["logreg"] = {
        "estimator": LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            solver="liblinear",
        ),
        "params": {
            "model__C": np.logspace(-3, 2, 18),
            "model__penalty": ["l1", "l2"],
        },
    }

    models["random_forest"] = {
        "estimator": RandomForestClassifier(
            n_estimators=350,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "params": {
            "model__n_estimators": [200, 350, 500],
            "model__max_depth": [None, 6, 10, 16],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
    }

    try:
        from xgboost import XGBClassifier  # type: ignore

        models["xgboost"] = {
            "estimator": XGBClassifier(
                n_estimators=550,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            "params": {
                "model__n_estimators": [300, 550, 800],
                "model__max_depth": [3, 4, 6, 8],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__min_child_weight": [1, 3, 5],
                "model__gamma": [0, 0.5, 1.0],
            },
        }
    except Exception:
        pass

    return models


class ModelTrainer:
    def __init__(self, cfg: ModelConfig, logger_name: str = "fashintelli.trainer") -> None:
        self.cfg = cfg
        self.logger = get_logger(logger_name)
        seed_everything(cfg.random_state)

    def train_classification(
        self,
        df: pd.DataFrame,
        *,
        target_col: str,
        test_size: Optional[float] = None,
        split_col: Optional[str] = None,
    ) -> TrainResult:
        """Train a binary classifier with robust splitting.

        If a split column (e.g., ``split``) is present, the trainer will respect it
        (train/test) for reproducibility and to support datasets that already
        contain explicit partitions.
        """
        # Ensures label column exists
        if target_col not in df.columns:
            raise KeyError(f"target_col '{target_col}' not found.")

        df = df.copy()

        # Auto-detect a split column if not provided
        if split_col is None:
            for cand in ["split", "dataset_split", "set"]:
                if cand in df.columns:
                    split_col = cand
                    break

        use_explicit_split = split_col is not None and split_col in df.columns

        if use_explicit_split:
            split_vals = df[split_col].astype(str).str.strip().str.lower()
            train_mask = split_vals.isin(["train", "tr", "0"])
            test_mask = split_vals.isin(["test", "te", "1", "val", "valid", "validation"])

            # Fallback to random split if the split column is malformed
            if (not train_mask.any()) or (not test_mask.any()):
                use_explicit_split = False

        # Remove split column from feature space (avoid leakage)
        df_nosplit = df.drop(columns=[split_col]) if (split_col is not None and split_col in df.columns) else df

        if use_explicit_split:
            df_train = df_nosplit.loc[train_mask].copy()
            df_test = df_nosplit.loc[test_mask].copy()

            X_train = df_train.drop(columns=[target_col])
            y_train = df_train[target_col].astype(int)

            X_test = df_test.drop(columns=[target_col])
            y_test = df_test[target_col].astype(int)
        else:
            test_size = float(test_size) if test_size is not None else self.cfg.test_size

            X = df_nosplit.drop(columns=[target_col])
            y = df_nosplit[target_col].astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=self.cfg.random_state,
                stratify=y,
            )

        numeric_cols, cat_cols, text_cols = infer_column_groups(df_nosplit, target_col=target_col)
        self.logger.info(
            f"Inferred columns -> numeric={len(numeric_cols)} categorical={len(cat_cols)} text={len(text_cols)}"
        )
        pre = build_preprocessor(numeric_cols=numeric_cols, cat_cols=cat_cols, text_cols=text_cols)

        triage_cv = StratifiedKFold(
            n_splits=self.cfg.baseline_cv_folds,
            shuffle=True,
            random_state=self.cfg.random_state,
        )

        baseline_rows = []
        model_specs = candidate_models(self.cfg.random_state)

        for name, spec in model_specs.items():
            pipe = Pipeline(steps=[("preprocess", pre), ("model", spec["estimator"])])
            try:
                scores = cross_val_score(
                    pipe, X_train, y_train,
                    cv=triage_cv,
                    scoring=self.cfg.primary_metric,
                    n_jobs=self.cfg.n_jobs,
                )
                baseline = float(np.mean(scores))
            except Exception:
                baseline = float("nan")
            baseline_rows.append({"model": name, "baseline_cv_f1": baseline})

        baseline_df = pd.DataFrame(baseline_rows).sort_values("baseline_cv_f1", ascending=False).reset_index(drop=True)

        candidates = [m for m in baseline_df["model"].tolist() if m != "dummy"]
        top_for_tune = candidates[: max(1, self.cfg.tune_top_k)]

        tune_cv = StratifiedKFold(
            n_splits=self.cfg.tune_cv_folds,
            shuffle=True,
            random_state=self.cfg.random_state,
        )

        tuned_rows = []
        best_name = None
        best_estimator = None
        best_score = -1e9

        for name in top_for_tune:
            spec = model_specs[name]
            params = spec.get("params", {})
            base_pipe = Pipeline(steps=[("preprocess", pre), ("model", spec["estimator"])])

            if not params:
                base_pipe.fit(X_train, y_train)
                tuned_rows.append({"model": name, "tuned_cv_f1": np.nan})
                continue
            #searches for best hyperparameter
            search = RandomizedSearchCV(
                estimator=base_pipe,
                param_distributions=params,
                n_iter=int(self.cfg.search_iter),
                scoring={"f1": "f1", "roc_auc": "roc_auc"},
                refit=self.cfg.primary_metric,
                cv=tune_cv,
                n_jobs=self.cfg.n_jobs,
                random_state=self.cfg.random_state,
                verbose=0,
            )
            search.fit(X_train, y_train)
            tuned_f1 = float(search.best_score_)
            tuned_rows.append({"model": name, "tuned_cv_f1": tuned_f1})

            if tuned_f1 > best_score:
                best_score = tuned_f1
                best_name = name
                best_estimator = search.best_estimator_

        if best_estimator is None:
            best_name = str(baseline_df.iloc[0]["model"])
            best_estimator = Pipeline(steps=[("preprocess", pre), ("model", model_specs[best_name]["estimator"])])
            best_estimator.fit(X_train, y_train)

        leaderboard = baseline_df.merge(pd.DataFrame(tuned_rows), on="model", how="left")
        leaderboard["is_best"] = leaderboard["model"] == best_name
        leaderboard = leaderboard.sort_values(["is_best", "tuned_cv_f1", "baseline_cv_f1"], ascending=[False, False, False]).reset_index(drop=True)

        return TrainResult(
            best_model_name=str(best_name),
            best_estimator=best_estimator,
            leaderboard=leaderboard,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
