from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

from .utils import get_logger


#   Implement Explainable AI techniques including permutation importance,
# SHAP, and LIME to interpret and visualize machine learning predictions.
def _safe_get_feature_names(preprocess) -> List[str]:
    try:
        names = preprocess.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return []

@dataclass
class PermutationImportanceResult:
    feature_names: List[str]
    importances_mean: List[float]
    importances_std: List[float]


class ExplainabilityEngine:
    def __init__(self, logger_name: str = "fashintelli.xai") -> None:
        self.logger = get_logger(logger_name)

    def permutation_importance(
        self,
        pipeline: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        n_repeats: int = 18,
        random_state: int = 42,
        scoring: str = "f1",
    ) -> PermutationImportanceResult:
        r = permutation_importance(
            pipeline,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=-1,
        )

        feature_names: List[str] = []
        try:
            pre = pipeline.named_steps.get("preprocess")

            feature_names = _safe_get_feature_names(pre)


        except Exception:
            feature_names = []

        if not feature_names or len(feature_names) != len(r.importances_mean):
            feature_names = [f"f_{i}" for i in range(len(r.importances_mean))]

        return PermutationImportanceResult(
            feature_names=feature_names,
            importances_mean=[float(x) for x in r.importances_mean],
            importances_std=[float(x) for x in r.importances_std],
        )

    def try_shap_global(
        self,
        pipeline: BaseEstimator,
        X: pd.DataFrame,
        *,
        max_samples: int = 400,
        random_state: int = 42,
    ) -> Optional[pd.DataFrame]:
        try:
            import shap  # type: ignore
        except Exception as e:
            self.logger.warning(f"SHAP unavailable: {e}")
            return None

        rng = np.random.default_rng(random_state)
        if len(X) > max_samples:
            Xs = X.iloc[rng.choice(len(X), size=max_samples, replace=False)]
        else:
            Xs = X

        try:
            pre = pipeline.named_steps["preprocess"]
            model = pipeline.named_steps["model"]

            X_trans = pre.transform(Xs)
            feature_names = _safe_get_feature_names(pre)
            if not feature_names:
                feature_names = [f"f_{i}" for i in range(X_trans.shape[1])]

            try:
                X_dense = X_trans.toarray()
            except Exception:
                X_dense = np.asarray(X_trans)

            explainer = None
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                explainer = None

            if explainer is None:
                try:
                    explainer = shap.LinearExplainer(model, X_dense)
                except Exception:
                    explainer = None

            if explainer is None:
                explainer = shap.Explainer(model, X_dense)

            shap_values = explainer(X_dense)
            vals = shap_values.values
            if vals.ndim == 3:
                vals = vals[:, :, 1]

            imp = np.mean(np.abs(vals), axis=0)
            out = pd.DataFrame({"feature": feature_names, "shap_importance": imp})
            return out.sort_values("shap_importance", ascending=False).reset_index(drop=True)
        except Exception as e:
            self.logger.warning(f"SHAP computation failed: {e}")
            return None

    def try_lime_local(
        self,
        pipeline: BaseEstimator,
        X_train: pd.DataFrame,
        X_instance: pd.DataFrame,
        *,
        class_names: Tuple[str, str] = ("low", "high"),
        random_state: int = 42,
    ) -> Optional[List[Tuple[str, float]]]:
        try:
            from lime.lime_tabular import LimeTabularExplainer  # type: ignore
        except Exception as e:
            self.logger.warning(f"LIME unavailable: {e}")
            return None

        try:
            pre = pipeline.named_steps["preprocess"]
            model = pipeline.named_steps["model"]

            Xtr = pre.transform(X_train)
            Xin = pre.transform(X_instance)

            try:
                Xtr_dense = Xtr.toarray()
                Xin_dense = Xin.toarray()
            except Exception:
                Xtr_dense = np.asarray(Xtr)
                Xin_dense = np.asarray(Xin)

            feature_names = _safe_get_feature_names(pre)
            if not feature_names:
                feature_names = [f"f_{i}" for i in range(Xtr_dense.shape[1])]

            explainer = LimeTabularExplainer(
                training_data=Xtr_dense,
                feature_names=feature_names,
                class_names=list(class_names),
                discretize_continuous=True,
                random_state=random_state,
            )

            def _predict_fn(z: np.ndarray) -> np.ndarray:
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(z)
                scores = model.decision_function(z)
                scores = 1 / (1 + np.exp(-scores))
                return np.column_stack([1 - scores, scores])

            exp = explainer.explain_instance(
                data_row=Xin_dense[0],
                predict_fn=_predict_fn,
                num_features=12,
            )

            return exp.as_list()
        except Exception as e:
            self.logger.warning(f"LIME computation failed: {e}")
            return None
