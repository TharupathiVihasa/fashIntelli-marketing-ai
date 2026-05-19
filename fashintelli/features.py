from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .utils import get_logger

# This creates extra useful features from existing dataset columns.
class FeatureEngineer:
    def __init__(self, logger_name: str = "fashintelli.features") -> None:
        self.logger = get_logger(logger_name)

    @staticmethod
    def infer_onehot_label(df: pd.DataFrame, prefix: str, default: str = "unknown") -> pd.Series:
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

    def add_platform_brand_labels(
        self,
        social_df: pd.DataFrame,
        *,
        platform_default: str = "instagram",
        platform_col: str = "platform",
        brand_col: str = "brand",
    ) -> pd.DataFrame:
        df = social_df.copy()
        if platform_col not in df.columns:
            df[platform_col] = platform_default

        if any(c.startswith("ownerUsername_") for c in df.columns):
            df[brand_col] = self.infer_onehot_label(df, "ownerUsername_", default="unknown")
        elif brand_col not in df.columns:
            df[brand_col] = "unknown"

        return df

    # groups social media dataset by platform and calculates the average engagement score for each platform
    def build_platform_engagement_index(
        self,
        social_df_scored: pd.DataFrame,
        score_col: str = "engagement_score",
        platform_col: str = "platform",
    ) -> Dict[str, float]:
        if platform_col not in social_df_scored.columns:
            return {"unknown": float(np.nanmean(social_df_scored[score_col].values))}
        grp = social_df_scored.groupby(platform_col)[score_col].mean().to_dict()
        return {str(k): float(v) for k, v in grp.items()}

    def attach_engagement_index_to_survey(
        self,
        survey_df: pd.DataFrame,
        platform_index: Dict[str, float],
        *,
        platform_prefix: str = "platform_",
        out_col: str = "platform_engagement_index",
    ) -> pd.DataFrame:
        """Attach the (predicted) platform engagement index to each survey row.

        Supports both:
        - One-hot platform columns (e.g., platform_instagram, platform_facebook, ...)
        - A single categorical column (e.g., 'platform' or 'primary_platform')
        """
        df = survey_df.copy()

        # Case A) One-hot encoded platforms
        platform_cols = [c for c in df.columns if c.startswith(platform_prefix)]
        if platform_cols:
            weights = []
            for c in platform_cols:
                plat = c.replace(platform_prefix, "", 1).strip().lower()
                weights.append(float(platform_index.get(plat, platform_index.get("unknown", 0.0))))
            weights = np.array(weights, dtype=float)

            mat = df[platform_cols].copy()
            for c in platform_cols:
                if mat[c].dtype == bool:
                    mat[c] = mat[c].astype(int)
            M = mat.values.astype(float)

            denom = np.clip(M.sum(axis=1), 1.0, None)
            df[out_col] = (M @ weights) / denom
            return df

        # Case B) Single platform column
        platform_col = None
        for cand in ["platform", "primary_platform"]:
            if cand in df.columns:
                platform_col = cand
                break

        if platform_col is not None:
            def _norm(p: str) -> str:
                s = str(p).strip().lower()
                # common variants
                if s in {"fb"}:
                    return "facebook"
                if s in {"insta"}:
                    return "instagram"
                if s in {"tik tok"}:
                    return "tiktok"
                return s

            df[out_col] = df[platform_col].apply(lambda p: float(platform_index.get(_norm(p), platform_index.get("unknown", 0.0))))
            return df

        # Fallback
        df[out_col] = float(platform_index.get("unknown", 0.0))
        return df

        # weights = []
        # for c in platform_cols:
        #     plat = c.replace(platform_prefix, "", 1).lower()
        #     weights.append(float(platform_index.get(plat, platform_index.get("unknown", 0.0))))
        # weights = np.array(weights, dtype=float)
        #
        # mat = df[platform_cols].copy()
        # for c in platform_cols:
        #     if mat[c].dtype == bool:
        #         mat[c] = mat[c].astype(int)
        # M = mat.values.astype(float)
        #
        # denom = np.clip(M.sum(axis=1), 1.0, None)
        # df[out_col] = (M @ weights) / denom
        # return df
