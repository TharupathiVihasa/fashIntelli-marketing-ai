from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from .utils import get_logger

# Stores the summary of what happened during cleaning
@dataclass
class CleaningReport:
    n_rows_before: int
    n_rows_after: int
    n_duplicates_dropped: int
    missing_before: Dict[str, int]
    missing_after: Dict[str, int]


class DataCleaner:
    def __init__(self, logger_name: str = "fashintelli.cleaner") -> None:
        self.logger = get_logger(logger_name)

    @staticmethod
    def _missing_counts(df: pd.DataFrame) -> Dict[str, int]:
        return df.isna().sum().to_dict()

    def basic_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningReport]:
        df = df.copy()
        n_before = len(df)
        missing_before = self._missing_counts(df)

        before_dups = len(df)
        df = df.drop_duplicates()
        dups = before_dups - len(df)

        df.columns = [str(c).strip() for c in df.columns]

        for c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype(int)

        for c in df.columns:
            if df[c].isna().any():
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].median())
                else:
                    mode = df[c].mode(dropna=True)
                    fill = mode.iloc[0] if len(mode) else ""
                    df[c] = df[c].fillna(fill)

        missing_after = self._missing_counts(df)

        return df, CleaningReport(
            n_rows_before=n_before,
            n_rows_after=len(df),
            n_duplicates_dropped=dups,
            missing_before=missing_before,
            missing_after=missing_after,
        )

    def clean_survey(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, CleaningReport]:
        df2, rep = self.basic_clean(df)
        if target_col not in df2.columns:
            raise KeyError(f"Survey target column '{target_col}' not found.")
        return df2, rep

    def clean_social(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, CleaningReport]:
        df2, rep = self.basic_clean(df)
        if target_col not in df2.columns:
            raise KeyError(f"Social target column '{target_col}' not found.")
        return df2, rep
