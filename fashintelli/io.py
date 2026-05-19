from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# Load survey and social media datasets from CSV or Excel files
# and convert them into pandas DataFrames for preprocessing and ML pipeline usage.
def load_excel(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if sheet_name is None:
        return pd.read_excel(path)
    return pd.read_excel(path, sheet_name=sheet_name)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_survey_data(
    *,
    excel_path: Optional[Path] = None,
    excel_sheet: Optional[str] = None,
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    if csv_path is not None and csv_path.exists():
        return load_csv(csv_path)
    if excel_path is None:
        raise FileNotFoundError("Survey dataset not provided.")
    if not excel_path.exists():
        raise FileNotFoundError(f"Survey Excel not found: {excel_path}")
    return load_excel(excel_path, sheet_name=excel_sheet)


def load_social_data(
    *,
    excel_path: Optional[Path] = None,
    excel_sheet: Optional[str] = None,
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    if csv_path is not None and csv_path.exists():
        return load_csv(csv_path)
    if excel_path is None:
        raise FileNotFoundError("Social dataset not provided.")
    if not excel_path.exists():
        raise FileNotFoundError(f"Social Excel not found: {excel_path}")
    return load_excel(excel_path, sheet_name=excel_sheet)
