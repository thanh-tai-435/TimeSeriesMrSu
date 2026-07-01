"""Small model utilities shared by the final pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TABLES_DIR


def load_feature_columns(path: str | Path = TABLES_DIR / "feature_columns.csv") -> list[str]:
    """Load feature column names produced by `main_features.py`."""
    return pd.read_csv(path)["feature"].tolist()
