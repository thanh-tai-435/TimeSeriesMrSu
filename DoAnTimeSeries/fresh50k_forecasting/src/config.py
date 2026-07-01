"""Project configuration for Fresh50K forecasting."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"

RANDOM_SEED = 42
DATETIME_COL = "dt"
TARGET_COL = "sale_amount"
SERIES_COL = "series_id"


def ensure_directories() -> None:
    """Create expected data and output directories."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
