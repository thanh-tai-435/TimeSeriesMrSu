"""Run residual diagnostics and prediction intervals."""

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, TABLES_DIR, ensure_directories
from src.diagnostics import run_diagnostics


def main() -> None:
    ensure_directories()
    run_diagnostics(
        features_path=PROCESSED_DATA_DIR / "fresh50k_features.parquet",
        tables_dir=TABLES_DIR,
        figures_dir=FIGURES_DIR,
    )
    print(f"Saved diagnostics tables to: {TABLES_DIR}")
    print(f"Saved diagnostics figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
