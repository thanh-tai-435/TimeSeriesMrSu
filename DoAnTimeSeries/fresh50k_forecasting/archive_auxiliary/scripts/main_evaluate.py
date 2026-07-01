"""Evaluate Fresh50K model predictions and generate report artifacts."""

from src.config import FIGURES_DIR, TABLES_DIR, ensure_directories
from src.evaluation import run_evaluation
from src.plots import run_plots


def main() -> None:
    ensure_directories()
    run_evaluation(TABLES_DIR)
    run_plots(TABLES_DIR, FIGURES_DIR)
    print(f"Saved evaluation tables to: {TABLES_DIR}")
    print(f"Saved report figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
