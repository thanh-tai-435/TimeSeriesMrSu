"""Run advanced analyses beyond the base task list."""

from src.advanced import run_advanced_analysis
from src.config import ensure_directories


def main() -> None:
    ensure_directories()
    run_advanced_analysis()
    print("Saved advanced analysis tables, figures, and model rationale.")


if __name__ == "__main__":
    main()
