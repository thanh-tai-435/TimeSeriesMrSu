"""Run diagnostics for latent-demand imputation quality."""

from src.config import ensure_directories
from src.imputation_quality import run_imputation_quality_checks


def main() -> None:
    ensure_directories()
    run_imputation_quality_checks()
    print("Saved imputation quality diagnostics.")


if __name__ == "__main__":
    main()
