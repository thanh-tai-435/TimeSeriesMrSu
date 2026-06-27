"""Run frequency-domain seasonality diagnostics."""

from src.config import ensure_directories
from src.spectrum import run_spectrum_analysis


def main() -> None:
    ensure_directories()
    run_spectrum_analysis()
    print("Saved spectrum analysis tables and figures.")


if __name__ == "__main__":
    main()
