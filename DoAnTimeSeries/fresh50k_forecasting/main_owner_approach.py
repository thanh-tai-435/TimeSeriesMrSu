"""Run the FreshRetailNet-50K paper/repo-aligned two-stage approach."""

from src.config import ensure_directories
from src.owner_approach import run_owner_aligned_pipeline


def main() -> None:
    ensure_directories()
    run_owner_aligned_pipeline()
    print("Saved owner-aligned latent demand recovery and forecasting outputs.")


if __name__ == "__main__":
    main()
