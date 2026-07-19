"""Run the Fresh50K stockout-aware forecasting pipeline end to end."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


COMMANDS = [
    ["main_eda.py", "--sample_frac", "0.1", "--frequency", "hourly"],
    ["main_features.py", "--sample_frac", "0.1", "--frequency", "hourly"],
    ["main_split.py", "--val_days", "7", "--test_days", "14"],
    ["main_owner_approach.py"],
    ["main_imputation_quality.py"],
    ["main_spectrum.py"],
]


def main() -> None:
    for command in COMMANDS:
        script = PROJECT_DIR / command[0]
        full_command = [sys.executable, str(script), *command[1:]]
        print(f"\n>>> {' '.join(full_command)}")
        subprocess.run(full_command, check=True, cwd=PROJECT_DIR)


if __name__ == "__main__":
    main()
