"""Run the Fresh50K stockout-aware forecasting pipeline end to end."""

from __future__ import annotations

import subprocess
import sys


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
        full_command = [sys.executable, *command]
        print(f"\n>>> {' '.join(full_command)}")
        subprocess.run(full_command, check=True)


if __name__ == "__main__":
    main()
