"""Project configuration and data-file discovery."""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = two levels up from this file (reagent/core/config.py).
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("REAGENT_DATA", ROOT / "data"))


def aizynth_config() -> Path:
    """Path to the AiZynthFinder config.yml produced by download_public_data."""
    cfg = DATA_DIR / "config.yml"
    if not cfg.exists():
        raise FileNotFoundError(
            f"AiZynthFinder config not found at {cfg}. Run:\n"
            f'  download_public_data "{DATA_DIR}"'
        )
    return cfg
