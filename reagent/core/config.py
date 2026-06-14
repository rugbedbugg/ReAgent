"""Project configuration and data-file discovery."""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = two levels up from this file (reagent/core/config.py).
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("REAGENT_DATA", ROOT / "data"))


def load_dotenv() -> None:
    """Load ``KEY=value`` lines from a repo-root .env into the environment.

    Existing environment variables win, so a real export always overrides the
    file. Missing file is a no-op.
    """
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def aizynth_config() -> Path:
    """Path to the AiZynthFinder config.yml produced by download_public_data."""
    cfg = DATA_DIR / "config.yml"
    if not cfg.exists():
        raise FileNotFoundError(
            f"AiZynthFinder config not found at {cfg}. Run:\n"
            f'  download_public_data "{DATA_DIR}"'
        )
    return cfg
