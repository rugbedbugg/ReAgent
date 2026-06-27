"""Load the USPTO template corpus that precedent retrieval indexes over."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reagent.core.config import DATA_DIR


def load_templates(path: str | Path | None = None, min_occurence: int = 1) -> list[dict]:
    """Return template records: hash, retro SMARTS, and corpus occurrence count.

    ``min_occurence`` filters out rarely-seen templates, trading recall for a
    smaller, faster index.
    """
    path = Path(path) if path else DATA_DIR / "uspto_templates.csv.gz"
    df = pd.read_csv(path, sep="\t")
    if min_occurence > 1:
        df = df[df["library_occurence"] >= min_occurence]
    return [
        {
            "template_hash": row.template_hash,
            "retro_template": row.retro_template,
            "library_occurence": int(row.library_occurence),
        }
        for row in df.itertuples(index=False)
    ]
