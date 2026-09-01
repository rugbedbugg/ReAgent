# Contributing to ReAgent

ReAgent combines retrosynthetic planning, cheminformatics, optimization, and LLM-backed interpretation. Changes should remain evidence-grounded and reproducible.

## Development setup

The project supports Python 3.10 and 3.11 and is pinned locally to uv-managed 3.11:

```sh
uv venv --python 3.11
uv pip install -e ".[dev]"
uv run pytest
uv run ruff check .
```

Some workflows require pretrained models, building-block data, or provider credentials described in `README.md`. Keep those assets and secrets outside Git.

## Change guidelines

- Keep package code under `reagent/` and tests under `tests/`.
- Use the configured 100-character Ruff line length and Python 3.11 target.
- Add deterministic fixtures for route ranking, feature vectors, and evaluation changes.
- Isolate live LLM or network behavior behind interfaces that can be tested without external calls.
- Explain changes to scoring weights, chemistry constraints, or evaluation splits.

## Pull requests

Include pytest and Ruff results, describe the data/model configuration used, and distinguish measured outcomes from qualitative examples.
