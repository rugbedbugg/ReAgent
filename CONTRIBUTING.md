# Contributing to ReAgent

ReAgent combines retrosynthetic planning, cheminformatics, optimization, and LLM-backed interpretation. Changes should remain evidence-grounded and reproducible.

## Development setup

The project supports Python 3.10 and 3.11. `mise.toml` pins the local toolchain to
Python 3.11 with uv and creates `.venv` on entering the directory; CI runs both
versions.

```sh
mise trust
mise install        # Python 3.11 + uv; creates .venv
mise run install    # editable install with dev extras
mise run test
mise run lint
```

Without mise (`.python-version` selects 3.11 for uv):

```sh
uv venv
uv pip install -e ".[dev]"
uv run pytest
uv run ruff check .
```

Some workflows require pretrained models, building-block data, or provider credentials described in `README.md`. Keep those assets and secrets outside Git.

## Change guidelines

- Keep package code under `reagent/` and tests under `tests/`.
- Ruff config lives in `pyproject.toml`: 100-character line length, a `py310`
  target (the lowest supported interpreter), and an explicitly pinned rule set
  so a Ruff upgrade cannot silently change what CI enforces.
- Add deterministic fixtures for route ranking, feature vectors, and evaluation changes.
- Isolate live LLM or network behavior behind interfaces that can be tested without external calls.
- Explain changes to scoring weights, chemistry constraints, or evaluation splits.

## Pull requests

Include pytest and Ruff results, describe the data/model configuration used, and distinguish measured outcomes from qualitative examples.
