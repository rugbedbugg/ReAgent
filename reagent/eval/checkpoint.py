"""Atomic, per-target search checkpoints for interrupted evaluations.

A directory belongs to one search configuration. Only completed searches are
saved, including empty results and clock-limit status. Ranking is recomputed
from full routes, so changing a weight profile does not require another search.
Use one evaluation process per checkpoint directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, StrictBool

from reagent.core.models import Route


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def search_identity(config: Path, kwargs: dict, max_routes: int) -> dict:
    """Fingerprint parameters, model/stock contents and search implementation."""
    import yaml

    from reagent.core.config import DATA_DIR
    from reagent.singlestep.stock import cache_path_for

    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    assets = {str(config.resolve()): _digest(config)}

    def collect(value):
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str) and Path(value).is_file():
            path = Path(value).resolve()
            assets[str(path)] = _digest(path)

    for key in kwargs["expansion"]:
        collect(settings.get("expansion", {}).get(key))
    collect(settings.get("filter", {}).get(kwargs["expansion"][0]))
    if kwargs["hashed_stock"]:
        stock = Path(kwargs["stock_cache"]) if kwargs["stock_cache"] else cache_path_for(
            DATA_DIR / "zinc_stock.hdf5"
        )
        assets[str(stock.resolve())] = _digest(stock)
    else:
        collect(settings.get("stock"))

    root = Path(__file__).resolve().parents[1]
    implementation = {
        str(path.relative_to(root)): _digest(path)
        for directory in ("core", "singlestep", "search")
        for path in sorted((root / directory).glob("*.py"))
    }
    parameters = dict(kwargs)
    if parameters["stock_cache"]:
        parameters["stock_cache"] = str(Path(parameters["stock_cache"]).resolve())
    return {
        "schema": 1,
        "parameters": parameters,
        "max_routes": max_routes,
        "assets": assets,
        "implementation": implementation,
        "dependencies": {
            name: version(name)
            for name in ("aizynthfinder", "rdkit", "numpy", "onnxruntime", "reaction-utils")
        },
    }


def _write(path: Path, value: dict) -> None:
    """Replace only after the entire JSON document has reached disk."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    routes: list[Route]
    time_capped: StrictBool


class Checkpoint:
    def __init__(self, directory: Path, identity: dict):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != identity:
                raise ValueError(
                    "Checkpoint search settings, assets or implementation differ. "
                    "Use a new --checkpoint directory for this experiment."
                )
        else:
            if any(directory.iterdir()):
                raise ValueError("Checkpoint directory has files but no manifest; use a new directory.")
            _write(manifest, identity)

    def _path(self, target: str) -> Path:
        key = hashlib.sha256(target.encode()).hexdigest()
        return self.directory / f"{key}.json"

    def load(self, target: str) -> SearchResult | None:
        path = self._path(target)
        if not path.exists():
            return None
        result = SearchResult.model_validate_json(path.read_text(encoding="utf-8"))
        if result.target != target or any(route.target != target for route in result.routes):
            raise ValueError(f"Checkpoint target mismatch in {path}")
        return result

    def save(self, target: str, routes: list[Route], time_capped: bool) -> None:
        result = SearchResult(target=target, routes=routes, time_capped=time_capped)
        if any(route.target != target for route in routes):
            raise ValueError("Cannot checkpoint routes for a different target")
        _write(self._path(target), result.model_dump(mode="json"))
