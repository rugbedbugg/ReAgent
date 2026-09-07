"""Recompute profile reports from saved routes, without loading a search model."""

import argparse
import hashlib
import json

from reagent.core.chem import canonical
from reagent.eval.checkpoint import Checkpoint
from reagent.eval.harness import evaluate
from reagent.eval.targets import HARD_TARGETS, TARGETS
from reagent.optimize.aggregate import WEIGHT_PROFILES


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--hard", action="store_true")
    args = parser.parse_args()
    saved = Checkpoint.open_readonly(args.checkpoint)
    targets = HARD_TARGETS if args.hard else TARGETS
    records = {}
    for name, smiles in targets:
        target = canonical(smiles) or smiles
        record = saved.load(target)
        if record is None:
            raise ValueError(f"Missing checkpoint for {name}")
        records[target] = record

    result = {
        "weights": WEIGHT_PROFILES,
        "manifest": json.loads((args.checkpoint / "manifest.json").read_text()),
        "record_sha256": {
            target: hashlib.sha256(record.model_dump_json().encode()).hexdigest()
            for target, record in records.items()
        },
        "time_capped_targets": sum(record.time_capped for record in records.values()),
        "candidate_counts": {
            name: len(records[canonical(smiles) or smiles].routes) for name, smiles in targets
        },
        "profiles": {
            profile: evaluate(
                targets, lambda smiles: records[canonical(smiles) or smiles].routes, weights,
            ) for profile, weights in WEIGHT_PROFILES.items()
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
