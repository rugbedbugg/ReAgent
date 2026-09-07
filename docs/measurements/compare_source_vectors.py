"""Compare balanced and source ranking using the saved adaptive-run vectors.

Run from the repository root with:
    uv run --no-sync python docs/measurements/compare_source_vectors.py

These vectors contain solved routes only and omit route identities. Report
objective scores, not route lengths, leaf fractions, or route-identity changes.
Ties are safe only when every winning vector is identical; refuse to guess
which route the production signature tiebreak would select otherwise.
"""

import hashlib
import json
from pathlib import Path
from statistics import mean

from reagent.eval.targets import HARD_TARGETS, TARGETS
from reagent.optimize.aggregate import mode_weights, normalized_vectors, weighted_from_vector


def compare(data, targets):
    rows = []
    for name, _ in targets:
        vectors = data[name]
        row = {"target": name, "solved_candidates": len(vectors), "modes": {}}
        normalized = normalized_vectors(vectors)
        for mode in ("balanced", "source"):
            if not vectors:
                continue
            utilities = [weighted_from_vector(v, mode_weights(mode)) for v in normalized]
            winners = [i for i, u in enumerate(utilities) if u == max(utilities)]
            scores = vectors[winners[0]]
            if any(vectors[i] != scores for i in winners):
                raise ValueError(f"{name}/{mode}: tied distinct vectors need route identities")
            row["modes"][mode] = {"candidate_indices": winners, "scores": scores}
        rows.append(row)
    solved = [row for row in rows if row["solved_candidates"]]
    return {
        "targets": len(rows),
        "solved_targets": len(solved),
        "unsolved_targets": [r["target"] for r in rows if not r["solved_candidates"]],
        "changed_score_vectors": sum(
            r["modes"]["balanced"]["scores"] != r["modes"]["source"]["scores"]
            for r in solved
        ),
        "mean_scores_over_solved_targets": {
            mode: {
                objective: mean(r["modes"][mode]["scores"][objective] for r in solved)
                for objective in mode_weights(mode)
            } if solved else {}
            for mode in ("balanced", "source")
        },
        "per_target": rows,
    }


def main():
    path = Path(__file__).with_name("adaptive-vectors-49-targets.json")
    raw = path.read_bytes()
    data = json.loads(raw)
    result = {
        "input": path.name,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "Offline selection on saved solved-route vectors; no new search.",
        "weights": {mode: mode_weights(mode) for mode in ("balanced", "source")},
        "sets": {
            "moderate": compare(data, TARGETS),
            "hard": compare(data, HARD_TARGETS),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
