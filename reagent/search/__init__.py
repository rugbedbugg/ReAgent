"""Search backends.

AiZynthFinder ships four tree searches over the same single-step model. Only
MCTS is wired up by default; the rest are selected by handing
``config.search.algorithm`` a fully-qualified class path, which this module maps
from short names so callers never spell one out.

A different search over an unchanged model is one of the few levers that can
surface different routes without new training data or a better policy.
"""

from __future__ import annotations

# Short name -> the class AiZynthFinder loads for it. "mcts" is special-cased
# inside AiZynthFinder itself and must stay the bare string.
ALGORITHMS: dict[str, str] = {
    "mcts": "mcts",
    "retrostar": "aizynthfinder.search.retrostar.search_tree.SearchTree",
    "dfpn": "aizynthfinder.search.dfpn.search_tree.SearchTree",
    "breadth-first": "aizynthfinder.search.breadth_first.search_tree.SearchTree",
}


def resolve(name: str) -> str:
    """Map a short algorithm name to what AiZynthFinder expects.

    Retro* needs no extra model: left unconfigured its molecule cost defaults to
    ZeroMoleculeCost, so it runs on the files already downloaded.
    """
    key = name.strip().lower()
    if key not in ALGORITHMS:
        raise ValueError(
            f"Unknown search algorithm {name!r}. Choose one of: "
            + ", ".join(sorted(ALGORITHMS))
        )
    return ALGORITHMS[key]
