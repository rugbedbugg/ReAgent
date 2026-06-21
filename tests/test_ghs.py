"""GHS module tests: severity mapping, caching, and offline fallback (no network)."""

from reagent.core.models import Molecule, Reaction, Route
from reagent.features.extract import compute_features
from reagent.features.ghs import GHSClient, _score_from_codes, enrich_ghs
from reagent.features.scoring import deterministic_scores


def test_severity_worst_tier_wins():
    assert _score_from_codes({"H314", "H350"}) == 0.10  # confirmed carcinogen dominates
    assert _score_from_codes({"H314", "H335"}) == 0.45  # corrosive over irritant
    assert _score_from_codes({"H335"}) == 0.65  # irritant
    assert _score_from_codes({"H330"}) == 0.10  # fatal if inhaled
    assert _score_from_codes(set()) == 1.0


def test_client_uses_cache_and_no_network_when_disabled(tmp_path):
    client = GHSClient(cache_path=tmp_path / "ghs.json", enabled=False)
    # disabled + empty cache -> None (no network attempted), so callers fall back
    assert client.h_codes("CC(=O)Cl") is None
    # seed the cache and confirm it is used without network
    client._cache["CC(=O)Cl"] = ["H314", "H225"]
    assert client.h_codes("CC(=O)Cl") == ["H314", "H225"]


def test_enrich_uses_cached_ghs_and_scorer_prefers_it(tmp_path):
    client = GHSClient(cache_path=tmp_path / "ghs.json", enabled=False)
    client._cache["CC(=O)Cl"] = ["H314"]  # corrosive -> 0.45 tier
    client._cache["CC(=O)Oc1ccccc1C(=O)O"] = ["H300"]  # target, excluded from scoring
    client._cache["O=C(O)c1ccccc1O"] = []

    route = Route(
        target="CC(=O)Oc1ccccc1C(=O)O",
        reactions=[Reaction(product="CC(=O)Oc1ccccc1C(=O)O",
                            precursors=["CC(=O)Cl", "O=C(O)c1ccccc1O"],
                            metadata={"policy_probability": 0.7})],
        leaves=[Molecule(smiles="CC(=O)Cl", in_stock=True),
                Molecule(smiles="O=C(O)c1ccccc1O", in_stock=True)],
        solved=True,
    )
    compute_features(route)
    assert enrich_ghs(route, client) is True
    # target's H300 is excluded; worst reagent code is CC(=O)Cl's H314 -> 0.45
    assert route.features["safety"]["ghs_safety"] == 0.45
    assert deterministic_scores(route)["safety"] == 0.45


def test_enrich_returns_false_when_no_ghs_data(tmp_path):
    client = GHSClient(cache_path=tmp_path / "ghs.json", enabled=False)  # no cache, offline
    route = Route(target="CCO", leaves=[Molecule(smiles="CCO", in_stock=True)], solved=True)
    compute_features(route)
    assert enrich_ghs(route, client) is False
    assert "ghs_safety" not in route.features["safety"]  # keeps offline Brenk safety
