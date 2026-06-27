"""RAG-layer tests: embedder, index, and retriever grounding (offline)."""

import numpy as np

from reagent.core.models import Reaction, Route
from reagent.rag.embed import ReactionFingerprintEmbedder
from reagent.rag.index import FingerprintIndex
from reagent.rag.retrieve import PrecedentRetriever

# A couple of real USPTO-style retro templates.
T_ESTER = "[C:2]-[NH2;D1;+0:1]>>C-C-C-C-O-C(=O)-[NH;D2;+0:1]-[C:2]"
T_AMIDE = "[C:1](=[O:2])-[NH:3]>>[C:1](=[O:2])-[OH].[NH2:3]"


def _index() -> FingerprintIndex:
    emb = ReactionFingerprintEmbedder()
    records = [
        {"template_hash": "aaaa1111", "retro_template": T_ESTER, "library_occurence": 50},
        {"template_hash": "bbbb2222", "retro_template": T_AMIDE, "library_occurence": 900},
    ]
    matrix = np.vstack([emb.embed(r["retro_template"]) for r in records]).astype(np.uint8)
    return FingerprintIndex(matrix, records)


def test_embedder_shape_and_invalid():
    emb = ReactionFingerprintEmbedder()
    vec = emb.embed(T_ESTER)
    assert vec is not None and vec.shape == (emb.n_bits,)
    assert set(np.unique(vec)).issubset({0, 1})
    assert emb.embed("not a reaction") is None


def test_index_self_query_is_top():
    idx = _index()
    emb = ReactionFingerprintEmbedder()
    hits = idx.search(emb.embed(T_AMIDE), k=2)
    assert hits[0][0]["template_hash"] == "bbbb2222"
    assert hits[0][1] == 1.0


def test_index_save_load_roundtrip(tmp_path):
    idx = _index()
    p = tmp_path / "idx.npz"
    idx.save(p)
    loaded = FingerprintIndex.load(p)
    assert len(loaded) == 2
    emb = ReactionFingerprintEmbedder()
    assert loaded.search(emb.embed(T_ESTER), k=1)[0][0]["template_hash"] == "aaaa1111"


def test_retriever_grounds_route():
    from reagent.features.extract import compute_features

    retriever = PrecedentRetriever(index=_index())
    route = Route(
        target="CCN",
        reactions=[Reaction(product="CCN", precursors=["CCO"], metadata={"template": T_ESTER})],
    )
    compute_features(route)
    retriever.ground_route(route, k=1)

    precedents = route.reactions[0].metadata["precedents"]
    assert len(precedents) == 1
    assert precedents[0]["template_hash"] == "aaaa1111"  # self-match ranks top
    assert precedents[0]["similarity"] == 1.0
    assert route.features["feasibility"]["precedent_count"] == 1
    assert route.features["feasibility"]["max_precedent_occurence"] == 50
