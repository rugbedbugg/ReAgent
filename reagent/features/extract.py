"""Deterministic feature extraction: raw chemistry facts, one block per objective.

``compute_features(route)`` fills ``route.features`` with a nested dict keyed by
objective. The agent layer reads these numbers and interprets them; it never
computes them. Values that would need data we do not hold offline (live supplier
counts, real prices, GHS categories, reaction solvents) are left out rather than
faked, and noted as proxies where a stand-in is used.
"""

from __future__ import annotations

from statistics import mean

from reagent.core.models import Route
from reagent.features import descriptors as d


def _reaction_depth(node: dict) -> int:
    """Longest chain of reactions from this node down (the linear sequence length)."""
    children = node.get("children") or []
    if node.get("type") == "reaction":
        return 1 + max((_reaction_depth(c) for c in children), default=0)
    return max((_reaction_depth(c) for c in children), default=0)


def efficiency(route: Route) -> dict:
    steps = route.num_steps
    lls = _reaction_depth(route.tree) if route.tree else steps
    return {
        "num_steps": steps,
        "longest_linear_sequence": lls,
        # 1.0 = fully linear; >1.0 = convergent (steps shared across branches).
        "convergence": (steps / lls) if lls else 1.0,
    }


def availability(route: Route) -> dict:
    total = len(route.leaves)
    in_stock = sum(1 for m in route.leaves if m.in_stock)
    return {
        "leaf_count": total,
        "in_stock_count": in_stock,
        "in_stock_fraction": (in_stock / total) if total else 0.0,
        "not_in_stock": [m.smiles for m in route.leaves if not m.in_stock],
    }


def cost(route: Route) -> dict:
    """Proxy only: no price database offline.

    Rises with the drivers of real cost: reaction steps (reagents, workup, yield
    loss), building blocks that must be sourced or synthesised because they are
    not in stock, the number of building blocks, and ring complexity as a rough
    price stand-in.
    """
    non_stock_ha = sum(d.heavy_atoms(m.smiles) for m in route.leaves if not m.in_stock)
    stock_ha = sum(d.heavy_atoms(m.smiles) for m in route.leaves if m.in_stock)
    n_building_blocks = len(route.leaves)
    # Synthetic accessibility of the building blocks stands in for their price:
    # harder-to-make blocks are rarer and dearer than raw size suggests.
    sa_total = round(sum(d.sa_score(m.smiles) for m in route.leaves), 2)
    cost_proxy = round(
        6 * route.num_steps
        + 4 * non_stock_ha
        + stock_ha
        + 2 * n_building_blocks
        + 2 * sa_total
    )
    return {
        "non_stock_heavy_atoms": non_stock_ha,
        "stock_heavy_atoms": stock_ha,
        "building_blocks": n_building_blocks,
        "synthetic_accessibility": sa_total,
        "num_steps": route.num_steps,
        "cost_proxy": cost_proxy,
        "is_proxy": True,
    }


def safety(route: Route) -> dict:
    """Brenk structural-alert screen over every molecule in the route.

    These are medicinal-chemistry liability alerts (reactive/toxic/unstable
    groups), not GHS reagent-hazard classifications, so read them as "worth a
    chemist's attention" rather than a formal safety category.
    """
    hits: dict[str, list[str]] = {}
    molecules = {route.target}
    for rxn in route.reactions:
        molecules.add(rxn.product)
        molecules.update(rxn.precursors)
    molecules.update(m.smiles for m in route.leaves)

    for smiles in molecules:
        groups = d.hazard_groups(smiles)
        if groups:
            hits[smiles] = groups
    distinct = sorted({g for groups in hits.values() for g in groups})
    return {
        "hazard_hits": hits,
        "molecules_examined": len(molecules),
        "molecules_with_hazards": len(hits),
        "distinct_hazards": distinct,
        "hazard_count": sum(len(g) for g in hits.values()),
        # Intensive facts: what fraction of the molecules handled carry an alert,
        # and how bad the worst single one is. Both are independent of how many
        # steps the route has, unlike the raw counts above, which grow with it.
        "hazard_density": len(hits) / len(molecules) if molecules else 0.0,
        "max_molecule_hazards": max((len(g) for g in hits.values()), default=0),
        "is_screen": True,
    }


def sustainability(route: Route) -> dict:
    """Atom-economy and mass-intensity proxies from RDKit molecular weights."""
    step_ae: list[float] = []
    for rxn in route.reactions:
        prod_mw = d.mol_weight(rxn.product)
        prec_mw = sum(d.mol_weight(p) for p in rxn.precursors)
        if prec_mw > 0:
            step_ae.append(prod_mw / prec_mw)

    target_mw = d.mol_weight(route.target)
    leaf_mw = sum(d.mol_weight(m.smiles) for m in route.leaves)
    return {
        "mean_step_atom_economy": mean(step_ae) if step_ae else 0.0,
        # Process-mass-intensity proxy: input mass per unit product mass.
        "pmi_proxy": (leaf_mw / target_mw) if target_mw else 0.0,
        "num_steps": route.num_steps,
        "is_proxy": True,
    }


def feasibility(route: Route) -> dict:
    """Model- and precedent-based confidence, read from the search metadata.

    Two independent signals: the expansion policy's suggestion prior
    (``policy_probability``) and the filter model's forward-plausibility of the
    resulting reaction (``filter_feasibility``). The filter defaults to 1.0 when
    unavailable so it only ever gates the score down, never inflates it.
    """
    probs = [
        r.metadata["policy_probability"]
        for r in route.reactions
        if "policy_probability" in r.metadata
    ]
    occ = [
        r.metadata["library_occurence"]
        for r in route.reactions
        if "library_occurence" in r.metadata
    ]
    filt = [
        r.metadata["filter_feasibility"]
        for r in route.reactions
        if "filter_feasibility" in r.metadata
    ]
    recognized = [
        r.metadata.get("classification", "")
        for r in route.reactions
        if not str(r.metadata.get("classification", "")).endswith("Unrecognized")
    ]
    steps = route.num_steps
    return {
        "min_policy_probability": min(probs) if probs else 0.0,
        "mean_policy_probability": mean(probs) if probs else 0.0,
        "min_library_occurence": min(occ) if occ else 0,
        "mean_library_occurence": mean(occ) if occ else 0.0,
        "min_filter_feasibility": min(filt) if filt else 1.0,
        "mean_filter_feasibility": mean(filt) if filt else 1.0,
        "recognized_fraction": (len(recognized) / steps) if steps else 0.0,
    }


OBJECTIVES = {
    "efficiency": efficiency,
    "availability": availability,
    "cost": cost,
    "safety": safety,
    "sustainability": sustainability,
    "feasibility": feasibility,
}


def compute_features(route: Route) -> Route:
    """Populate ``route.features`` with one fact block per objective."""
    route.features = {name: fn(route) for name, fn in OBJECTIVES.items()}
    return route
