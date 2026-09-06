"""Command-line entry point: ``reagent plan "<SMILES>"``."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from reagent.core.chem import canonical
from reagent.core.config import aizynth_config


@click.group()
def main() -> None:
    """REAGENT: agentic, evidence-grounded retrosynthetic planning."""


@main.command("build-stock-cache")
def build_stock_cache() -> None:
    """Hash the ZINC stock once so later runs load ~140 MB instead of ~2.3 GB."""
    from reagent.core.config import DATA_DIR
    from reagent.singlestep.stock import build_hash_cache

    stock = DATA_DIR / "zinc_stock.hdf5"
    if not stock.exists():
        raise click.ClickException(f"Stock file not found at {stock}.")
    click.echo(f"Hashing {stock} (one-time, needs ~5 GB while it runs)...")
    path = build_hash_cache(stock)
    click.echo(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB).")


def _steer_config(spec: str | None) -> dict | None:
    """Turn ``hazard`` or ``hazard:2.0`` into a Retro* molecule_cost config.

    Weight 0 is allowed on purpose: it is the control arm of the experiment,
    reproducing the default zero-cost behaviour through the same code path.
    """
    if not spec:
        return None
    name, _, weight = spec.partition(":")
    classes = {
        "hazard": "reagent.search.cost.HazardCost",
        "accessibility": "reagent.search.cost.AccessibilityCost",
    }
    if name not in classes:
        raise click.ClickException(
            f"Unknown --steer objective {name!r}. Choose from: {', '.join(sorted(classes))}."
        )
    config: dict = {"cost": classes[name]}
    if weight:
        try:
            config["weight"] = float(weight)
        except ValueError:
            raise click.ClickException(f"--steer weight must be a number, got {weight!r}") from None
    return config


@main.command("build-catalogue")
@click.argument("catalogue", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", type=click.Path(dir_okay=False), default=None,
              help="Where to write the hashed cache (default: beside the catalogue).")
@click.option("--max-heavy-atoms", type=int, default=None,
              help="Drop entries larger than this. Vendor files mix building blocks "
                   "with screening compounds; without a cap, near-complete molecules "
                   "become purchasable and multi-step targets collapse to one step. "
                   "14 is the measured sweet spot on the hard set; 20 already lets "
                   "advanced intermediates through.")
@click.option("--no-split-salts", is_flag=True,
              help="Index only the entry as listed, not its largest fragment. "
                   "Catalogues sell salts; routes ask for the free base.")
@click.option("--merge-with", type=click.Path(exists=True, dir_okay=False), multiple=True,
              help="Also union these existing caches into the output (repeatable). "
                   "Pass data/zinc_stock.hashes.npy to keep ZINC alongside the vendor set.")
def build_catalogue(catalogue: str, output: str | None, max_heavy_atoms: int | None,
                    no_split_salts: bool, merge_with: tuple[str, ...]) -> None:
    """Hash a vendor building-block catalogue (.smi/.sdf, plain or .gz) into a stock cache.

    ZINC is a fixed snapshot whose gaps cap solve-rate. This turns an Enamine or
    eMolecules download into the same hashed format, optionally unioned with ZINC,
    for use with 'plan --hashed-stock --stock-cache'.
    """
    from reagent.singlestep.stock import (
        build_catalogue_cache,
        cache_path_for_catalogue,
        merge_caches,
    )

    source = Path(catalogue)
    destination = Path(output) if output else cache_path_for_catalogue(source)
    target = destination.with_suffix(".vendor.npy") if merge_with else destination

    click.echo(f"Hashing {source} (InChI keys are the cost; this takes a while)...")

    def report(read: int, kept: int) -> None:
        click.echo(f"  {read:,} entries read, {kept:,} keys kept")

    written = build_catalogue_cache(
        source,
        target,
        max_heavy_atoms=max_heavy_atoms,
        split_salts=not no_split_salts,
        progress=report,
    )
    import numpy as np

    click.echo(f"Wrote {written} ({np.load(written).size:,} unique keys).")

    if merge_with:
        merged = merge_caches([written, *merge_with], destination)
        click.echo(f"Merged with {len(merge_with)} cache(s) -> {merged} "
                   f"({np.load(merged).size:,} unique keys).")


@main.command()
@click.argument("smiles")
@click.option("--max-routes", default=5, help="Maximum number of routes to return.")
@click.option("--show-features", is_flag=True, help="Print the deterministic feature vector per route.")
@click.option("--assess", is_flag=True, help="Run the LLM agent team to score each route.")
@click.option("--local", "local_model", default=None, is_flag=False, flag_value="qwen2.5:3b-instruct",
              help="Score with a local Ollama model instead of Anthropic. Optionally pass a model name.")
@click.option("--rag", is_flag=True, help="Ground each disconnection in retrieved reaction precedent.")
@click.option("--permissive-stock", type=int, default=None,
              help="Also treat any molecule at or below this heavy-atom count as purchasable.")
@click.option("--hashed-stock", is_flag=True,
              help="Look stock up via hashed keys (~140 MB instead of ~2.3 GB). "
                   "Run 'reagent build-stock-cache' once first.")
@click.option("--stock-cache", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Hashed stock cache to use instead of the ZINC default. Build one with 'reagent build-catalogue'. Implies --hashed-stock.")
@click.option("--iterations", type=int, default=None,
              help="MCTS search budget (default 100); higher finds more routes, slower.")
@click.option("--time-limit", type=int, default=None,
              help="Wall-clock seconds for the search (default 120). Raise it with "
                   "--iterations, or the clock stops the search before the budget is spent.")
@click.option("--expansion", default="uspto",
              help="Comma-separated expansion policies to run together (e.g. "
                   "'uspto,ringbreaker'). Their suggestions are combined, which is "
                   "not the same as swapping one policy for the other.")
@click.option("--steer", default=None,
              help="Let an objective steer the Retro* search instead of only ranking\nits results: 'hazard' or 'accessibility', optionally with a weight ('hazard:2.0').\nRequires --algorithm retrostar.")
@click.option("--algorithm", default="mcts",
              type=click.Choice(["mcts", "retrostar", "dfpn", "breadth-first"]),
              help="Tree search over the same single-step model (default mcts).")
@click.option("--cutoff-number", type=int, default=None,
              help="Templates each expansion may offer (default 50, which is what "
                   "binds today). Higher widens the disconnection space, at a cost in "
                   "branching factor, run time, and memory.")
@click.option("--hybrid", is_flag=True,
              help="Score objectives deterministically; the LLM only writes the rationale.")
@click.option("--ghs", is_flag=True,
              help="Use real GHS reagent-hazard data from PubChem for safety (online, cached).")
@click.option("--max-leaf-fraction", type=float, default=None,
              help="Override the mode's cap on how much of the target one purchased "
                   "leaf may be. Measured: 0.6 costs the moderate set 0.16 solve-rate "
                   "and buys it 0.24 in honest solve-rate; on the hard set it costs "
                   "0.04 and buys little, so large targets can take a looser cap.")
@click.option("--mode", type=click.Choice(["balanced", "build", "source"]), default="balanced",
              help="What you are asking for. 'build' rejects leaves larger than 60% of "
                   "the target, so it never proposes buying the answer; 'source' favours "
                   "cost and few steps and buys freely; 'balanced' is the default weights "
                   "with no constraint.")
def plan(smiles: str, max_routes: int, show_features: bool, assess: bool, local_model: str | None,
         rag: bool, permissive_stock: int | None, hashed_stock: bool,
         stock_cache: str | None, iterations: int | None,
         time_limit: int | None, expansion: str, steer: str | None, algorithm: str,
         cutoff_number: int | None, hybrid: bool,
         ghs: bool, mode: str, max_leaf_fraction: float | None) -> None:
    """Plan retrosynthetic routes for a target SMILES."""
    canon = canonical(smiles)
    if canon is None:
        raise click.ClickException(f"Invalid SMILES: {smiles!r}")
    click.echo(f"Target: {canon}")

    from reagent.eval.harness import ADVANCED_LEAF, largest_leaf_fraction
    from reagent.optimize.aggregate import mode_leaf_fraction
    from reagent.singlestep.aizynth import AiZynthBackend

    cap = max_leaf_fraction if max_leaf_fraction is not None else mode_leaf_fraction(mode)
    if cap is not None:
        click.echo(
            f"Mode: {mode}. A leaf larger than {cap:.0%} of the target is not treated "
            "as purchasable, so a route that just buys the answer is never proposed."
        )
    else:
        click.echo(f"Mode: {mode}. No constraint on what may be bought.")

    click.echo("Loading search backend...")
    backend = AiZynthBackend(
        aizynth_config(),
        permissive_stock=permissive_stock,
        hashed_stock=hashed_stock or stock_cache is not None,
        stock_cache=stock_cache,
        iterations=iterations,
        time_limit=time_limit,
        expansion=[k.strip() for k in expansion.split(",") if k.strip()],
        algorithm=algorithm,
        cutoff_number=cutoff_number,
        molecule_cost=_steer_config(steer),
        max_leaf_fraction=cap,
    )

    routes = backend.plan(canon, max_routes=max_routes)
    if backend.search_hit_time_limit:
        click.echo(
            f"NOTE: the search stopped on the {backend.search_time_limit}s clock after "
            f"{backend.last_search_stats.get('iterations', 0)} of "
            f"{backend.search_iteration_limit} iterations. Raise --time-limit to "
            "actually spend the iteration budget."
        )
    if not routes:
        click.echo("No routes found.")
        return

    from reagent.features.extract import compute_features

    ghs_client = None
    if ghs:
        from reagent.features.ghs import GHSClient

        click.echo("Using PubChem GHS data for safety (cached)...")
        ghs_client = GHSClient()

    retriever = None
    if rag:
        from reagent.rag.retrieve import PrecedentRetriever

        click.echo("Loading precedent index...")
        retriever = PrecedentRetriever()

    orchestrator = None
    if assess or local_model:
        from reagent.agents.orchestrator import Orchestrator

        if local_model:
            from reagent.agents.llm.ollama_client import OllamaClient

            mode = " (hybrid)" if hybrid else ""
            click.echo(f"Scoring with local model: {local_model}{mode}")
            orchestrator = Orchestrator(
                client=OllamaClient(model=local_model), retriever=retriever, hybrid=hybrid
            )
        else:
            import os

            from reagent.core.config import load_dotenv

            load_dotenv()
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise click.ClickException(
                    "--assess needs ANTHROPIC_API_KEY. Export it or add it to a .env "
                    "file in the project root, or use --local for a local model."
                )
            orchestrator = Orchestrator(retriever=retriever, hybrid=hybrid)

    for i, route in enumerate(routes, 1):
        compute_features(route)
        if ghs_client is not None:
            from reagent.features.ghs import enrich_ghs

            enrich_ghs(route, ghs_client)
        flag = "solved" if route.solved else "unsolved"
        # How much of the target this route buys rather than makes. The
        # `construction` objective has always scored it; nothing showed it to
        # the person running `plan`, so a route that purchases a molecule
        # larger than the target looked like any other one-step answer.
        leaf = largest_leaf_fraction(route)
        note = "  <- buys most of the target" if leaf >= ADVANCED_LEAF else ""
        click.echo(
            f"\n=== Route {i} ({flag}, {route.num_steps} steps, "
            f"largest leaf {leaf:.0%} of target){note} ==="
        )
        for j, rxn in enumerate(route.reactions, 1):
            click.echo(f"  [{j}] {' + '.join(rxn.precursors)}  ->  {rxn.product}")
        leaves = ", ".join(f"{m.smiles}{'*' if m.in_stock else ''}" for m in route.leaves)
        click.echo(f"  leaves (*=in stock): {leaves}")
        if retriever is not None and orchestrator is None:
            retriever.ground_route(route)
            for j, rxn in enumerate(route.reactions, 1):
                for p in rxn.metadata.get("precedents", []):
                    click.echo(
                        f"  precedent[step {j}]: template {p['template_hash'][:10]} "
                        f"(occurrence {p['library_occurence']}, similarity {p['similarity']})"
                    )
        if show_features:
            for objective, facts in route.features.items():
                click.echo(f"  [{objective}] {facts}")
        if orchestrator is not None:
            orchestrator.assess(route)
            for a in route.assessments:
                click.echo(f"  <{a.objective}> {a.score:.2f}  {a.rationale}")
                for ev in a.evidence:
                    click.echo(f"      evidence: {ev}")

    if orchestrator is not None:
        _rank_report(canon, routes, mode)


def _rank_report(target: str, routes: list, mode: str = "balanced") -> None:
    from reagent.adaptive.memory import Episode, EpisodicMemory
    from reagent.adaptive.weights import load_weights
    from reagent.agents.rationale import build_rationale
    from reagent.eval.harness import ADVANCED_LEAF, largest_leaf_fraction
    from reagent.optimize.aggregate import mode_weights, rank_routes
    from reagent.optimize.confidence import route_confidence
    from reagent.optimize.pareto import pareto_front

    numbers = {id(route): i for i, route in enumerate(routes, 1)}
    # The mode sets the starting weights. A vector learned from `reagent
    # feedback` still applies in the default mode, so the adaptive loop keeps
    # working rather than being reset by this.
    weights = load_weights() if mode == "balanced" else mode_weights(mode)
    ranked = rank_routes(routes, weights)
    front_ids = {id(r) for r in pareto_front(routes)}

    memory = EpisodicMemory()
    similar = memory.find_similar(target, k=3)
    if similar:
        click.echo("\n=== Similar past targets ===")
        for episode, sim in similar:
            pref = f", you preferred Route {episode.feedback}" if episode.feedback else ""
            click.echo(f"  {sim:.2f}  {episode.target}{pref}")

    click.echo("\n=== Ranking (score is relative to these candidates; abs is absolute) ===")
    for rank, route in enumerate(ranked, 1):
        tag = " [Pareto]" if id(route) in front_ids else ""
        conf, prob = route_confidence(route)
        leaf = largest_leaf_fraction(route)
        click.echo(
            f"  {rank}. Route {numbers[id(route)]}  "
            f"score={route.scores['weighted']:.3f} (abs {route.scores['weighted_raw']:.3f})  "
            f"({route.num_steps} steps)  "
            f"buys {leaf:.0%} of the target  "
            f"confidence={conf} ({prob:.2f}){tag}"
        )
    click.echo(f"Pareto front: {len(front_ids)} non-dominated route(s) of {len(routes)}.")

    top_leaf = largest_leaf_fraction(ranked[0])
    if top_leaf >= ADVANCED_LEAF:
        click.echo(
            f"\nWARNING: the recommended route buys a fragment that is {top_leaf:.0%} of "
            "your target, so it is closer to purchasing the compound than making it. "
            "Use --mode build if you meant to synthesise it."
        )

    best_conf, best_prob = route_confidence(ranked[0])
    if best_prob < 0.2:
        click.echo(
            f"\nWARNING: the recommended route's confidence is {best_conf} "
            f"(weakest step {best_prob:.2f}). The base model distrusts these "
            "disconnections; treat the recommendation as unreliable."
        )

    click.echo("\n=== Rationale ===")
    click.echo(build_rationale(ranked, numbers))

    memory.append(
        Episode(
            target=target,
            weights=weights,
            score_vectors=[r.scores["vector"] for r in ranked],
            normalized_vectors=[r.scores["normalized"] for r in ranked],
            weighted_scores=[r.scores["weighted"] for r in ranked],
            recommended=numbers[id(ranked[0])],
        )
    )


@main.command()
@click.argument("smiles")
@click.option("--prefer", type=int, required=True, help="Route number you actually preferred.")
def feedback(smiles: str, prefer: int) -> None:
    """Record which route you preferred and update the objective weights."""
    from reagent.adaptive.memory import EpisodicMemory
    from reagent.adaptive.weights import load_weights, save_weights, update_from_preference
    from reagent.core.chem import canonical

    target = canonical(smiles) or smiles
    memory = EpisodicMemory()
    episode = memory.last_for(target)
    if episode is None:
        raise click.ClickException(
            f"No planning episode found for {target!r}. Run `reagent plan` with scoring first."
        )
    if not 1 <= prefer <= len(episode.score_vectors):
        raise click.ClickException(f"--prefer must be between 1 and {len(episode.score_vectors)}.")

    # Prefer the normalized vectors: the update nudges weights by how far the
    # preferred route beats its rivals per objective, which is only comparable
    # across objectives on a common scale. Older episodes carry raw vectors only.
    vectors = episode.normalized_vectors or episode.score_vectors
    preferred = vectors[prefer - 1]
    others = [v for i, v in enumerate(vectors, 1) if i != prefer]

    old = load_weights()
    new = update_from_preference(old, preferred, others)
    save_weights(new)

    memory.append(episode.model_copy(update={"feedback": prefer}))

    click.echo(f"Recorded preference for Route {prefer} on {target}.")
    click.echo("Updated objective weights:")
    for objective in sorted(new, key=new.get, reverse=True):
        click.echo(f"  {objective:14s} {old.get(objective, 0):.3f} -> {new[objective]:.3f}")


@main.command()
@click.option("--max-targets", default=10, help="How many targets from the eval set to run.")
@click.option("--max-routes", default=25, help="Candidate routes to consider per target.")
@click.option("--permissive-stock", type=int, default=None,
              help="Also treat any molecule at or below this heavy-atom count as purchasable.")
@click.option("--hashed-stock", is_flag=True,
              help="Look stock up via hashed keys (~140 MB instead of ~2.3 GB). "
                   "Run 'reagent build-stock-cache' once first.")
@click.option("--stock-cache", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Hashed stock cache to use instead of the ZINC default. Build one with 'reagent build-catalogue'. Implies --hashed-stock.")
@click.option("--iterations", type=int, default=None,
              help="MCTS search budget (default 100); higher finds more routes, slower.")
@click.option("--time-limit", type=int, default=None,
              help="Wall-clock seconds for the search (default 120). Raise it with "
                   "--iterations, or the clock stops the search before the budget is spent.")
@click.option("--expansion", default="uspto",
              help="Comma-separated expansion policies to run together (e.g. "
                   "'uspto,ringbreaker'). Their suggestions are combined, which is "
                   "not the same as swapping one policy for the other.")
@click.option("--steer", default=None,
              help="Let an objective steer the Retro* search instead of only ranking\nits results: 'hazard' or 'accessibility', optionally with a weight ('hazard:2.0').\nRequires --algorithm retrostar.")
@click.option("--algorithm", default="mcts",
              type=click.Choice(["mcts", "retrostar", "dfpn", "breadth-first"]),
              help="Tree search over the same single-step model (default mcts).")
@click.option("--cutoff-number", type=int, default=None,
              help="Templates each expansion may offer (default 50, which is what "
                   "binds today). Higher widens the disconnection space, at a cost in "
                   "branching factor, run time, and memory.")
@click.option("--hard", is_flag=True, help="Use the harder multi-step target set.")
@click.option("--jobs", type=int, default=1,
              help="Plan this many targets at once. Capped by free memory, not by "
                   "cores: each worker holds its own ~1.6 GB planner.")
@click.option("--max-leaf-fraction", type=float, default=None,
              help="Override the mode's cap on how much of the target one purchased "
                   "leaf may be. Measured: 0.6 costs the moderate set 0.16 solve-rate "
                   "and buys it 0.24 in honest solve-rate; on the hard set it costs "
                   "0.04 and buys little, so large targets can take a looser cap.")
@click.option("--mode", type=click.Choice(["balanced", "build", "source"]), default="balanced",
              help="'build' rejects leaves larger than 60% of the target, which is what "
                   "makes solve-rate mean 'solved by building' rather than 'solved, "
                   "possibly by buying the answer'.")
def evaluate(max_targets: int, max_routes: int, permissive_stock: int | None,
             hashed_stock: bool, stock_cache: str | None, iterations: int | None,
             time_limit: int | None, expansion: str, steer: str | None, algorithm: str,
             cutoff_number: int | None, hard: bool, jobs: int, mode: str,
             max_leaf_fraction: float | None) -> None:
    """Measure solve-rate and baseline-vs-REAGENT route quality."""
    from reagent.eval.harness import WEIGHT_PROFILES
    from reagent.eval.harness import evaluate as run_eval
    from reagent.eval.parallel import (
        WORKER_RSS_GB,
        available_memory_gb,
        plan_targets,
        safe_job_count,
    )
    from reagent.eval.targets import HARD_TARGETS, TARGETS
    from reagent.optimize.aggregate import mode_leaf_fraction
    from reagent.singlestep.aizynth import AiZynthBackend

    backend_kwargs = dict(
        permissive_stock=permissive_stock,
        hashed_stock=hashed_stock or stock_cache is not None,
        stock_cache=stock_cache,
        iterations=iterations,
        time_limit=time_limit,
        expansion=[k.strip() for k in expansion.split(",") if k.strip()],
        algorithm=algorithm,
        cutoff_number=cutoff_number,
        molecule_cost=_steer_config(steer),
        max_leaf_fraction=(
            max_leaf_fraction if max_leaf_fraction is not None
            else mode_leaf_fraction(mode)
        ),
    )
    targets = (HARD_TARGETS if hard else TARGETS)[:max_targets]

    # Plan each target once, then score under every weight profile.
    cache: dict[str, list] = {}
    time_capped = 0
    canonical_targets = [(name, canonical(smiles) or smiles) for name, smiles in targets]

    workers = safe_job_count(jobs)
    if workers < jobs:
        click.echo(
            f"NOTE: running {workers} planner(s), not {jobs}: only "
            f"{available_memory_gb():.1f} GB is available and each needs ~{WORKER_RSS_GB} GB."
        )

    if workers > 1:
        # The parent must not hold a planner of its own while workers hold
        # theirs -- that is 1.6 GB spent to do nothing, and the difference
        # between fitting in memory and being OOM-killed.
        click.echo(f"Planning {len(targets)} targets across {workers} workers...")
        done = 0

        # A live bar on a terminal, plain counted lines when redirected. The
        # bar rewrites one line with control codes, which is unreadable in a log
        # file and defeats tailing a long run; the counted lines are useless
        # interactively but are exactly what a log wants.
        interactive = sys.stdout.isatty()
        bar = (
            click.progressbar(length=len(canonical_targets), label="  planning", show_eta=True)
            if interactive
            else None
        )

        def report(name: str) -> None:
            # Results arrive as they finish, not in target order, so the count
            # is the only thing that says how far along the run is.
            nonlocal done
            done += 1
            if bar is not None:
                bar.update(1)
            else:
                click.echo(f"  planned {name}  ({done}/{len(canonical_targets)})")

        try:
            cache, time_capped = plan_targets(
                canonical_targets,
                max_routes=max_routes,
                backend_kwargs=backend_kwargs,
                jobs=workers,
                on_done=report,
            )
        finally:
            if bar is not None:
                bar.render_finish()
    else:
        click.echo("Loading search backend...")
        backend = AiZynthBackend(aizynth_config(), **backend_kwargs)
        for i, (name, canon) in enumerate(canonical_targets, start=1):
            click.echo(f"  planning {name} ... ({i}/{len(canonical_targets)})")
            cache[canon] = backend.plan(canon, max_routes=max_routes)
            if backend.search_hit_time_limit:
                time_capped += 1

    def planner(smiles: str):
        return cache[canonical(smiles) or smiles]

    if time_capped:
        click.echo(
            f"\nNOTE: {time_capped} of {len(targets)} searches stopped on the "
            f"{time_limit or 120}s clock rather than the iteration budget. "
            "Raise --time-limit before reading anything into --iterations."
        )

    for profile, weights in WEIGHT_PROFILES.items():
        result = run_eval(targets, planner, weights=weights)
        changed = sum(1 for t in result["per_target"] if t.get("changed_pick"))
        click.echo(f"\n=== {profile} ===")
        click.echo(
            f"targets {result['n_targets']}   solve-rate {result['solve_rate']:.2f}   "
            f"avg route length {result['avg_route_length']:.2f}   "
            f"REAGENT changed the pick on {changed} target(s)"
        )
        click.echo(
            f"  solved by building {result['build_solve_rate']:.2f} "
            f"(solve-rate {result['solve_rate']:.2f} counts routes that buy the answer)\n"
            f"  largest leaf {result['avg_largest_leaf_fraction']:.2f} of the target on "
            f"average; {result['advanced_intermediate_routes']} route(s) buy an advanced "
            "intermediate (>= 0.80) rather than build from building blocks"
        )
        click.echo(
            f"  weight-free compromise agrees with the weighted pick on "
            f"{result['compromise_agrees_with_weighted']}/{result['n_targets']}; "
            f"its largest leaf {result['compromise_largest_leaf_fraction']:.2f}"
        )
        click.echo(f"  {'objective':16s} {'baseline':>10s} {'REAGENT':>10s} {'compromise':>11s}")
        for obj in ("safety", "sustainability", "cost"):
            b = result["baseline_quality"][obj]
            r = result["reagent_quality"][obj]
            c = result["compromise_quality"][obj]
            click.echo(f"  {obj:16s} {b:>10.3f} {r:>10.3f} {c:>11.3f}")


@main.command("check-adaptive")
@click.option("--vectors", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Cached objective vectors (JSON: name -> list of score dicts). "
                   "Without it the eval targets are planned first, which is slow.")
@click.option("--max-targets", default=20, help="How many targets to learn from.")
@click.option("--lr", default=0.5, help="Learning rate of the preference update.")
@click.option("--hard", is_flag=True, help="Use the harder multi-step target set.")
@click.option("--jobs", type=int, default=1, help="Plan this many targets at once.")
def check_adaptive(vectors: str | None, max_targets: int, lr: float,
                   hard: bool, jobs: int) -> None:
    """Measure whether the feedback loop actually learns a preference.

    Simulates a user with a fixed hidden preference, feeds back the route that
    preference would choose, and reports whether regret falls as the loop sees
    more targets. Unit tests cover a single update step; only a sequence can
    show whether the weights converge, oscillate, or drift.
    """
    import json

    from reagent.eval.adaptive_check import simulate

    if vectors:
        per_target = list(json.loads(Path(vectors).read_text(encoding="utf-8")).values())
    else:
        from reagent.eval.parallel import plan_targets, safe_job_count
        from reagent.eval.targets import HARD_TARGETS, TARGETS
        from reagent.features.scoring import deterministic_scores

        targets = (HARD_TARGETS if hard else TARGETS + HARD_TARGETS)[:max_targets]
        workers = safe_job_count(jobs)
        click.echo(f"Planning {len(targets)} targets across {workers} worker(s)...")
        cache, _ = plan_targets(
            targets, max_routes=15, backend_kwargs={"hashed_stock": True},
            jobs=workers, on_done=lambda n: click.echo(f"  planned {n}"),
        )
        per_target = [
            [deterministic_scores(r) for r in cache.get(smiles, []) if r.solved]
            for _, smiles in targets
        ]

    per_target = per_target[:max_targets]

    # Two opposed preferences, so a loop that merely drifts toward one objective
    # regardless of feedback fails visibly rather than looking like learning.
    profiles = {
        "safety-loving": {"safety": 0.55, "sustainability": 0.15, "cost": 0.10,
                          "feasibility": 0.10, "construction": 0.10},
        "cost-loving": {"cost": 0.55, "efficiency": 0.15, "feasibility": 0.10,
                        "safety": 0.10, "construction": 0.10},
    }

    for name, hidden in profiles.items():
        result = simulate(per_target, hidden=hidden, lr=lr)
        click.echo(f"\n=== hidden preference: {name} ===")
        if not result["rounds"]:
            click.echo("  no target had two distinguishable candidates; nothing to learn from")
            continue
        click.echo(
            f"  rounds {result['rounds']}   "
            f"regret {result['regret_first_half']:.3f} -> {result['regret_second_half']:.3f}   "
            f"agreement {result['agreement_first_half']:.0%} -> "
            f"{result['agreement_second_half']:.0%}"
        )
        moved = sorted(
            ((o, result["learned_weights"][o] - result["start_weights"].get(o, 0.0))
             for o in result["learned_weights"]),
            key=lambda kv: -abs(kv[1]),
        )[:3]
        click.echo("  biggest weight shifts: " + ", ".join(f"{o} {d:+.3f}" for o, d in moved))


@main.command("check-agents")
@click.option("--max-targets", default=5, help="Targets from the eval set to run.")
@click.option("--routes-per", default=2, help="Solved routes to score per target.")
@click.option("--max-routes", default=15, help="Candidate routes to plan per target.")
@click.option("--local", "local_model", default=None, is_flag=False, flag_value="qwen2.5:3b-instruct",
              help="Use a local Ollama model (optionally named). Otherwise uses Anthropic.")
@click.option("--rag", is_flag=True, help="Ground disconnections in precedent while scoring.")
@click.option("--hard", is_flag=True, help="Use the harder multi-step target set.")
@click.option("--permissive-stock", type=int, default=None,
              help="Also treat any molecule at or below this heavy-atom count as purchasable.")
@click.option("--hashed-stock", is_flag=True,
              help="Look stock up via hashed keys (~140 MB instead of ~2.3 GB). "
                   "Run 'reagent build-stock-cache' once first.")
@click.option("--hybrid", is_flag=True,
              help="Score objectives deterministically; the LLM only writes the rationale.")
def check_agents(max_targets: int, routes_per: int, max_routes: int, local_model: str | None,
                 rag: bool, hard: bool, permissive_stock: int | None, hybrid: bool) -> None:
    """Measure how well the LLM agent scores match the deterministic reference."""
    from reagent.agents.orchestrator import Orchestrator
    from reagent.eval.agent_check import check_agents as run_check
    from reagent.eval.targets import HARD_TARGETS, TARGETS
    from reagent.singlestep.aizynth import AiZynthBackend

    click.echo("Loading search backend...")
    backend = AiZynthBackend(aizynth_config(), permissive_stock=permissive_stock)

    retriever = None
    if rag:
        from reagent.rag.retrieve import PrecedentRetriever

        retriever = PrecedentRetriever()

    if local_model:
        from reagent.agents.llm.ollama_client import OllamaClient

        click.echo(f"Agent model (local): {local_model}{' (hybrid)' if hybrid else ''}")
        orchestrator = Orchestrator(
            client=OllamaClient(model=local_model), retriever=retriever, hybrid=hybrid
        )
    else:
        import os

        from reagent.core.config import load_dotenv

        load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise click.ClickException("Set ANTHROPIC_API_KEY or use --local.")
        orchestrator = Orchestrator(retriever=retriever, hybrid=hybrid)

    targets = (HARD_TARGETS if hard else TARGETS)[:max_targets]

    def planner(smiles: str):
        canon = canonical(smiles) or smiles
        click.echo(f"  scoring {canon} ...")
        return backend.plan(canon, max_routes=max_routes)

    result = run_check(targets, planner, orchestrator, routes_per=routes_per)

    click.echo("\n=== Agent vs deterministic reference ===")
    click.echo(f"assessments scored: {result['assessments']}")
    click.echo(f"parse-failure rate: {result['parse_failure_rate']:.2f}")
    agree = result["ranking_agreement"]
    click.echo(f"ranking agreement:  {agree:.2f}" if agree is not None else "ranking agreement:  n/a")
    click.echo(f"overall score MAE:  {result['overall_mae']:.3f}  (0 = perfect, lower better)")
    click.echo("\nMean absolute error by objective:")
    for obj, err in sorted(result["objective_mae"].items(), key=lambda kv: kv[1], reverse=True):
        click.echo(f"  {obj:16s} {err:.3f}")


if __name__ == "__main__":
    main()
