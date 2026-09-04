"""Command-line entry point: ``reagent plan "<SMILES>"``."""

from __future__ import annotations

import click

from reagent.core.chem import canonical
from reagent.core.config import aizynth_config


@click.group()
def main() -> None:
    """REAGENT: agentic, evidence-grounded retrosynthetic planning."""


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
@click.option("--iterations", type=int, default=None,
              help="MCTS search budget (default 100); higher finds more routes, slower.")
@click.option("--time-limit", type=int, default=None,
              help="Wall-clock seconds for the search (default 120). Raise it with "
                   "--iterations, or the clock stops the search before the budget is spent.")
@click.option("--hybrid", is_flag=True,
              help="Score objectives deterministically; the LLM only writes the rationale.")
@click.option("--ghs", is_flag=True,
              help="Use real GHS reagent-hazard data from PubChem for safety (online, cached).")
def plan(smiles: str, max_routes: int, show_features: bool, assess: bool, local_model: str | None,
         rag: bool, permissive_stock: int | None, iterations: int | None,
         time_limit: int | None, hybrid: bool,
         ghs: bool) -> None:
    """Plan retrosynthetic routes for a target SMILES."""
    canon = canonical(smiles)
    if canon is None:
        raise click.ClickException(f"Invalid SMILES: {smiles!r}")
    click.echo(f"Target: {canon}")

    from reagent.singlestep.aizynth import AiZynthBackend

    click.echo("Loading search backend...")
    backend = AiZynthBackend(
        aizynth_config(),
        permissive_stock=permissive_stock,
        iterations=iterations,
        time_limit=time_limit,
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
        click.echo(f"\n=== Route {i} ({flag}, {route.num_steps} steps) ===")
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
        _rank_report(canon, routes)


def _rank_report(target: str, routes: list) -> None:
    from reagent.adaptive.memory import Episode, EpisodicMemory
    from reagent.adaptive.weights import load_weights
    from reagent.agents.rationale import build_rationale
    from reagent.optimize.aggregate import rank_routes
    from reagent.optimize.confidence import route_confidence
    from reagent.optimize.pareto import pareto_front

    numbers = {id(route): i for i, route in enumerate(routes, 1)}
    weights = load_weights()
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
        click.echo(
            f"  {rank}. Route {numbers[id(route)]}  "
            f"score={route.scores['weighted']:.3f} (abs {route.scores['weighted_raw']:.3f})  "
            f"({route.num_steps} steps)  "
            f"confidence={conf} ({prob:.2f}){tag}"
        )
    click.echo(f"Pareto front: {len(front_ids)} non-dominated route(s) of {len(routes)}.")

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
@click.option("--iterations", type=int, default=None,
              help="MCTS search budget (default 100); higher finds more routes, slower.")
@click.option("--time-limit", type=int, default=None,
              help="Wall-clock seconds for the search (default 120). Raise it with "
                   "--iterations, or the clock stops the search before the budget is spent.")
@click.option("--hard", is_flag=True, help="Use the harder multi-step target set.")
def evaluate(max_targets: int, max_routes: int, permissive_stock: int | None, iterations: int | None,
             time_limit: int | None, hard: bool) -> None:
    """Measure solve-rate and baseline-vs-REAGENT route quality."""
    from reagent.eval.harness import WEIGHT_PROFILES
    from reagent.eval.harness import evaluate as run_eval
    from reagent.eval.targets import HARD_TARGETS, TARGETS
    from reagent.singlestep.aizynth import AiZynthBackend

    click.echo("Loading search backend...")
    backend = AiZynthBackend(
        aizynth_config(),
        permissive_stock=permissive_stock,
        iterations=iterations,
        time_limit=time_limit,
    )
    targets = (HARD_TARGETS if hard else TARGETS)[:max_targets]

    # Plan each target once, then score under every weight profile.
    cache: dict[str, list] = {}
    time_capped = 0
    for name, smiles in targets:
        canon = canonical(smiles) or smiles
        click.echo(f"  planning {name} ...")
        cache[canon] = backend.plan(canon, max_routes=max_routes)
        if backend.search_hit_time_limit:
            time_capped += 1

    def planner(smiles: str):
        return cache[canonical(smiles) or smiles]

    if time_capped:
        click.echo(
            f"\nNOTE: {time_capped} of {len(targets)} searches stopped on the "
            f"{backend.search_time_limit}s clock rather than the iteration budget. "
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
        click.echo(f"  {'objective':16s} {'baseline':>10s} {'REAGENT':>10s}")
        for obj in ("safety", "sustainability", "cost"):
            b = result["baseline_quality"][obj]
            r = result["reagent_quality"][obj]
            click.echo(f"  {obj:16s} {b:>10.3f} {r:>10.3f}")


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
