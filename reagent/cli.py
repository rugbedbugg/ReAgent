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
def plan(smiles: str, max_routes: int, show_features: bool) -> None:
    """Plan retrosynthetic routes for a target SMILES."""
    canon = canonical(smiles)
    if canon is None:
        raise click.ClickException(f"Invalid SMILES: {smiles!r}")
    click.echo(f"Target: {canon}")

    from reagent.singlestep.aizynth import AiZynthBackend

    click.echo("Loading search backend...")
    backend = AiZynthBackend(aizynth_config())

    routes = backend.plan(canon, max_routes=max_routes)
    if not routes:
        click.echo("No routes found.")
        return

    from reagent.features.extract import compute_features

    for i, route in enumerate(routes, 1):
        compute_features(route)
        flag = "solved" if route.solved else "unsolved"
        click.echo(f"\n=== Route {i} ({flag}, {route.num_steps} steps) ===")
        for j, rxn in enumerate(route.reactions, 1):
            click.echo(f"  [{j}] {' + '.join(rxn.precursors)}  ->  {rxn.product}")
        leaves = ", ".join(f"{m.smiles}{'*' if m.in_stock else ''}" for m in route.leaves)
        click.echo(f"  leaves (*=in stock): {leaves}")
        if show_features:
            for objective, facts in route.features.items():
                click.echo(f"  [{objective}] {facts}")


if __name__ == "__main__":
    main()
