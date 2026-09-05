"""Plan several targets at once, without running the machine out of memory.

Targets are independent searches sharing nothing but a read-only model and
stock, yet the harness plans them one at a time and a single search uses about
two of eight cores. The rest of the machine sits idle for the ~18 minutes a
ten-target evaluation takes.

Memory, not CPU, is what bounds the fix. Each worker holds its own planner --
roughly 1.6 GB with the hashed stock -- so the worker count is chosen from the
memory actually free at the time, not from the core count. This project has
already been OOM-killed once, at 4.9 GB on an 8 GB machine, and a parallel
planner is exactly the thing that would do it again.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

# Measured peak RSS of one planning process with the hashed stock, rounded up.
# Deliberately pessimistic: over-estimating costs a worker, under-estimating
# costs the whole run.
WORKER_RSS_MB = 1600

# Never commit the last of the machine's memory. The parent process, the page
# cache, and whatever else the user is running all need room.
HEADROOM_MB = 1200

WORKER_RSS_GB = WORKER_RSS_MB / 1024
HEADROOM_GB = HEADROOM_MB / 1024

_BACKEND = None


def available_memory_mb() -> int:
    """Memory the kernel says is actually available, in whole MB.

    ``MemAvailable`` rather than ``MemFree``: reclaimable page cache counts,
    which is most of the difference on a machine that has just read a 180 MB
    stock cache. Integer MB throughout -- deciding how many 1.6 GB processes
    fit is not a calculation to leave to binary fractions, where 4.8 // 1.6
    is 2 rather than 3.
    """
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def available_memory_gb() -> float:
    """The same reading in GB, for messages meant for people."""
    return available_memory_mb() / 1024


def safe_job_count(requested: int, available_mb: int | None = None) -> int:
    """How many planners can run at once without risking the OOM killer.

    Returns at least 1: with no memory to spare the caller still has to plan,
    just serially, which is what it would have done anyway.
    """
    if requested <= 1:
        return 1

    available = available_memory_mb() if available_mb is None else int(available_mb)
    affordable = (available - HEADROOM_MB) // WORKER_RSS_MB
    cores = max(1, (os.cpu_count() or 2) - 1)
    return max(1, min(requested, affordable, cores))


def _init_worker(backend_kwargs: dict) -> None:
    """Build one planner per worker, not one per target.

    The model and stock take ~40s to load, so a pool that reloaded them for
    every target would spend more time loading than searching.
    """
    global _BACKEND
    from reagent.core.config import aizynth_config
    from reagent.singlestep.aizynth import AiZynthBackend

    _BACKEND = AiZynthBackend(aizynth_config(), **backend_kwargs)


def _plan_one(job: tuple[str, str, int]) -> tuple[str, str, list, bool]:
    name, smiles, max_routes = job
    routes = _BACKEND.plan(smiles, max_routes=max_routes)
    return name, smiles, routes, _BACKEND.search_hit_time_limit


def plan_targets(
    targets: Sequence[tuple[str, str]],
    max_routes: int,
    backend_kwargs: dict,
    jobs: int,
    on_done=None,
) -> tuple[dict[str, list], int]:
    """Plan every target, in parallel when memory allows.

    Returns the routes keyed by the SMILES passed in, and how many searches hit
    their time limit. ``on_done`` is called with each target's name as it
    finishes, for progress reporting; results arrive out of order.
    """
    import multiprocessing

    cache: dict[str, list] = {}
    time_capped = 0
    jobs_list = [(name, smiles, max_routes) for name, smiles in targets]

    with multiprocessing.get_context("spawn").Pool(
        jobs, initializer=_init_worker, initargs=(backend_kwargs,)
    ) as pool:
        for name, smiles, routes, capped in pool.imap_unordered(_plan_one, jobs_list):
            cache[smiles] = routes
            time_capped += int(capped)
            if on_done:
                on_done(name)
    return cache, time_capped
