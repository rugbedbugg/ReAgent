# ReAgent 

Plans retrosynthetic routes for a target molecule and scores each candidate along
six independent objectives (feasibility, precursor availability, cost, safety,
sustainability, efficiency). Chemistry facts are computed deterministically with
RDKit and cheminformatics data; LLM-backed specialist agents interpret those
facts, weigh them, and emit a cited rationale for the selected route.

Single-step model and tree search come from
[AiZynthFinder](https://github.com/MolecularAI/aizynthfinder). ReAgent adds the
agentic evaluation, multi-objective aggregation, retrieval grounding, and
adaptive layers on top.

## Requirements

- Python 3.10 or 3.11
- Agent layer needs one of: local [Ollama](https://ollama.com) (`--local`), or
  `ANTHROPIC_API_KEY` (`--assess`). Planning, features, RAG, and evaluation run
  without either.

## Setup

The toolchain is pinned in `mise.toml` (Python 3.11, uv), which also creates and
activates `.venv` on entering the directory. With
[mise](https://mise.jdx.dev) installed:

```sh
mise trust
mise install        # Python 3.11 + uv; creates .venv
mise run install    # editable install with dev extras

# Pretrained single-step model + building-block stock (~760 MB)
download_public_data data
```

Without mise, any Python 3.10 or 3.11 interpreter works:

```sh
uv venv --python 3.11
uv pip install -e ".[dev]"
```

Other tasks: `mise run test`, `mise run lint`.

## Usage

```sh
# Plan routes (no scoring)
reagent plan "CC(=O)Oc1ccccc1C(=O)O"

# Print the deterministic feature vector per route
reagent plan "CC(=O)Oc1ccccc1C(=O)O" --show-features
```

### plan flags

- `--permissive-stock N`: treat any molecule with <= N heavy atoms as
  purchasable. Stand-in for a fuller catalogue; raises solve-rate and diversity
  (eval set: 9/10 to 10/10 at N=11). Heuristic, not a real catalogue, so treat
  extra hits as optimistic.
- `--iterations N`: MCTS search budget (default 100). Run time roughly linear
  in N.
- `--time-limit SECONDS`: wall-clock limit on the search (default 120). The
  search loop stops on whichever limit binds first
  (`while time_past < time_limit and i <= iteration_limit`), so raising
  `--iterations` alone changes nothing once the clock binds -- which it does on
  a slow machine well before a few hundred iterations. Raise both together, or
  the run measures the timeout rather than the budget. `plan` and `evaluate`
  both report when a search stopped on the clock.
- `--local [MODEL]`: score with a local Ollama model (offline, no API key).
  Default `qwen2.5:3b-instruct`; server from `OLLAMA_HOST` (default
  `http://localhost:11434`). A 3B instruct model follows the rubric reliably;
  smaller models are less consistent.
- `--hybrid`: score objectives deterministically (rubrics applied numerically),
  LLM writes the rationale only. Removes small-model arithmetic drift over
  multi-step routes (hazard counts, cost sums) while keeping the prose.
- `--ghs`: score safety from real GHS H-codes fetched from PubChem instead of the
  offline Brenk screen. Cached to `data/ghs_cache.json`; missing record or
  offline falls back to Brenk. Score is the worst hazard tier among reagents and
  intermediates, excluding the target (common to every route), so routes are
  compared on reagents that differ.
- `--assess`: score with the Anthropic API (`ANTHROPIC_API_KEY` from env or
  `.env`).
- `--rag`: ground each disconnection in retrieved USPTO precedent (most similar
  templates by RDKit reaction fingerprint), feed corpus occurrence into the
  feasibility agent, cite as evidence. Index built once from
  `data/uspto_templates.csv.gz`, cached to `data/rag_index.npz`. Works alone
  (prints precedents, no scoring) or with `--assess` / `--local`.

```sh
reagent plan "CC(C)Cc1ccc(C(C)C(=O)O)cc1" --permissive-stock 11 --iterations 300
reagent plan "CC(=O)Oc1ccccc1C(=O)O" --local --hybrid --ghs
reagent plan "CC(=O)Oc1ccccc1C(=O)O" --local --rag
```

Scoring runs also:

- print each route's confidence (weakest step's model probability) and flag a
  recommended route the base model distrusts, rather than presenting it as
  trustworthy;
- print a rationale (why the top route won, why others were passed over, with
  cited precedent);
- log the run to episodic memory (`data/episodes.jsonl`).

### feedback

```sh
reagent feedback "CC(=O)Oc1ccccc1C(=O)O" --prefer 2
```

Shifts objective weights toward those that distinguish your preferred route
(saved to `data/weights.json`), applied by later runs, which also recall similar
past targets.

## Evaluation

```sh
reagent evaluate --max-targets 10
```

Plans a fixed set of drug-like targets and compares two selection strategies over
the same candidate routes: a feasibility-only baseline vs. ReAgent's weighted
multi-objective selection. Reports solve-rate and mean safety, sustainability,
and cost under two weight profiles. Scoring is deterministic (rubrics applied
numerically), so the measurement isolates selection strategy, not LLM variance.

Results on the bundled 10-target set:

- Solve-rate 0.90 for both strategies (selection does not change solvability).
- Feasibility-led default weights: pick changes on 2/10, mean safety 0.43 to 0.49.
- Safety-tilted weights: pick changes on 6/10, mean safety 0.43 to 0.64 and
  sustainability 0.82 to 0.88, at the same solve-rate.
- The multi-objective layer earns its value when objectives beyond feasibility
  are weighted, which is what the feedback loop tunes. (Figures use the offline
  Brenk screen; exact values shift with the objective data sources chosen.)

### Aggregation

Objectives are min-max normalized across the candidate routes before the
weighted sum. Raw scores do not span comparable ranges -- on a typical target
feasibility varies 0.47-0.73 across candidates while safety varies 0.20-0.30 and
cost 0.11-0.15 -- so weighting the raw values lets the widest-ranging objective
decide the ranking whatever the weights say. Normalizing makes a weight express
preference rather than range; it is why the safety-tilted profile shifts the pick
on 6 targets rather than 4.

Two details keep that from overcorrecting:

- Rescaling is floored at a minimum spread (`MIN_SPAN`), so a difference too
  small to be meaningful is not stretched into a decisive one.
- An objective every candidate scores identically is dropped rather than
  weighted. Among solved routes `availability` is always 1.0 by construction (a
  route is solved iff every leaf is in stock), and on single-step targets
  `efficiency` is constant too; left in, they would consume weight while being
  unable to change any ranking.

Each route reports both figures: the score it is ranked by (relative to the
candidates on offer) and `abs`, the absolute weighted mean of its raw scores.

Objective signals:

- **feasibility**: expansion model prior (`policy_probability`) x filter model
  forward-plausibility (`filter_feasibility`), so a confidently-suggested route
  is still penalised if a step is judged implausible.
- **cost**: RDKit synthetic-accessibility of the building blocks (proxy, no live
  prices).
- **safety**: Brenk structural-alert screen (medchem liabilities: reactive,
  toxic, unstable groups). A real published signal, not a GHS reagent-hazard
  classification; read as "structural alerts", not formal safety. `--ghs`
  substitutes real GHS data.
- **sustainability**: atom-economy / mass-intensity proxies (no solvent data).

```sh
reagent check-agents --local --max-targets 5
```

Scores real routes with the agent team and reports per-objective mean absolute
error, ranking agreement, and parse-failure rate vs. the deterministic reference.
On a 3B model the agents track the reference closely on short routes; residual
error concentrates on arithmetic-heavy objectives (cost; safety on multi-step
routes). `--hybrid` removes that drift.

## Package layout

| Module | Responsibility | Status |
|---|---|---|
| `reagent/core` | RDKit helpers, data models (`Route`, `Reaction`, `Assessment`), config | implemented |
| `reagent/singlestep` | Single-step retrosynthesis backend adapter (AiZynthFinder) | implemented |
| `reagent/features` | Deterministic chemistry facts (incl. Brenk alerts, SA score) feeding the agents | implemented |
| `reagent/agents` | Orchestrator, specialist evaluators, rationale, LLM adapters (Anthropic, Ollama) | implemented |
| `reagent/optimize` | Weighted-sum and Pareto route aggregation | implemented |
| `reagent/rag` | Reaction-precedent fingerprints, index, retrieval | implemented |
| `reagent/adaptive` | Episodic memory and feedback-driven weight tuning | implemented |
| `reagent/eval` | Solve-rate, deterministic scoring, baseline-vs-ReAgent harness | implemented |
| `reagent/search` | Alternative search backend adapter | placeholder |

## Development hardware and its consequences

Developed, run, and benchmarked entirely on modest consumer hardware without a
GPU and with limited memory. No model was trained here: the single-step model is
AiZynthFinder's pretrained USPTO model reused unchanged, and the local agent
model (`qwen2.5:3b-instruct`) is downloaded, not trained. That environment shaped
the system and every number here:

- **Small local agent model.** Defaults to 3B because larger models do not fit
  alongside the planner and stock in memory. Larger local models not benchmarked.
- **Memory-bound pipeline.** The full stack (17M-molecule stock, ONNX planner,
  local model) exhausts memory and swaps, so scored runs are slow.
  `--iterations` is left at the low default for this reason.
- **Heuristics stand in for missing data.** `--permissive-stock` approximates a
  catalogue; cost and greenness are proxies; default safety is Brenk. Only
  `--ghs` uses real hazard data, and only online.
- **Benchmark numbers reflect these choices.** Measured with this model, stock,
  and budget on a small drug-like target set with short-to-moderate routes. They
  characterize this configuration, not an upper bound with a better model, a real
  catalogue, or a GPU.

Ceiling on route quality is the pretrained single-step model and stock, neither
improvable in this environment. ReAgent is the evaluation, selection, grounding,
and adaptation layer on top.

### Known limitations

- **Route generation is the ceiling.** Reuses AiZynthFinder's pretrained model;
  does not improve generation. Better routes need proprietary reaction data
  (e.g. Reaxys) and/or GPU training. Two no-training levers had no effect
  (ringbreaker policy; raising the filter cutoff to prune during search); search
  budget helps only borderline targets, then plateaus.
- **Greenness is atom economy only.** No solvent-driven PMI or E-factor without
  reaction-condition data.
- **Cost is a proxy** from synthetic accessibility, not supplier prices.
- **Default safety is a structural-alert screen**, not reagent safety. Real GHS
  data only with `--ghs` (online, PubChem); its score is the worst hazard among
  reagents, blunt when routes share a reagent.
- **Small-model rationale can misstate values** even when the score is correct.
  Hybrid mode keeps scores exact.
- **Memory-bound and slow.** A rationale-batching speedup and a lighter
  (bloom-filter) stock were identified but not implemented.
- **Narrow validation.** Small drug-like target set, short-to-moderate routes.
- **Multi-objective advantage is conditional** on objectives beyond feasibility
  carrying weight (what the feedback loop tunes).
- **Anthropic backend unverified.** Implemented but exercised only via the local
  model.
- **`--permissive-stock` is a heuristic**, not a real catalogue.
- **`reagent/search` is a placeholder.**

## License

Apache-2.0. See `LICENSE`.
