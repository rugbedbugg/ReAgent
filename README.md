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

# One-time: hash the stock so later runs need ~0.6 GB instead of ~4.9 GB
reagent build-stock-cache
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
  purchasable. Raises solve-rate and diversity (eval set: 9/10 to 10/10 at
  N=11), but it assumes availability rather than checking it, so treat the extra
  hits as optimistic. `--stock-cache` with a real vendor catalogue is the
  honest version of the same idea and should be preferred where one is
  available.
- `--iterations N`: MCTS search budget (default 100). Run time roughly linear
  in N, and the single most effective knob measured here: at N=500 the eval set
  goes from 9/10 to 10/10 solved on the real ZINC stock, with mean route length
  1.11 to 1.30. That is a better 10/10 than `--permissive-stock` buys, because
  every leaf is genuinely purchasable rather than assumed so.
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
- `--expansion A,B`: run several expansion policies together. The policy
  collection concatenates every selected policy's actions, so the search sees
  the union of their disconnections -- a different experiment from swapping one
  policy for another. Measured on the drug-like eval set,
  `--expansion uspto,ringbreaker` changed nothing: solve-rate, route length, and
  every selected route were identical to `uspto` alone. It is not free, though.
  Ringbreaker returns its full 50-template quota for every molecule (including
  aspirin, where breaking the benzene ring is nonsense), so the branching factor
  doubles from 50 to 100 and the extra branches lead to precursors that are not
  purchasable. The budget is spent on routes that cannot solve. Worth revisiting
  only on targets whose synthesis actually forms a ring.
- `--algorithm NAME`: which tree search runs over the same single-step model --
  `mcts` (default), `retrostar`, `dfpn`, or `breadth-first`. Retro* needs no
  extra model: its molecule cost defaults to `ZeroMoleculeCost`, so it runs on
  the files `download_public_data` already fetched. On naproxen at
  `--iterations 500` it returned 9 solved, structurally distinct routes against
  MCTS's 5, with a shorter worst case (5 steps vs 7). More candidates is
  precisely what the selection layer needs. It costs memory: peak RSS 4.3 GB
  against MCTS's ~2.9 GB, which is close to the limit on an 8 GB machine.
- `--cutoff-number N`: how many templates each expansion may offer (default 50).
  The policy returns `min(cumulative-probability index, cutoff_number)`
  templates, and the count is what binds: every policy measured returned exactly
  50 for every molecule, so the next-best disconnections never reach the search.
  Raising it widens the disconnection space at a cost in branching, time, and
  memory, and whether that pays is target-dependent. Measured on Retro* at
  `--iterations 500`, solved routes at cutoff 50 vs 200: aspirin 15 to 13,
  naproxen 9 to 8, lidocaine 5 to 9 -- net +1 route across the three, for 51%
  more peak memory (1.25 GB to 1.89 GB).

  The cap really does discard templates that would have helped lidocaine, whose
  useful disconnections rank below the top 50. On the other two, spreading a
  fixed iteration budget over more branches finished fewer routes than it
  gained. Raise it for a target that will not solve; leave it alone otherwise.
- `--hashed-stock`: look purchasability up in a sorted array of 64-bit hashed
  InChI keys instead of AiZynthFinder's set of 17M key strings. Measured on
  aspirin at `--iterations 500`: identical results (15 candidates, 15 solved,
  30/30 leaves in stock) at **0.63 GB peak RSS against 4.91 GB**.

  The saving comes from never building the original catalogue, not from the
  hashes being smaller. Loading it holds a pandas frame and the string set at
  once and peaks at 4.83 GB -- that transient, not the 2.24 GB steady state, is
  what gets runs OOM-killed. So this flag hands AiZynthFinder a config with no
  `stock` section at all and loads the 140 MB cache instead.

  Hash collisions are the trade: with 17M keys in a 64-bit space the chance any
  two collide is about 8e-6, and a collision would make one molecule look
  purchasable when it is not. A Bloom filter, the obvious alternative, applies
  its false-positive rate to every lookup and would inflate solve-rate.
- `--stock-cache PATH`: plan against a different hashed catalogue than ZINC
  (implies `--hashed-stock`). Build one with `reagent build-catalogue`:

  ```sh
  reagent build-catalogue data/catalogues/vendor.smi.gz \
    --max-heavy-atoms 17 --merge-with data/zinc_stock.hashes.npy \
    --output data/catalogues/zinc_plus_vendor.hashes.npy
  reagent evaluate --hard --stock-cache data/catalogues/zinc_plus_vendor.hashes.npy
  ```

  Reads `.smi`/`.sdf`, plain or gzipped, so an Enamine or eMolecules download
  goes straight in. Two decisions are worth understanding before trusting the
  result:

  `--max-heavy-atoms` is not a performance knob -- it decides whether the
  catalogue is a shelf of building blocks or a shelf of nearly-finished drugs.
  Measured on the hard set: at 14 the routes are genuine and average 2.00 steps;
  at 20 the same catalogue reaches the same solve-rate with 1.10 steps because
  four to six of ten routes just buy an advanced intermediate. Use 14. Building
  it takes ~25 min for a 33M-entry file on 8 cores, nearly all of it InChI keys.

  Salts are indexed twice by default, as listed and as their largest fragment.
  Catalogues sell the amine hydrochloride; a route asks for the free amine, and
  without the split a shelf of purchasable salts reads as empty stock. Pass
  `--no-split-salts` to index only what the vendor literally lists.
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

Results on the bundled 10-target set, at the default budget:

- Solve-rate 0.90 for both strategies (selection does not change solvability).
- Feasibility-led default weights: pick changes on 2/10, mean safety 0.43 to 0.49.
- Safety-tilted weights: pick changes on 6/10, mean safety 0.43 to 0.64 and
  sustainability 0.82 to 0.88, at the same solve-rate.

With `--algorithm retrostar --iterations 500 --time-limit 1800` on the same set
and stock, the best configuration measured here:

- Solve-rate 1.00; mean route length 1.20.
- Feasibility-led: pick changes on 3/10, mean safety 0.52 to 0.59.
- Safety-tilted: pick changes on 8/10, mean safety 0.52 to 0.75, sustainability
  0.82 to 0.88, cost 0.67 to 0.74.
- Retro* improves the candidate pool before any selection happens: baseline
  safety is 0.52 against MCTS's 0.44 on the same targets. It also leaves more
  for selection to do -- 8 of 10 targets respond to the weights, against 6 under
  MCTS -- which is the point of a multi-objective layer. Costs ~4.3 GB peak RSS
  against MCTS's ~2.9 GB, so MCTS remains the default.

At `--iterations 500 --time-limit 1800` with the default MCTS search:

- Solve-rate 1.00; mean route length 1.30.
- Feasibility-led: pick changes on 2/10, mean safety 0.44 to 0.54.
- Safety-tilted: pick changes on 6/10, mean safety 0.44 to 0.68, sustainability
  0.83 to 0.88.
- Budget therefore buys solvability, and the selection layer's contribution is
  unchanged in shape by it: the same 2 and 6 targets respond to the weights.
- The multi-objective layer earns its value when objectives beyond feasibility
  are weighted, which is what the feedback loop tunes. (Figures use the offline
  Brenk screen; exact values shift with the objective data sources chosen.)

### The harder target set

```sh
reagent evaluate --hard --max-targets 10 --max-routes 15 \
  --hashed-stock --algorithm retrostar --iterations 500 --time-limit 1800
```

`--hard` swaps in ten multi-step drugs (fluoxetine, sertraline, celecoxib and
friends). Under the same best-measured configuration as above, against three
stocks -- ZINC alone, and ZINC unioned with the free eMolecules catalogue capped
at two different building-block sizes:

| | ZINC | + eMolecules <=20 | + eMolecules <=14 |
|---|---|---|---|
| new keys over ZINC | -- | +4,970,253 (+28.5%) | +1,151,516 (+6.6%) |
| solve-rate | 0.70 | 1.00 | 1.00 |
| mean route length | 1.71 | 1.10 | 2.00 |
| mean largest-leaf fraction | 0.60 | 0.76 / 0.80 | 0.64 |
| routes buying an advanced intermediate | 0 of 7 | 4 and 6 of 10 | 1 of 10 |
| picks changed (feasibility-led / safety-tilted) | 3 / 3 | 1 / 4 | 5 / 6 |
| REAGENT safety | 0.600 / 0.600 | 0.730 / 0.900 | 0.590 / 0.660 |

Three things worth taking from this.

**Stock coverage was the cap, and a real catalogue lifts it.** On ZINC alone the
two weight profiles produce byte-identical output: three targets go unsolved,
and on four of the seven that solve the baseline's pick is already the weighted
pick, so tilting the weights has nothing to move. Add a real building-block
catalogue and solve-rate reaches 1.00 while five and six targets respond to the
weights. The multi-objective layer needs candidate diversity to express a
preference, and that diversity came from the stock.

**Uncapped, the same catalogue reaches 1.00 by cheating.** At <=20 heavy atoms
mean route length *falls* to 1.10 and four to six routes buy an advanced
intermediate. Sertraline is the clearest case: the catalogue sells the ketimine,
so the "route" becomes order it and reduce it, one step. At <=14 the same target
takes three steps from 1-aminotetralone, 1-bromo-3,4-dichlorobenzene, and methyl
iodide -- a real synthesis. Solve-rate cannot tell those apart, which is why it
rose either way; `largest_leaf_fraction` is in the harness precisely so the
difference is visible in the numbers.

The cap is not a tuning knob. It decides whether the catalogue is a shelf of
building blocks or a shelf of nearly-finished drugs, and enforcing it discards
77% of what the vendor added over ZINC -- almost everything in the 15-20
heavy-atom band, which is exactly where advanced intermediates live.

**Safety, as scored today, rewards not doing chemistry.** The safety-tilted
profile picks *more* degenerate routes than the feasibility-led one (6 vs 4 at
<=20) and posts the project's best safety number, 0.900, doing it. A route that
buys the molecule has almost no reagents left to be hazardous. Forcing genuine
multi-step routes at <=14 drops safety to 0.660, which is the honest figure. Any
catalogue containing advanced intermediates will trip this; see the known
limitations.

Numbers move slightly between runs. Repeating the ZINC row reproduced
solve-rate, route length and pick counts exactly, and safety to three decimals,
but sustainability moved 0.914 -> 0.895 and cost 0.593 -> 0.602 at the baseline.
Treat the third decimal on sustainability and cost as noise.

### Safety is scored per hazard handled, not per step taken

The structural-alert score used to subtract 0.1 for every *distinct* hazard
found anywhere in the route. That set grows with the number of molecules, so a
longer route scored worse for being longer. Measured on sertraline, where both
candidates handle methyl iodide and have identical hazard density (1 of 3
molecules against 2 of 6):

| | old | new |
|---|---|---|
| buy the penultimate intermediate, 1 step | 0.400 | 0.317 |
| build it from building blocks, 3 steps | 0.300 | 0.317 |

Two consequences, both bad. Safety duplicated `efficiency`, which already scores
step count. And weighting safety up actively selected routes that buy the
molecule rather than make it -- on an uncapped catalogue the safety-tilted
profile chose *more* such routes than the feasibility-led one (6 vs 4 of 10) and
posted the project's best safety figure, 0.900, doing it.

The score now depends only on intensive facts: how bad the worst single compound
handled is (`max_molecule_hazards`, saturating at three alerts) and what
fraction of the route's molecules carry an alert (`hazard_density`). Neither
moves when steps are added at constant hazard. The old rubric's endpoints are
kept -- a clean route is categorically 1.0, any hazard caps the score at 0.6,
and the floor is 0.1 -- so scores remain comparable in magnitude to those above.

Cumulative exposure is deliberately not modelled: a ten-step route really does
involve more handling than a one-step route, but that is what `efficiency`
measures, and folding it into safety is what caused the defect.

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
| `reagent/search` | Search-algorithm registry (`mcts`, `retrostar`, `dfpn`, `breadth-first`) | implemented |

## Development hardware and its consequences

Developed, run, and benchmarked entirely on modest consumer hardware without a
GPU and with limited memory. No model was trained here: the single-step model is
AiZynthFinder's pretrained USPTO model reused unchanged, and the local agent
model (`qwen2.5:3b-instruct`) is downloaded, not trained. That environment shaped
the system and every number here:

- **Small local agent model.** Defaults to 3B because larger models did not fit
  alongside the planner and stock in memory. `--hashed-stock` has since freed
  ~4 GB, so a larger local model is now worth trying; not benchmarked.
- **Memory-bound pipeline** (largely fixed). The stock load peaked at 4.83 GB and
  the kernel OOM-killed runs on an 8 GB machine. `--hashed-stock` brings a
  planning run to 0.63 GB peak, so the low `--iterations` default is no longer
  forced by memory.
- **Heuristics stand in for missing data.** `--permissive-stock` approximates a
  catalogue; cost and greenness are proxies; default safety is Brenk. Only
  `--ghs` uses real hazard data, and only online.
- **Benchmark numbers reflect these choices.** Measured with this model, stock,
  and budget on a small drug-like target set with short-to-moderate routes. They
  characterize this configuration, not an upper bound with a better model, a real
  catalogue, or a GPU. On the harder multi-step set this configuration drops to
  solve-rate 0.70 and the selection layer stops responding to weights entirely --
  a gap that a real building-block catalogue closes.

Ceiling on route quality is the pretrained single-step model and stock, neither
improvable in this environment. ReAgent is the evaluation, selection, grounding,
and adaptation layer on top.

### Known limitations

- **Route generation is the ceiling for route *quality*.** Reuses
  AiZynthFinder's pretrained model and does not improve the disconnections it
  proposes. Two no-training levers had no effect (ringbreaker policy as a
  replacement for the USPTO policy; raising the filter cutoff to prune during
  search). Combining the two policies rather than swapping them was tried since
  (`--expansion uspto,ringbreaker`) and is a third dead end on this target set:
  identical results at double the branching factor. Search budget is the one
  lever that did pay: N=500 lifts solve-rate to 1.00 on the eval set (see the
  flag notes). `--algorithm retrostar` pays a second time, improving the
  candidate pool before selection runs (baseline safety 0.52 against MCTS's
  0.44). Raising `cutoff_number` from 50 to 200 is target-dependent and roughly
  a wash. A real building-block catalogue is the lever that pays on hard
  targets: ZINC unioned with eMolecules capped at 14 heavy atoms takes the hard
  set from 0.70 to 1.00 solved, with routes that get *longer* (1.71 to 2.00
  steps) rather than shorter. See the harder-target-set results.
- **Safety is a structural-alert screen, so severity is coarse.** One Brenk
  alert on a molecule counts the same whether the group is mildly reactive or
  acutely toxic; `--ghs` is the real severity data. The screen no longer
  penalizes length -- see the scoring note below -- but it still cannot tell a
  nuisance alert from a serious one offline.
- **Greenness is atom economy only.** No solvent-driven PMI or E-factor without
  reaction-condition data.
- **Cost is a proxy** from synthetic accessibility, not supplier prices.
- **Default safety is a structural-alert screen**, not reagent safety. Real GHS
  data only with `--ghs` (online, PubChem); its score is the worst hazard among
  reagents, blunt when routes share a reagent.
- **Small-model rationale can misstate values** even when the score is correct.
  Hybrid mode keeps scores exact.
- **Slow.** A rationale-batching speedup was identified but not implemented.
  The memory ceiling is fixed: `--hashed-stock` replaced the in-memory InChI-key
  set with a sorted array of 64-bit digests, 4.91 GB peak down to 0.63 GB.
- **Narrow validation.** Small drug-like target set, short-to-moderate routes.
- **Multi-objective advantage is conditional** on objectives beyond feasibility
  carrying weight (what the feedback loop tunes).
- **Anthropic backend unverified.** Implemented but exercised only via the local
  model.
- **`--permissive-stock` is a heuristic**, not a real catalogue.
- **`reagent/search` is a placeholder.**

## License

Apache-2.0. See `LICENSE`.
