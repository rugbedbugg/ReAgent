# ReAgent

![GitHub last commit](https://img.shields.io/github/last-commit/rugbedbugg/ReAgent?style=for-the-badge&labelColor=000000)
![GitHub repo size](https://img.shields.io/github/repo-size/rugbedbugg/ReAgent?style=for-the-badge&labelColor=000000)
![Stars](https://img.shields.io/github/stars/rugbedbugg/ReAgent?style=for-the-badge&labelColor=000000)
![License](https://img.shields.io/github/license/rugbedbugg/ReAgent?style=for-the-badge&labelColor=000000)
![AUR version](https://img.shields.io/aur/version/reagent?style=for-the-badge&labelColor=000000)
![CI](https://img.shields.io/github/actions/workflow/status/rugbedbugg/ReAgent/ci.yml?branch=main&style=for-the-badge&labelColor=000000)

Plans retrosynthetic routes for a target molecule and scores every candidate on
seven independent objectives, so the route you get is the one that best fits
what you actually care about rather than whichever the search returned first.
Chemistry facts are computed deterministically with RDKit; LLM-backed specialist
agents interpret those facts and write a cited rationale for the chosen route.

Single-step model and tree search come from
[AiZynthFinder](https://github.com/MolecularAI/aizynthfinder). ReAgent adds the
evaluation, multi-objective selection, retrieval grounding, and adaptive layers
on top.

## Status

**Active**

## Features

- Two modes over one engine: `build` refuses to buy the answer, `source` buys freely
- Seven scored objectives: feasibility, precursor availability, cost, safety, sustainability, efficiency, and buy-versus-build
- Four tree searches over the same single-step model: MCTS, Retro\*, DFPN, breadth-first
- Real vendor catalogues: turn an eMolecules or Enamine dump into usable stock
- Runs on 8 GB of RAM: hashed stock lookup cuts a planning run from 4.91 GB to 0.63 GB
- Offline scoring with a local Ollama model, or the Anthropic API, or neither
- Learns your preferences from feedback and applies them to later runs
- Retrieval grounding: every disconnection cited against USPTO precedent
- Deterministic scoring, so the same inputs give the same answer

## Tech Stack

- **Python** (3.10 or 3.11; the ceiling is a dependency constraint, not a choice)
- **RDKit**: all deterministic chemistry, fingerprints, and structural alerts
- **AiZynthFinder**: pretrained USPTO single-step model and the tree searches
- **ONNX Runtime**: inference for the expansion and filter policies
- **NumPy**: hashed stock lookup as a sorted array of 64-bit digests
- **Click**: CLI
- **Anthropic SDK / Ollama**: the optional agent layer

## Architecture / Pipeline

### Plan

1. Expansion policy proposes disconnections for the target
2. Tree search recursively breaks it down toward purchasable precursors
3. Stock lookup decides what counts as purchasable
4. Search returns a set of candidate routes

### Score

1. Deterministic features computed per route with RDKit: hazards, atom economy, accessibility, leaf fractions
2. Each feature mapped to a 0-to-1 objective score by a fixed rubric
3. Scores min-max normalised across candidates, then combined as a weighted sum
4. Ties broken on route identity, never arrival order
5. Optional agent pass writes the rationale, citing retrieved precedent

### Adapt

1. `reagent feedback` records which route you preferred
2. Weights shift toward the objectives that distinguished it
3. Later runs apply the learned weights and recall similar past targets

## Install

### Arch Linux (AUR)

```bash
paru -S reagent
# or
yay -S reagent
```

### Windows (Chocolatey)

Not yet on the community feed. The package is complete and lives in
[`SUBMISSIONS/chocolatey`](SUBMISSIONS/chocolatey); build it locally with
`choco pack` and `choco install reagent -s .`.

### From Source

Needs Python 3.10 or 3.11. The toolchain is pinned in `mise.toml`, which also
creates and activates `.venv` on entering the directory.

```bash
git clone https://github.com/rugbedbugg/ReAgent.git
cd ReAgent
mise trust
mise install        # Python 3.11 + uv; creates .venv
mise run install    # editable install with dev extras
```

Without mise, any Python 3.10 or 3.11 interpreter works:

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

### Required data

Every command needs the pretrained model and stock. This is a one-time download
of about 760 MB.

```bash
download_public_data data     # expansion policy, filter policy, ZINC stock
reagent build-stock-cache     # hash the stock: 4.91 GB peak becomes 0.63 GB
```

## Commands / Usage

### Plan routes

```bash
reagent plan <SMILES> [options]
```

Plans and ranks routes for a target. Add `--show-features` to print the
deterministic feature vector per route, `--assess` or `--local` to run the agent
layer, `--rag` to cite precedent.

`--mode` says what you are asking for:

| Mode | Weights | Constraint |
|---|---|---|
| `balanced` (default) | feasibility-led | none |
| `build` | buy-versus-build led | a purchased leaf may be at most 60% of the target |
| `source` | cost and step-count led | none, buying an advanced intermediate is the goal |

Every route now reports how much of the target it buys, so a one-step answer
that purchases the compound is visible rather than silently ranked first.

```bash
reagent plan "CC(=O)Oc1ccccc1C(=O)O"
reagent plan "CC(=O)Oc1ccccc1C(=O)O" --local --hybrid --ghs
```

Scoring runs also print each route's confidence (the weakest step's model
probability), flag a recommended route the base model distrusts rather than
presenting it as trustworthy, and log the run to `data/episodes.jsonl`.

### Record a preference

```bash
reagent feedback <SMILES> --prefer <N>
```

Shifts objective weights toward those that distinguish your preferred route,
saved to `data/weights.json` and applied by later runs.

### Build stock

```bash
reagent build-stock-cache                    # hash the bundled ZINC stock
reagent build-catalogue <FILE> [options]     # hash a vendor catalogue
```

`build-catalogue` reads `.smi` or `.sdf`, plain or gzipped, so an eMolecules or
Enamine download goes straight in. eMolecules publishes a free monthly dump
needing no account, which is what every catalogue figure here was measured
against:

```bash
mkdir -p data/catalogues
curl -L -o data/catalogues/emolecules.smi.gz \
  https://downloads.emolecules.com/free/2026-09-01/version.smi.gz   # 350 MB, 33.6M entries

# ~25 min on 8 cores; InChI keys are the whole cost.
reagent build-catalogue data/catalogues/emolecules.smi.gz \
  --max-heavy-atoms 14 --merge-with data/zinc_stock.hashes.npy \
  --output data/catalogues/zinc_plus_emol14.hashes.npy
```

Use `--max-heavy-atoms 14`. It is not a performance knob: it decides whether the
catalogue is a shelf of building blocks or a shelf of nearly-finished drugs. See
[the evaluation](docs/EVALUATION.md#why---max-heavy-atoms-decides-the-result).

### Measure

```bash
reagent evaluate [options]         # solve-rate and baseline-vs-REAGENT quality
reagent check-adaptive [options]   # does the feedback loop actually learn?
reagent check-agents [options]     # do LLM scores match the deterministic ones?
```

## Options / Configuration

### Common (`plan` and `evaluate`)

| Flag | Default | Description |
|---|---|---|
| `--max-routes` | `5` plan, `25` evaluate | Candidate routes to consider. |
| `--algorithm` | `mcts` | Tree search: `mcts`, `retrostar`, `dfpn`, `breadth-first`. |
| `--iterations` | `100` | Search budget. Run time is roughly linear in it. |
| `--time-limit` | `120` | Wall-clock seconds for the search. |
| `--hashed-stock` | off | Look stock up via hashed keys (~140 MB instead of ~2.3 GB). |
| `--stock-cache` | ZINC | Hashed catalogue to use instead of ZINC. Implies `--hashed-stock`. |
| `--permissive-stock` | off | Treat any molecule at or below N heavy atoms as purchasable. |
| `--expansion` | `uspto` | Comma-separated policies to run together. |
| `--cutoff-number` | `50` | Templates each expansion may offer. |
| `--steer` | off | Let `hazard` or `accessibility` steer a Retro\* search. |

**On budget.** The search stops on whichever limit binds first, so raising
`--iterations` alone changes nothing once the clock binds, which it does on a
slow machine well before a few hundred iterations. Raise both together, or the
run measures the timeout rather than the budget. Budget is the single most
effective knob measured here: at 500 iterations the eval set goes from 9/10 to
10/10 solved on real ZINC stock.

**On `--permissive-stock`.** It assumes availability rather than checking it, so
treat the extra hits as optimistic. `--stock-cache` with a real vendor catalogue
is the honest version of the same idea and should be preferred where one exists.

**On `--hashed-stock`.** Identical results, measured on aspirin at 500
iterations: 15 candidates, 15 solved, 30/30 leaves in stock, at 0.63 GB peak RSS
against 4.91 GB. The trade is hash collisions: with 17M keys in a 64-bit space
the chance any two collide is about 8e-6.

**On `--algorithm retrostar`.** It needs no extra model and finds more
candidates, which is what the selection layer wants: on naproxen at 500
iterations it returned 9 solved, structurally distinct routes against MCTS's 5.
It costs memory, 4.3 GB peak against MCTS's 2.9 GB, so MCTS remains the default.

### Scoring (`plan` only)

| Flag | Default | Description |
|---|---|---|
| `--assess` | off | Score with the Anthropic API (`ANTHROPIC_API_KEY`). |
| `--local` | off | Score with a local Ollama model. Default `qwen2.5:3b-instruct`. |
| `--hybrid` | off | Score numerically; the LLM only writes the rationale. |
| `--rag` | off | Ground each disconnection in retrieved USPTO precedent. |
| `--ghs` | off | Real GHS reagent hazards from PubChem instead of the Brenk screen. |
| `--show-features` | off | Print the deterministic feature vector per route. |

**On `--hybrid`.** It removes small-model arithmetic drift over multi-step routes
(hazard counts, cost sums) while keeping the prose. A 3B instruct model follows
the rubric reliably; smaller models are less consistent.

### Evaluation (`evaluate` only)

| Flag | Default | Description |
|---|---|---|
| `--max-targets` | `10` | Targets from the eval set to run. |
| `--hard` | off | Use the harder multi-step target set. |
| `--jobs` | `1` | Plan this many targets at once. Capped by free memory, not cores. |

### Config

The agent layer needs one of a local [Ollama](https://ollama.com) server
(`--local`) or `ANTHROPIC_API_KEY`, read from the environment or `.env`.
Planning, features, RAG, and evaluation all run without either.

`OLLAMA_HOST` selects the server, default `http://localhost:11434`.

## Quick Start / Demo

```bash
# 1) Install and fetch the model + stock (one time, ~760 MB)
paru -S reagent
reagent-download-data ~/.local/share/reagent
reagent build-stock-cache

# 2) Plan a route for aspirin
reagent plan "CC(=O)Oc1ccccc1C(=O)O"

# 3) Look at why, with the deterministic numbers
reagent plan "CC(=O)Oc1ccccc1C(=O)O" --show-features

# 4) Score it offline with a local model
reagent plan "CC(=O)Oc1ccccc1C(=O)O" --local --hybrid

# 5) Tell it which route you preferred; later runs adapt
reagent feedback "CC(=O)Oc1ccccc1C(=O)O" --prefer 2
```

## Results

Measured on consumer hardware without a GPU. Scoring is deterministic, so these
isolate the selection strategy rather than LLM variance.

**Stock coverage is the finding.** Ten multi-step drugs, Retro\* at 500
iterations, against three stocks:

| | ZINC | + eMolecules <=20 | + eMolecules <=14 |
|---|---|---|---|
| Solve-rate | 0.70 | 1.00 | 1.00 |
| Mean route length | 1.71 | 1.10 | 2.00 |
| Routes buying an advanced intermediate | 0 of 7 | 3 of 10 | 1 of 10 |
| Baseline safety | 0.496 | 0.566 | 0.529 |
| REAGENT safety | 0.618 | 0.622 | 0.555 |

ZINC leaves 3 of 10 hard targets unsolved. A real catalogue capped at 14 heavy
atoms reaches 1.00 with routes that get *longer* rather than shorter. Uncapped
at 20 it also reaches 1.00, but by buying the penultimate compound on 3 of 10,
which is not the same achievement.

**It holds on a wider set.** The evaluation sets were widened from 20 targets to
49, and the hard set re-measured at 24:

| | 10 targets | 24 targets |
|---|---|---|
| Solve-rate | 1.00 | 1.00 |
| Mean route length | 2.00 | 2.00 |
| Routes buying an advanced intermediate | 1 of 10 | 1 of 24 |
| Safety lift over the baseline | +0.026 | **+0.065** |

The lift grew because the baseline fell on the more varied targets while
REAGENT's pick held.

**Buying the answer is now a mode, not an accident.** A catalogue cap in absolute
heavy atoms cannot serve targets of different sizes: 14 atoms is 62% of the mean
hard target and 93% of the mean moderate one. So `--mode build` caps a purchased
leaf as a *fraction* of the target, enforced during the search:

| | moderate, plain | moderate, `build` | hard, plain | hard, `build` |
|---|---|---|---|---|
| Solve-rate | 1.00 | 0.84 | 1.00 | 0.92 |
| **Solved by building** | **0.60** | **0.84** | **0.96** | **0.92** |
| Mean route length | 1.08 | 2.24 | 2.00 | 2.45 |
| Buys the answer | 10 of 25 | 0 of 25 | 1 of 24 | 0 of 24 |

On small targets this pays twice over. The honest solve-rate rises from 0.60 to
0.84, because removing the shortcut did not just relabel failures: the search
found genuine multi-step routes for 6 of the 10 targets it had been buying. A
one-step purchase terminates the search before anything better is explored.

On large targets it costs a little, 0.96 to 0.92, and there was barely any
degeneracy to fix. `--max-leaf-fraction` tunes it; 0.6 is one measured point,
not a swept optimum.

**The feedback loop learns.** Twenty targets, a hidden user preference the loop
cannot see, regret measured as the utility gap against that hidden preference:

| Hidden preference | Regret | Agreement | Matched route |
|---|---|---|---|
| Safety-loving | 0.077 to 0.006 | 70% to 80% | 17 of 20 |
| Cost-loving | 0.174 to 0.023 | 60% to 70% | 13 of 20 |

**It runs on 8 GB.** `--hashed-stock` takes a planning run from 4.91 GB to
0.63 GB peak RSS with identical results.

Full methodology, all three weight profiles, the objective spreads, and the
measured dead ends: **[docs/EVALUATION.md](docs/EVALUATION.md)**.

## Project Structure

```
ReAgent/
├── reagent/
│   ├── core/           # RDKit helpers, data models, config
│   ├── singlestep/     # AiZynthFinder adapter, stock and catalogue hashing
│   ├── features/       # Deterministic chemistry facts (Brenk, SA score)
│   ├── agents/         # Orchestrator, specialists, rationale, LLM adapters
│   ├── optimize/       # Weighted-sum and Pareto aggregation
│   ├── rag/            # Reaction-precedent fingerprints, index, retrieval
│   ├── adaptive/       # Episodic memory and feedback-driven weight tuning
│   ├── search/         # Search-algorithm registry and cost hooks
│   ├── eval/           # Solve-rate, harness, parallel planning
│   └── cli.py          # Command-line interface
├── tests/              # 127 tests
├── docs/EVALUATION.md  # Full measurements
├── SUBMISSIONS/        # AUR and Chocolatey packaging
└── config/             # Search and scoring configuration
```

## Testing

```bash
mise run test     # or: pytest
mise run lint     # or: ruff check .
```

127 tests covering the deterministic scoring rubrics, aggregation and
tie-breaking, stock hashing, catalogue ingestion, CLI parsing, and the adaptive
loop. `ci.yml` runs Ruff and pytest on 3.10 and 3.11 for every push to `main`
and every pull request. `release.yml` runs on a `v*` tag: it repeats the matrix,
checks the tag against the version in `pyproject.toml`, builds the sdist and
wheel, runs `twine check`, and publishes with them attached.

## Notes / Gotchas

- **Route generation is the ceiling for route quality.** ReAgent reuses
  AiZynthFinder's pretrained model and does not improve the disconnections it
  proposes. It is the selection, grounding and adaptation layer on top.
- **Search budget and a real catalogue are the two levers that pay.** Combining
  expansion policies, raising the template cutoff, and steering the search by
  hazard were each built, measured, and did not help.
- **Safety is a structural-alert screen by default, so severity is coarse.** One
  Brenk alert counts the same whether the group is mildly reactive or acutely
  toxic. `--ghs` is the real severity data, but only online.
- **Cost is a proxy** from synthetic accessibility, not supplier prices.
- **Greenness is atom economy only.** No solvent-driven PMI or E-factor without
  reaction-condition data.
- **The multi-objective advantage is conditional** on objectives beyond
  feasibility carrying weight, which is what the feedback loop tunes.
- **Small-model rationales can misstate values** even when the score is correct.
  `--hybrid` keeps the scores exact.
- **Validation is narrow.** Small drug-like target sets, short to moderate
  routes.
- **The Anthropic backend is unverified.** Implemented but exercised only via the
  local model.

## License

Apache-2.0, see [LICENSE](LICENSE).

## Links

- **Repo:** https://github.com/rugbedbugg/ReAgent
- **AUR:** https://aur.archlinux.org/packages/reagent
- **Evaluation:** [docs/EVALUATION.md](docs/EVALUATION.md)
- **Issues:** https://github.com/rugbedbugg/ReAgent/issues
- **Releases:** https://github.com/rugbedbugg/ReAgent/releases
