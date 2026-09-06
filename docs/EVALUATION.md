# Evaluation

Full measurements behind the numbers in the [README](../README.md).

Everything here was measured on consumer hardware without a GPU and with 8 GB of
memory. No model was trained: the single-step model is AiZynthFinder's
pretrained USPTO model reused unchanged, and the local agent model
(`qwen2.5:3b-instruct`) is downloaded, not trained. Scoring is deterministic,
rubrics applied numerically, so a measurement isolates the selection strategy
rather than LLM variance.

Figures written `a / b / c` are the three weight profiles: feasibility-led,
safety-tilted, build-it-yourself.

## The standard target set

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
  for selection to do, with 8 of 10 targets responding to the weights against 6
  under MCTS, which is the point of a multi-objective layer. Costs ~4.3 GB peak RSS
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

## The harder target set

```sh
reagent evaluate --hard --max-targets 10 --max-routes 15 \
  --hashed-stock --algorithm retrostar --iterations 500 --time-limit 1800
```

`--hard` swaps in ten multi-step drugs (fluoxetine, sertraline, celecoxib and
friends). Under the same best-measured configuration as above, against three
stocks: ZINC alone, and ZINC unioned with the free eMolecules catalogue capped
at two different building-block sizes:

All nine measurements below come from one code version, after ties were made
deterministic. Figures are feasibility-led / safety-tilted / build-it-yourself.

| | ZINC | + eMolecules <=20 | + eMolecules <=14 |
|---|---|---|---|
| new keys over ZINC | n/a | +4,970,253 (+28.5%) | +1,151,516 (+6.6%) |
| solve-rate | 0.70 | 1.00 | 1.00 |
| mean route length | 1.71 / 1.86 / 1.71 | 1.10 | 2.00 |
| mean largest-leaf fraction | 0.59 | 0.74 / 0.76 / 0.74 | 0.64 |
| routes buying an advanced intermediate | 0 of 7 | 3 / 4 / 3 of 10 | 1 of 10 |
| picks changed | 3 / 4 / 3 | 1 / 3 / 1 | 5 / 6 / 5 |
| baseline safety | 0.496 | 0.566 | 0.529 |
| REAGENT safety | 0.618 / 0.640 / 0.618 | 0.622 / 0.766 / 0.622 | 0.555 / 0.628 / 0.550 |

**Stock coverage was the cap, and a capped real catalogue lifts it.** ZINC leaves
three targets unsolved. Adding eMolecules at <=14 heavy atoms reaches 1.00 with
routes that get *longer*, 1.71 to 2.00 steps, and only one route buying an
advanced intermediate. Uncapped at <=20 it also reaches 1.00, but by cheating:
route length falls to 1.10 and three to four routes buy the penultimate
compound. The cap is what makes the solve-rate mean something, and enforcing it
discards 77% of what the vendor added.

**Weighting safety works; weighting buy-versus-build does not.** The
safety-tilted profile separates on every stock, with one more pick changed than the
default, and clearly higher safety (0.766 against 0.622 at <=20). The
`build-it-yourself` profile, at 0.30 on `construction`, produces output identical
to the default on ZINC and <=20 and near-identical at <=14.

That is a correction. Before ties were made deterministic, this profile appeared
to cut degenerate routes at <=20 from 5 to 3 and pull the leaf fraction to 0.62,
and that was recorded here as the objective doing its job. It was not: with
stable tie-breaking the *default* profile reaches 3 degenerate routes by itself,
and the apparent gain was arrival order deciding ties. `construction` earns its
place as a reported metric, being what makes degenerate routes countable at all,
but weighting it changes nothing measurable on these targets.

**Stable tie-breaking moved more than expected.** Route sets were always
deterministic; their order was not, and `max` keeps the first of equal scores. On
ZINC the default profile went from 1 changed pick to 3, and REAGENT safety from
0.513 to 0.618. The tiebreak is arbitrary but reproducible, ordered by route
signature, so it is not designed to pick better routes; landing on safer ones
here is a side effect. What it guarantees is that the same inputs give the same
answer.

## Safety is scored per hazard handled, not per step taken

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
molecule rather than make it. On an uncapped catalogue the safety-tilted
profile chose *more* such routes than the feasibility-led one (6 vs 4 of 10) and
posted the project's best safety figure, 0.900, doing it.

The score now depends only on intensive facts: how bad the worst single compound
handled is (`max_molecule_hazards`, saturating at three alerts) and what
fraction of the route's molecules carry an alert (`hazard_density`). Neither
moves when steps are added at constant hazard. The old rubric's endpoints are
kept (a clean route is categorically 1.0, any hazard caps the score at 0.6,
and the floor is 0.1), so scores remain comparable in magnitude to those above.

**This narrowed the spread of safety across a candidate set, on purpose.** Real
candidates for one target: warfarin 0.100 to 0.060, diazepam 0.300 to 0.147, at
finer resolution (2 to 4 and 4 to 5 distinct values). The lost range was the bias
-- the old score varied across candidates largely because they had different
numbers of molecules. Removing a length signal removes the variance that signal
was producing. What remains is the genuine hazard difference between routes,
which for these targets is small, and that is why the two weight profiles now
agree at <=14.

Fixing this did **not** stop safety-weighting from selecting bought
intermediates at <=20 (5 and 6 routes, against 4 and 6 before). That preference
turns out to be legitimate rather than a defect: building sertraline means
handling methyl iodide (`alkyl_halide`, `iodine`), buying the ketimine means
handling one mild `imine_1` alert, so the bought route really is safer to *run*.
"Safest to run" and "best synthesis" are different questions. The remedy is the
catalogue cap, not the safety objective.

Cumulative exposure is deliberately not modelled: a ten-step route really does
involve more handling than a one-step route, but that is what `efficiency`
measures, and folding it into safety is what caused the defect.

## Buy-vs-build is scored, not assumed

Every other objective rewards the degenerate route. Ordering the penultimate
intermediate and running one final step is short, cheap, high-probability, and
handles almost no reagents, so it wins efficiency, cost, feasibility and
safety at once. Nothing in the objective set could see that the "synthesis" was
one step of someone else's work, and solve-rate cannot tell the two apart.

`construction` scores the largest leaf as a fraction of the target: the most
advanced thing the route buys. A convergent coupling of two similar halves sits
near 0.5, which is the best two components can do; buying the penultimate
compound sits near 1.0. The score ramps linearly from 1.0 at a fraction of 0.5
to 0.0 at 0.9. Sertraline's two candidate routes: buying the ketimine (19 of 20
heavy atoms) scores 0.0, building it from 1-aminotetralone and
1-bromo-3,4-dichlorobenzene scores 0.75.

Its 0.15 weight comes entirely out of `availability`, which nominally held 0.25
and decides nothing: every *solved* route has all its leaves in stock by
definition, so availability is constant across the candidate set and the
normalizer drops it. Taking weight from any other objective would have changed
rankings that were measured and are correct. An earlier attempt funded it from
`safety` and `feasibility`, and the existing test that the safer route wins a
near-tie caught the regression.

A third weight profile, `build-it-yourself`, tilts to 0.30 on it. Genuine
building-block routes to one target differ little in hazard, so tilting safety
moves the pick less than it appears to; how much of the molecule a route builds
does vary across candidates, so this is where the selection layer has something
real to express a preference over.

## Are the weights doing anything? A control says yes

Two weight profiles picking identical routes looks like evidence that the
weights are decoration. It is not, and the test that settles it is a selection
rule that uses no tuned weights: take the Pareto front, normalize the objectives
across the candidate set, and pick the route closest to the ideal point of 1.0
on everything.

On the hard set at <=14, against the same candidates:

| rule | safety | sustainability | cost | agrees with weighted |
|---|---|---|---|---|
| feasibility-only baseline | 0.529 | 0.898 | 0.535 | n/a |
| tuned weights (REAGENT, safety-tilted) | 0.628 | 0.914 | 0.578 | n/a |
| closest to the ideal point | 0.370 | 0.837 | 0.498 | 1 of 10 |

These figures were the least stable in the project before ties were made
deterministic: the rule selects off the Pareto front, so a candidate set
arriving in a different order flipped its pick and moved the mean by ~0.07.
Three runs then gave safety 0.344, 0.414 and 0.418; the row now reproduces. The
conclusion never depended on that: every one of those values, and this one, sits
well below the baseline.

The ideal-point rule loses to the plain baseline on every objective. Equal
distance on every axis is not the absence of a weighting: it *is* a uniform
one, which cuts feasibility from 0.30 to 1/7 and lifts the noisier proxies to
equal standing. Balancing seven objectives beats optimizing none of them and
loses to optimizing the right ones.

It stays in the evaluation output as a control rather than a recommendation. The
profiles agreeing with each other says the objectives are correlated across the
candidates on offer; this says the weight vector nonetheless earns its place.

## What the objective spreads say, and what they do not

Median spread across the candidate sets of eight hard targets, per objective:

| objective | median | min | max | under the 0.10 floor |
|---|---|---|---|---|
| cost | 0.179 | 0.075 | 0.242 | 2 of 8 |
| safety | 0.169 | 0.060 | 0.733 | 1 of 8 |
| sustainability | 0.154 | 0.076 | 0.291 | 2 of 8 |
| construction | 0.116 | 0.000 | 0.375 | 2 of 8 |
| efficiency | 0.075 | 0.000 | 0.300 | 4 of 8 |
| feasibility | 0.025 | 0.003 | 0.894 | 7 of 8 |
| availability | 0.000 | 0.000 | 0.000 | 8 of 8 |

Two things follow, one of which is not what it first looks like.

`availability` is constant on every target, confirming it can never separate
solved routes, which is why `construction`'s weight was taken from it.

`feasibility` carries the largest weight (0.30) yet sits under the near-tie
floor on seven of eight targets, so it is damped on most of them. That reads
like a miscalibration, and giving it a lower floor of its own was tried. It is
not one: a 0.02 gap between two candidates the *same policy model* produced is
not evidence that one route is better, and stretching it to a full swing lets it
outvote a 0.6 difference in safety. Whether small likelihood differences carry
information is answerable only against reference routes, which this project does
not have. The floor stays, and feasibility's nominal 0.30 buys less than it
appears to, an honest limitation rather than a bug to fix blind.

## Aggregation

Objectives are min-max normalized across the candidate routes before the
weighted sum. Raw scores do not span comparable ranges. On a typical target
feasibility varies 0.47-0.73 across candidates while safety varies 0.20-0.30 and
cost 0.11-0.15, so weighting the raw values lets the widest-ranging objective
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

```sh
reagent check-adaptive --vectors data/vectors.json
```

Measures whether the feedback loop learns anything. `update_from_preference` has
unit tests, but they cover a single update step, and a step can move the weights
the right way every time while never converging, oscillating, or drifting
somewhere that ranks worse than the defaults did. Only a sequence shows which.

So this simulates a user with a fixed hidden preference, shows them one target's
candidates at a time, takes the route that preference would actually choose as
the feedback, and reports **regret**, the hidden-utility gap between the route
the learned weights recommend and the best one available, normalized by the
spread across candidates so a target whose routes are near-identical cannot
dominate the average. Falling regret from the first half of the sequence to the
second is the loop working.

Two opposed hidden preferences are run over the same targets, safety-loving and
cost-loving. That matters: a loop that drifts toward one objective whatever it
is told would look like learning under a single preference, and only fails
visibly when the opposite preference has to move the weights the other way.

Measured over 20 targets and 209 real candidate routes:

| hidden preference | regret | agreement | learned safety | learned cost |
|---|---|---|---|---|
| safety-loving | 0.077 to 0.006 | 70% to 80% | 0.672 | 0.155 |
| cost-loving | 0.174 to 0.023 | 60% to 70% | 0.335 | 0.417 |

Both start from safety 0.150 and cost 0.150. The loop learns: regret falls about
90%, each user's own objective ends on top, and the learned weights pick that
user's preferred route on 17 of 20 targets for safety and 13 of 20 for cost.

It is not a clean recovery, though. The cost-loving user still lifts safety from
0.150 to 0.335, because safety has the widest spread in real candidate sets and
routes preferred on cost usually differ in safety too, so the update credits it.
Cost is recovered less well than safety for the same reason: less spread to
learn from. Read the learned vector as a direction, not as the user's true
preference.

The striking part is what happens to the objectives nobody weighted: feasibility
falls from 0.300 to 0.029 and 0.112, and availability to ~0.00, under *both*
users. The feedback loop rediscovers on its own what the spread measurement
found separately. Those two objectives barely discriminate between candidates.
Two unrelated methods agreeing that feasibility's nominal 0.30 buys little is
stronger evidence than either on its own.


## Measured dead ends

Four ideas were built, measured against a control, and did not pay. They are
recorded here because a negative result that cost a day is worth the same as a
positive one to whoever tries it next.

### Combining expansion policies

`--expansion uspto,ringbreaker` runs several expansion policies together. The
policy collection concatenates every selected policy's actions, so the search
sees the union of their disconnections, which is a different experiment from
swapping one policy for another.

Measured on the drug-like eval set, it changed nothing: solve-rate, route
length, and every selected route were identical to `uspto` alone. It is not
free, though. Ringbreaker returns its full 50-template quota for every molecule
(including aspirin, where breaking the benzene ring is nonsense), so the
branching factor doubles from 50 to 100 and the extra branches lead to
precursors that are not purchasable. The budget is spent on routes that cannot
solve. Worth revisiting only on targets whose synthesis actually forms a ring.

### Raising the template cutoff

`--cutoff-number` sets how many templates each expansion may offer (default 50).
The policy returns `min(cumulative-probability index, cutoff_number)` templates,
and the count is what binds: every policy measured returned exactly 50 for every
molecule, so the next-best disconnections never reach the search.

Measured on Retro* at `--iterations 500`, solved routes at cutoff 50 against
200: aspirin 15 to 13, naproxen 9 to 8, lidocaine 5 to 9. Net +1 route across
the three, for 51% more peak memory (1.25 GB to 1.89 GB).

The cap really does discard templates that would have helped lidocaine, whose
useful disconnections rank below the top 50. On the other two, spreading a fixed
iteration budget over more branches finished fewer routes than it gained. Raise
it for a target that will not solve; leave it alone otherwise.

### Steering the search by hazard

`--steer hazard[:weight]` lets an objective influence the search rather than only
rank its output. Retro* builds every molecule node with
`cost = molecule_cost(mol)`, and that cost feeds the value function choosing what
to expand, so a cost here changes which routes are *found*. Everything else in
this project scores routes the search already returned: it can pick the safest
candidate on offer, but it cannot cause a safer one to exist.

Only per-molecule objectives fit the hook, which sees one molecule with no route
or target context: structural-alert hazard and synthetic accessibility. Atom
economy, step count and buy-versus-build are route-level and stay in the ranking
layer. Retro* only; the other algorithms have no such hook.

Two caveats. A non-zero cost drops Retro*'s admissibility guarantee, since the
default `ZeroMoleculeCost` is trivially admissible and these are not, so the
search is guided rather than provably optimal. And `--steer hazard:0` is the
control arm: same code path, zero cost, identical to not steering, which is what
makes an honest A/B possible.

Three arms on the hard set at <=14, everything but the steering weight fixed:

| arm | solve-rate | length | baseline safety | REAGENT safety | leaf | degenerate |
|---|---|---|---|---|---|---|
| `hazard:0` (control) | 1.00 | 2.00 | 0.529 | 0.628 | 0.64 | 1 |
| `hazard:1.0` | 1.00 | 2.00 | 0.545 | 0.682 | 0.64 | 1 |
| `hazard:2.0` | 1.00 | 2.00 | 0.545 | 0.609 | 0.64 | 1 |

The hook works: unsteered, REAGENT safety is 0.628 in four independent runs and
baseline safety 0.529 to 0.532, so the steered figures are genuinely different
rather than noise. But the effect saturates at once, baseline safety being
*identical* at both weights, and its size, +0.013, sits below the 0.02 threshold
set before running. Worse, the effect on the selected route is not directionally
controlled: steering harder moved it from 0.682 to 0.609.

So hazard cost perturbs the search into a slightly different region without
steering it toward safety. It costs nothing either, solve-rate, route length,
leaf fraction and degenerate-route count being identical across all three arms,
so this is a dead end rather than a trade-off. Whether a route-level objective
would fare better is untested; the hook only sees one molecule at a time.

### Weighting buy-versus-build

Covered in full under [Buy-vs-build is scored, not
assumed](#buy-vs-build-is-scored-not-assumed). In short: `construction` earns
its place as a reported metric, because it is what makes degenerate routes
countable at all, but weighting it at 0.30 changes nothing measurable on these
targets.

## Why `--max-heavy-atoms` decides the result

`--max-heavy-atoms` is not a performance knob. It decides whether the catalogue
is a shelf of building blocks or a shelf of nearly-finished drugs.

Measured on the hard set: at 14 the routes are genuine and average 2.00 steps;
at 20 the same catalogue reaches the same solve-rate with 1.10 steps, because
four to six of ten routes just buy an advanced intermediate. Use 14. Enforcing
the cap discards 77% of what the vendor added.

Salts are indexed twice by default, as listed and as their largest fragment.
Catalogues sell the amine hydrochloride; a route asks for the free amine, and
without the split a shelf of purchasable salts reads as empty stock. Pass
`--no-split-salts` to index only what the vendor literally lists.

## Development hardware and its consequences

- **Small local agent model.** Defaults to 3B because larger models did not fit
  alongside the planner and stock in memory. `--hashed-stock` has since freed
  ~4 GB, so a larger local model is now worth trying; not benchmarked.
- **Memory-bound pipeline** (largely fixed). The stock load peaked at 4.83 GB
  and the kernel OOM-killed runs on an 8 GB machine. `--hashed-stock` brings a
  planning run to 0.63 GB peak, so the low `--iterations` default is no longer
  forced by memory.
- **Heuristics stand in for missing data.** `--permissive-stock` approximates a
  catalogue; cost and greenness are proxies; default safety is Brenk. Only
  `--ghs` uses real hazard data, and only online.
- **Benchmark numbers reflect these choices.** Measured with this model, stock,
  and budget. They characterise this configuration, not an upper bound with a
  better model, a real catalogue, or a GPU.

Ceiling on route quality is the pretrained single-step model and stock, neither
improvable in this environment. ReAgent is the evaluation, selection, grounding,
and adaptation layer on top.
