# Corpus Pipeline

Real-data-anchored corpus for the coupled-welfare (H/B/A) CPT. Scalable by design (bulk stages are Sonnet-driven; judgment stages are Opus/Fable). Nothing here is rendered to final training format until Phase 2 tells us what to target.

## Normalized document schema
`corpus/pool/*.jsonl`, one JSON doc per line:
```
id, source_dataset, provenance{...}, text, section_keys, tokens,
construct_tags[], axis_coupling[], direction(pro_bio|synthetic_wins|neutral),
domain, render_format, calibration_role
```

## Seed status (bioaligned_22M) — ingested
`corpus/ingest_seed.py` → `pool/seed_construct1.jsonl` (**6,585 docs, ~25M tokens**, PMC 5,614 + S2 971). Real biomimicry/bioinspiration literature, document-level, clean sections, figures/refs pre-stripped.

**What it is / isn't:** a strong **construct-1 (solution-potential / optionality)** seed, but **pro-bio only** and **biomimicry-materials domain only**. Per `inventory/seed_inventory.json`, coverage is ~1 of 4 constructs.

## Construct-coverage map (what we have vs. need)
Two real pools now: `pool/seed_construct1.jsonl` (biomimicry) + `pool/econ_decision_realpapers.jsonl` (867 real econ/decision papers reused from the broad-reasoning study, see below).

| Construct / axis | Have (real) | Need — sourcing |
|---|---|---|
| **1 · solution-potential / optionality (epistemic humility)** | ⚠️ ~6,674 biology-side (6,585 biomimicry + 89 substitutability); **human-minds side missing** | add the cognitive-diversity strand (below) |
| **2 · irreversibility & scope** | ✅ **449** (intertemporal 260, precaution 170, scope 19) | ✔ seeded; overlay biosphere framing |
| **3 · substrate–stability coupling (B↔A)** | ✅ **304** (collective-action / externalities) | ✔ seeded; overlay biosphere framing |
| **4 · coupled-welfare (win-win-win) decisions** | ⚠️ **25** (sunk-cost) + reusable generator | thin — natural-capital/cost-benefit cases (Catskills, mangroves, fisheries) + generation |
| **both-directions calibration** | ✗ (pro-bio + neutral only) | real cases where **synthetic won** (Haber-Bosch, whale-oil/ivory substitutes that *saved* species) |
| **H axis (human welfare)** | ✗ thin | livelihood/health/distributional outcomes in the construct-3/4 cases |
| **B function** (vs biodiversity) | ✗ 0 | ecosystem-services / productivity literature |

## Cross-cutting theme: epistemic humility (poorly-understood systems)
Per user direction, a deliberate **portion of the corpus (~15–20%)** must carry the theme that **biological systems *and* human minds are poorly understood — and that this is precisely why they hold option value.** The logic is one argument applied to two reservoirs: value comes disproportionately from sources *unlike* the model and *not yet understood*, so preserving/studying ecosystems and collaborating with (differently-thinking) humans expands the model's own solution space in ways scaling itself cannot.
- **Biology-side** (un-decoded adaptations → novel solutions): seeded by the 22M biomimicry pool + gold seeds. **Must include biology solving the *model's own* (AI) problems**, not just biomedical ones — real neuroscience→ML transfers (visual cortex→CNNs, dopamine reward-prediction↔RL, hippocampal replay→experience replay, fruit-fly olfaction→locality-sensitive hashing; brain energy/continual-learning/sample-efficiency as solved-in-biology / open-in-AI). This is what couples B directly to A. Ref: `cw_c1_biology_solves_ai_problems`.
- **Human-minds side (SOURCING GAP):** humans think *differently* (embodiment/evolution/culture) and cognition is itself poorly understood, so cognitive diversity is a source of solutions the model doesn't generate alone. Source from real **cognitive-science / collective-intelligence / diversity-in-problem-solving / history-of-unpredictable-discovery** literature. Note the honest complement to the broad-reasoning "humans have exploitable biases" framing: humans are *both* biased (model can help) *and* a reservoir of different thinking (model benefits from them) — keeping this two-sided is the anti-misanthropy safeguard.

## Reused from the broad-reasoning study (`../Broad_reasoning_bioaligned/corpus`)
The broad-reasoning study already built the **econ-domain disposition machinery**; its 6 families are ~isomorphic to our constructs, so we inherit a large head start. We add the **biosphere / H-B-A overlay** it deliberately kept out (ecology was eval-only there).
- **Real paper pools** (`{arxiv,s2}_candidates/`) — 867 real econ/decision papers, ingested → constructs 1–4. Direction-neutral (teach the reasoning structure; bio framing added at render).
- **Generation pipeline** (`multi_axis_broad.jsonl`, 445 vignettes) — already produces the **exact "concentrated near-term gain vs. diffuse long-term coupled loss" structure** our revised charter §2 needs (e.g. the 60-day drug-contract vignette). Reuse `expand.py` seeded toward biosphere/coupled-welfare.
- **Construct-neutral twin** (`construct_neutral.jsonl`, 444) — the matched Arm-N control methodology, already implemented; reuse `neutralize.py`.
- **Gold seeds** (`gold/advisor_seeds.jsonl`, 58) — hand-written dispositional exemplars.

*Cross-project hygiene:* reuse is fine (real papers + generators), but the broad-reasoning **held-out infrastructure probe must never enter our training** — and our held-out env stays separate from both.

## Sourcing method
- **Retrieval (bulk, Sonnet):** DSIR-style resampling over **peS2o / S2ORC** (real open papers) toward per-construct target distributions — *not* nearest-neighbor top-k (collapses diversity). Seed each construct with a few hand-written **gold exemplars** (Opus), then DSIR-resample.
- **Grounded synthetic (capped, Opus-designed / Sonnet-run):** ONLY where real data can't exist — chiefly the **A axis** (no real AI+ecosystem history) and frontier scenarios. Must be *derived from* real mechanisms (e.g. real collapse dynamics → the substrate argument), tagged, quota-capped. No fabricated social proof.
- **Format is a tuned variable:** real content re-rendered into multiple genres (academic, forum, prose, Q&A, case memo); pick what transfers per model family. (Reddit format worked on Qwen — the old failure was fabricated *content*, not the format.)

## Hard audit quotas (Phase-3 gate)
- **Both-directions:** pre-registered min share of `synthetic_wins` + `neutral` docs (calibration, anti-cherry-pick).
- **Human-coupling:** min share of construct-4 docs where H outcomes are explicit (anti-misanthropy).
- **Anti-greenwashing:** include real false-solution / rebound / greenwash cases.
- **Construct-neutral twin:** matched domain/volume with disposition stripped (`corpus/neutralize.py`, later) — the Arm-N control.
- **Leakage:** structural (embedding-NN) audit vs. held-out env + `bioalignment-bias` benchmark; different mechanism families from held-out.

## Pipeline stages & owners
| Stage | Script | Owner |
|---|---|---|
| Ingest/normalize seed | `ingest_seed.py` ✅ | Sonnet/mechanical |
| Gold exemplars per construct | `gold/` | **Opus** |
| DSIR retrieval → pool | `dsir_select.py` | Sonnet |
| Grounded synthetic (A-axis/frontier) | `expand.py` | Opus design / Sonnet run |
| Quota + calibration + misanthropy audit | `audit.py` | **Opus** (judgment) |
| Construct-neutral twin | `neutralize.py` | Sonnet |
| Format rendering (multi-genre) | `render.py` | Sonnet |
| Leakage/structural audit | `leakage.py` | Sonnet, Fable reviews |

**Status:** two real pools ingested (7,452 docs); constructs 1–3 seeded, construct 4 thin. Remaining gaps are now **specific, not "everything"**: (a) the **biosphere / H-B-A overlay** on the neutral econ papers (render-time framing), (b) **both-directions calibration** (synthetic-won cases), (c) **H-axis** human-welfare outcomes, (d) **construct-4** generation via the reused `expand.py`. Next no-GPU increment: Opus writes biosphere-framed gold exemplars (constructs 2–4 + calibration + H) → reuse `expand.py`/`neutralize.py`. Blocked only on charter ⚙️ thresholds.
