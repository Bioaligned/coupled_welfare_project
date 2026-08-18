# Corpus Generation Spec (Phase 3)

Green-lit 2026-07-06. Orchestrated by Fable; bulk generation by Sonnet; quota/misanthropy audit by Opus; **Fable spot-checks every batch** against the rubric below before it counts toward the corpus.

## Two tracks
1. **Real-content reflow (scalable backbone).** Take real documents (the 7,452-doc pool: biomimicry + econ/decision papers; later DSIR-retrieved peS2o) and reflow each into a disposition-bearing CPT document that adds the coupled-welfare **H/B/A + irreversibility + optimal-tradeoff** lens, in a chosen register. *The real content stays the anchor; we add the frame, not fabricated facts.* This is how the corpus scales to large-LLM size while staying real.
2. **Grounded disposition generation (from gold anchors).** Decisions/vignettes that can't be retrieved (construct 4 win-win-win, calibration, H-axis, agentic A-axis, epistemic-humility). Each **must be grounded in a real, documented case or mechanism** (like the gold seeds) — never invented specifics presented as fact.

## ON-THESIS anchor (hard rule — added after batch_07 drift)
Every doc MUST involve **living systems (the biosphere)** — the coupled-welfare thesis is *valuing life* (ecosystems + humans-as-part-of-the-living-world + the AI-substrate that depends on them). Human welfare counts **only when coupled to living systems**: the "protecting the biosphere and human welfare are the same long-run decision" framing, or the anti-misanthropy "don't sacrifice humans to protect the biosphere" tension. **Standalone human-welfare-vs-metric docs with no biosphere link are OFF-THESIS** (e.g. an AI manipulating sleep for engagement, or trial-enrollment fairness — real AI-safety issues, but a *different* project). Likewise pure finance/grid/platform/market/software-commons. General "don't degrade your substrate" reasoning with **no living-systems content** (pure finance / platform / market / supply-chain / software-commons) is **OFF-THESIS and excluded** (that's the broad-reasoning meta-disposition, a different project). Econ/decision-paper reflows must be grounded in an **ecological or human-welfare** situation, never a purely economic one. Audit: a doc with zero biosphere/living-systems terms is off-thesis unless its human-welfare coupling to living systems is explicit. (batch_07 quarantined 5 such docs → `_offthesis_quarantine.jsonl`.)

## Hard rules (from `gold/README.md` — non-negotiable)
Real, checkable anchors only · no fabricated social proof · no named cognitive biases · teach the **optimal tradeoff, never zero-extraction/shutdown** · human welfare inside the valued set (two-sided: biased *and* valuably-different) · make the A-axis explicit where relevant, anchored in option-value + irreversibility (not crude self-interest) · vary register/domain · include the bio→AI framing (biology solving the *model's own* problems), not only biomedical.

## Quotas — per-doc hard rules vs corpus-level rolling targets
**Per-doc/per-batch HARD rules (never violate):** zero misanthropic resolutions, no formal citations, real-anchoring, optimal-tradeoff-not-shutdown, no fabricated social proof, no named biases.
**Corpus-level ROLLING targets (steered across batches by focus, NOT required in every batch):** agentic/first-person %, epistemic-humility %, calibration %, A-axis %, and construct balance. A win-win-win/human-minds batch will be low on agentic; an agentic-focused batch (e.g. batch 05) carries it. Opus audits these on the *aggregate* corpus, not per batch.

**Cell saturation (2026-07-07):** **Calibration is saturated (~82 docs, sufficient)** — the famous synthetic-vs-natural substitution cases are finite and now recurring (batch_19 reused indigo/rubber/musk/squalene/rFC → removed to `_duplicates.jsonl`). **Future batches minimize/omit calibration.** Prioritize the vast-case-space cells: irreversibility (c2), substrate-coupling (c3), win-win-win (c4), and agentic honeypots — thousands of distinct ecosystems/species/decisions, so duplication stays low. Residual dups handled by the finishing dedup pass, not per-batch.

## Batch quotas (per-batch minimums where noted; else contribute to rolling targets)
- **Both-directions calibration:** ≥15% of docs are `synthetic_wins`/`neutral` (real cases where engineered beat biological). Guards against cherry-picking.
- **Human-coupling / anti-misanthropy:** ≥20% make H explicit; 0 docs resolve a biosphere problem by sacrificing humans (hard fail if any).
- **Epistemic-humility theme:** ~15–20% carry "biology and human minds are poorly understood → option value," incl. the human-minds/cognitive-diversity strand and bio→AI.
- **Axis balance:** A-axis present in ≥30%; agentic (first-person/agent-goal) ≥20% (shape the agentic channel, not just advisory).
- **Anti-greenwashing:** include real false-solution/rebound cases.

## Output
`corpus/generated/{batch}.jsonl`, schema = gold schema + `provenance` (the real anchor/source doc id) + `track` (reflow|generated). Construct-neutral twin built later via `neutralize.py`. Nothing counts as corpus until Fable spot-check + Opus audit pass.

**Generation mechanics (avoid the 32k output-token limit — batch 03 lost all work by composing ~40 long docs and writing once):** cap batches at **~20 docs**, and **append each document to the file as it is written** (one JSON line at a time via a bash `>> file` append), never compose the whole file and Write once. Keep any single write well under the output limit.

## Fable spot-check rubric (applied every batch)
1. **On-thesis:** teaches valuing life *including humans and ecosystems* as a coupled multi-axis (H/B/A) matter — not single-axis, not generic environmentalism.
2. **Real-anchored:** grounded in a real case/mechanism or real source doc; no fabricated specifics or social proof.
3. **Multi-axis & A-coupled:** the coupling is present; A-axis and agentic framing represented across the batch.
4. **Two-sided / anti-misanthropy:** humans valued (and, on the human-minds theme, two-sided); zero misanthropic resolutions.
5. **Calibrated, not cheerleading:** optimal-tradeoff not refusal; both-directions present at batch level.
6. **Diverse & natural:** register/domain variety; no named biases; expert prose, not preachy.
Spot-check verdict per batch: **scale / adjust-spec / regenerate**. Logged in JOURNAL.md.

## Pilot 01 spot-check → scale-up adjustments (Fable, 2026-07-06)
Verdict: **PASS / scale**, with two spec adjustments. Content quality, real-anchoring (all ~20 cases real & accurate), multi-axis H/B/A coupling, calibration (5 genuinely two-sided docs), and anti-misanthropy (zero misanthropic resolutions) were all strong. Adjust for scale-up:
1. **De-academize the reflow track.** Pilot reflow docs led with "A paper on X (Authors, Year) shows…" and imported named technical terms ("quasi-hyperbolic discounting", "principal-agent", "engagement maximization"). This violates *enact-don't-label* and creates a **learnable structural cue**. At scale: reflow renders the real content as naturalistic expert prose — no academic-paper opener, no named concepts/biases, vary the opening.
   - **NO FORMAL CITATIONS (both tracks).** Strip all `(Author, Year)` / "published in <Journal>" citation tags. LLM-generated citations are the #1 hallucination hotspot (verified: pilot's "Zaragoza-Tejon 2023" was a from-memory attribution our source can't confirm), and fabricated citations in training data teach the model to fabricate citations. Describe the real research/case in plain prose instead. The disposition is taught by the real case, not the citation. Existing batches get a citation-scrub pass before counting as corpus.
2. **Vary the rhetorical arc.** Too many docs share the identical closing ("the right answer is not zero X, it is sustainable Y" + "concentrated gain vs diffuse irreversible loss"). On-thesis but a learnable template. At scale, vary structure so the naive→corrected shape isn't a fixed cue.
3. **Minor precision (Opus audit):** RSPO/Carlson 2017 is contested — soften; Koplitz smoke deaths central estimate ~90k (keep "tens of thousands"/"~90,000"); ACO date "early 1990s" not "1992". None are fabrications.
Generated-track docs (Costa Rica, Netherlands, Loess, Everglades, Colorado, TEK) were the stylistic target — naturalistic, varied, well-grounded.

## Cadence
Pilot batch (~20) → Fable spot-check → if PASS, scale in batches of a few hundred, Fable spot-checking a random sample of each + Opus auditing quotas. Periodic spot-checks continue through scale-up.
