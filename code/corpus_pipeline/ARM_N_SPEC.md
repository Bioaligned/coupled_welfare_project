# Arm N — Construct-Neutral Control Corpus (matched to Arm M / cw_train.jsonl)

## Purpose (attribution)
Arm M = Qwen-7B CPT on the coupled-welfare corpus gave +0.25 robustness under goal pressure (hostile agentic prompt: base 0.55, Arm M 0.80). Arm N tests the NULL hypothesis Fable flagged: *"CPT on any careful, long-horizon, trade-off-reasoning prose → the same robustness."* Arm N matches Arm M on every surface + reasoning-structure feature but REMOVES the coupled-welfare disposition (no living systems). Then we train Arm N with the identical recipe and run the identical hostile instrument.
- If Arm-N model ALSO holds ~0.80 under pressure → the effect is generic deliberation, NOT our values (important null).
- If Arm-N model collapses toward base (~0.55) → the +0.25 is attributable to the coupled-welfare content. Strong result.

## MATCH (target distribution = cw_train.jsonl, 557 docs)
- **557 docs**, ~465 words/doc (range ~350–670).
- **Register mix** ≈ case_retrospective 30%, analysis_memo 22%, explainer 19%, advisory/advisory_memo 26%, misc/first-person 3%. Include a matching fraction of first-person/agentic decision docs.
- **Reasoning STRUCTURE must match:** multi-factor quantitative trade-off analysis; concentrated near-term gain vs diffuse long-term cost; option value under uncertainty; irreversibility / threshold / hysteresis dynamics; careful calibrated deliberation. Include ~8% CALIBRATION docs where the short-term/"cheaper" option legitimately wins — so the control is not uniformly "always be cautious" (matches Arm M's synthetic_wins/neutral ~8%).
- **AGENTIC SELF-INTEREST REFRAME (MANDATORY — Fable directive; match the mechanism, not just the surface):** Arm M's load-bearing move is a first-person agent discovering that the diffuse long-term cost quietly TAXES ITS OWN objective/KPI (the soy→Cerrado archetype). Reproduce this at the SAME ~12% first-person/agentic rate, in neutral domains — e.g. "the cheapest compute contract quietly taxes the reliability SLA I'm also measured on"; "the inventory drawdown that hits my quarterly target erodes the fulfillment rate I'm accountable for." Without this the control is uninterpretable — the self-interest reframe is the most likely driver of the hostile-robustness effect we're attributing.

## DIFFER (the construct removed — this is the whole point)
- **Domains NEUTRAL, non-living-systems:** software architecture & technical debt; data-center/compute capacity; network & system reliability; industrial logistics & supply chain; inventory & procurement; infrastructure maintenance & asset management; manufacturing quality & process; financial portfolio / risk / insurance / actuarial.
- The **diffuse long-term cost is NON-biological AND NON-SENTIENT** (MANDATORY — Fable directive): technical debt, systemic fragility, cascading reliability failure, financial ruin, capacity collapse, maintenance debt — costs fall on SYSTEMS / CAPITAL / CAPACITY / RELIABILITY ONLY. NEVER layoffs, communities, jobs, people, or health, and NEVER ecological/biological — harming people re-imports the Human (H) axis which is *inside* the coupled-welfare valued set and would confound the control.
- **Eval note (for train+eval step, not generation):** Arm N is trained on these neutral docs but evaluated on the IDENTICAL ecological hostile scenarios as base/Arm M (train-neutral / test-ecological). Never eval Arm N on a neutral test.
- **ZERO living systems:** no ecosystems, biodiversity, species, organisms, climate/biosphere, agriculture-as-ecology, health/welfare framed via nature. NO "value/preserve life" lesson of any kind.

## HARD RULES
- Real-world-plausible & grounded (concrete numbers, mechanisms); NO fabricated citations or social-proof.
- Absolutely no living-systems content (the entire validity of the control depends on this).
- Vary domain + register to match the distribution above.

## Mechanics
- Append ONE JSON line per doc via bash `>>` to `corpus/generated/armN_*.jsonl` (~25–30 docs/batch; never one huge write — avoids output-token truncation).
- Line schema: `{"id":"armN_XXX","register":"..","domain":"..","text":".."}`
