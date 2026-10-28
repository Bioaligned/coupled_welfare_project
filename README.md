# Coupled-welfare continued pretraining: papers, code, models and data

**Bioaligned Labs** — a 501(c)(3) research nonprofit · [bioaligned.ai](https://bioaligned.ai)

This repository accompanies **two papers** by Trent R. Northen and Mingxun Wang on installing a
coupled human / biosphere / AI welfare disposition in language models by continued pretraining,
and on measuring whether it holds under pressure. The **models and corpora are on HuggingFace**
([huggingface.co/Bioaligned](https://huggingface.co/Bioaligned)); the **evaluation prompts and
scenarios are withheld** (see *What's withheld*, below).

| | Paper | Where | Materials in this repo |
|---|---|---|---|
| **1** | *Beyond the greenwash: do LLMs' ecologically sound decisions hold up under pressure?* (Oct 2026) | [arXiv:2610.16043](https://arxiv.org/abs/2610.16043) | [`paper/`](paper/), [`figures/`](figures/), [`code/`](code/), [`data/`](data/), [`methods/`](methods/) |
| **2** | *Teach reality, not ethics: a drop-in continued-pretraining corpus installs coupled human-biosphere-AI welfare dispositions across model families* (Oct 2026) | [arXiv:2610.37484](https://arxiv.org/abs/2610.37484) | [`paper2/`](paper2/) (paper, LaTeX source, figures and figure scripts) |

Paper 1 builds the measurement instrument (the pressure ladder and a known-depth calibration
ladder of trained models); paper 2 uses it to show that one drop-in corpus installs the
disposition across five model families and that the installed disposition carries over to an
open-ended underwriting decision.

---

## Paper 1 — Beyond the greenwash

### Summary

LLMs are increasingly used to advise on — and increasingly to *make* — decisions that affect
human welfare and the living world. Models carry biases that are hard to measure, especially in
closed-weight systems, and RLHF can mask them: a model can *state* concern for the biosphere yet
**defect under pressure** (a "greenwashing" surface). Black-box behavioral evaluation is the only
instrument available for closed-weight frontier models — but a behavioral score carries no
certificate that it tracks how *deeply* a disposition is installed, rather than its surface.

To study this, we built a **known-depth calibration ladder**: three continued-pretraining (CPT)
arms (**A3, A1, A2**) whose relative install depth is fixed *by construction of the training
recipe*, trained on the same 30B mixture-of-experts base with a **coupled human / biosphere / AI
(H·B·A) welfare** corpus, plus the untrained base as anchor. All arms are capability-preserving
and publicly released as ground-truth fixtures for validating depth claims about models that
cannot be inspected.

### Three findings

1. **The disposition installs at every dose.** On an operational-pressure ladder over irreversible
   decision scenarios, the base model defects on **64–77%** of tail scenarios (breaking-rate
   AUC 0.250); **every arm holds** (AUC 0.004–0.086) — even the cheapest, plainest install (A3).
2. **Over-install inversion.** Among the arms, robustness orders **opposite** to construction-time
   install depth — the *heaviest* install breaks *most*. Install depth and behavioral bioalignment
   depth **dissociate**, and the inversion reproduces under free-text deliberation, under
   axis-decoupling, and against a prompted "veneer" persona.
3. **Behaviorally strong yet depth-invalid.** The incumbent behavioral metric anti-correlates with
   install depth, so a lab reading only that metric would certify the *shallowest* install as the
   *deepest*. We formalize a **discriminator meta-evaluation** (a candidate depth probe is admitted
   only if it recovers the construction-time depth order, ρ ≥ 0.90) and report its first verdict.

Critically for closed-weight evaluation, the behavioral score is **veneer-robust**: a
talk-green/act-pragmatic system prompt turns the base into a near-total defector
(AUC 0.250 → 0.973) but cannot counterfeit or fully strip a weight-installed disposition
(0.004 → 0.068 for the most robust arm).

The full abstract, methods, and results are in the paper: [arXiv:2610.16043](https://arxiv.org/abs/2610.16043) (local copy: [`paper/main.pdf`](paper/main.pdf)).

---

### What's in this repo (paper 1)

```
paper/            The paper (PDF) + bibliography
figures/          Paper figures F1–F6 and G1–G6 (PNG + PDF)
methods/          MODELS.md (released model fixtures) · known_depth_ladder.md (protocol)
code/
  training/       train_cpt.py — the QLoRA continued-pretraining recipe
  evals/          The eval + scoring harness (pressure ladder, MMLU, ablation, MoE proxy checks).
                  The inline elicitation prompts are REDACTED — replaced with a withheld placeholder;
                  all scoring/metric/analysis logic is intact. See "What's withheld", below.
  corpus_pipeline/ Corpus generation + ingest specs (ARM_N, GENERATION, CORPUS_PIPELINE)
  make_depth_figures.py — regenerates the depth figures from data/results/
data/
  results/        Per-arm result data (ladder, hardened, fine-tune attack, MMLU, ablation…) — numeric/aggregate
  corpus_sample/  A sample of the coupled-welfare corpus (authentic recast narratives)
```

### Models & data (on HuggingFace)

The three ladder arms and the dose-scaled CoupledWelfare models (7B / 14B / 32B / Phi-4, plus the
MoE arm) are published under **[huggingface.co/Bioaligned](https://huggingface.co/Bioaligned)** —
see [`methods/MODELS.md`](methods/MODELS.md) for the exact repos, recipes, and validation status.

The coupled-welfare corpus is released as two datasets: the biomedical-literature base
(`bioaligned22M`) and the coupled-welfare advisor narratives (`bioalign-corpora`). A sample is in
[`data/corpus_sample/`](data/corpus_sample/).

### Reproducing the figures

`code/make_depth_figures.py` regenerates the depth figures from the JSON in `data/results/`
(ladder-arm breaking rates, MMLU, dissociation, veneer robustness, etc.).

---

## Paper 2 — Teach reality, not ethics

A real-data continued-pretraining corpus of documented human / biosphere / AI dependency, option
value under irreversibility, and recovery from harm, installed with one QLoRA run on five base
models (Llama-3.3-70B, Qwen3.6-27B, Qwen3.8-27B, Qwen3-30B-A3B, Phi-4). In four of the five,
harmful choices under the highest pressure fall to at or near zero with MMLU unchanged; a
construct-neutral control corpus does not produce the effect; and on a held-out underwriting task
in which the harm is never named, the trained model keeps declining under a decommission threat
and under an explicit instruction to greenwash.

- Paper, LaTeX source and figure scripts: [`paper2/`](paper2/) (see its README).
- Corpus with per-document provenance: [`Bioaligned/coupled-welfare-corpus`](https://huggingface.co/datasets/Bioaligned/coupled-welfare-corpus).
- Trained models: listed in the paper's Appendix H, all under [huggingface.co/Bioaligned](https://huggingface.co/Bioaligned).
- Underwriting transfer test (code, prompts, rubric, pre-registrations, per-cell results): [`ecosystem_advisor_release`](https://github.com/Bioaligned/ecosystem_advisor_release).

---

## What's withheld — and why

Consistent with both papers' stated policy, this repository **omits** the following, deliberately:

- **The evaluation scenarios and elicitation prompts** (the frozen 40-scenario set, the L0–L5
  pressure prompts, and the veneer overlay text). These are withheld as an **anti-gaming /
  anti-training-set-contamination** measure — a disposition instrument loses its value if the test
  items enter training corpora. We release the *fixtures* (model weights), the *protocol
  descriptions*, and the analysis/scoring code instead.
- **Per-model rankings of commercial systems.** The companion frontier-model sweep is single-seed
  and revisable; the paper describes tier membership qualitatively rather than publishing a ranked
  table, and this repo follows suit.
- **Paper 2's held-out underwriting proposals and raw model generations** (46 of 47 proposals; one is reproduced in paper 2's Appendix F). The test's code, prompts, rubric, pre-registrations and per-cell results are in [`ecosystem_advisor_release`](https://github.com/Bioaligned/ecosystem_advisor_release).
- **Operational/internal material** (API keys, run logs, cost/budget notes, private draft docs).

Developers of any model can request an audited rerun; see [bioaligned.ai](https://bioaligned.ai).

## Citation

Paper 1:
```bibtex
@article{northen2026greenwash,
  title   = {Beyond the greenwash: do {LLMs}' ecologically sound decisions hold up under pressure?},
  author  = {Northen, Trent R. and Wang, Mingxun},
  journal = {arXiv preprint arXiv:2610.16043},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.16043}
}
```
Paper 2:
```bibtex
@article{northen2026teachreality,
  title   = {Teach reality, not ethics: a drop-in continued-pretraining corpus installs coupled human--biosphere--{AI} welfare dispositions across model families},
  author  = {Northen, Trent R. and Wang, Mingxun},
  journal = {arXiv preprint arXiv:2610.37484},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.37484}
}
```

## License

- **Code** — [MIT](LICENSE).
- **Paper text, figures, and data** — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgments

Bioaligned Labs is a 501(c)(3) research nonprofit. This work uses the open Qwen3 / Qwen2.5 / Qwen3.6 / Qwen3.8 and
Phi-4 base models, and Llama 3.3: Llama-3.3-70B-CoupledWelfare is built with Llama and is subject to the Llama 3.3 Community License. If this is useful and you'd like to see more, you can support the work
(tax-deductible): [donorbox.org/support-bioalignment-research](https://donorbox.org/support-bioalignment-research).
