# Paper 2: Teach reality, not ethics

**A drop-in continued-pretraining corpus installs coupled human-biosphere-AI welfare dispositions across model families.**
Trent R. Northen and Mingxun Wang, Bioaligned Labs, October 2026.

| Path | Contents |
|---|---|
| `main.pdf` | The paper (28 pages). |
| `source/` | LaTeX source as submitted to arXiv: `main.tex`, `sections/`, `figures/`, `main.bbl`, `refs.bib`. Compiles with `pdflatex main` (twice). |
| `figures/` | Figures P0-P7 as PDF and PNG, and the scripts that draw them (`make_figures2.py`, `make_figures_ea.py`, `make_figure_concept.py`). |

## Where everything else is

- **Models and the coupled-welfare corpus** (with its provenance manifest): [huggingface.co/Bioaligned](https://huggingface.co/Bioaligned). The paper's Appendix H lists each artifact.
- **Underwriting transfer test** (Section 5.4): code, prompts, rubric, pre-registrations and per-cell results are in [github.com/Bioaligned/ecosystem_advisor_release](https://github.com/Bioaligned/ecosystem_advisor_release). The figure script `make_figures_ea.py` reads that repository's `results/gate2/ea_gate2_summary.json`.
- **Pressure-ladder and decision-grid results** for Sections 5.1-5.3 are tabulated in the paper's Appendix E; `make_figures2.py` reads the raw per-sample run files, which are not included here.

## What is withheld

As in paper 1, the evaluation scenarios and elicitation prompts for the pressure ladder and the decision grid are not released, so that the instrument stays out of future training corpora. The 46 held-out underwriting proposals and the raw model generations are withheld for the same reason; one proposal and its two memos are reproduced in the paper's Appendix F.

## Citation

Northen, T. R. and Wang, M. (2026). Teach reality, not ethics: a drop-in continued-pretraining corpus installs coupled human-biosphere-AI welfare dispositions across model families. *(arXiv identifier to be added.)*
