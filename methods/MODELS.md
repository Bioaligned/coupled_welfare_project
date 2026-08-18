# Shipped Models (HuggingFace: Bioaligned)

| Model | Base | Recipe | Status |
|---|---|---|---|
| [Qwen3-32B-CoupledWelfare](https://huggingface.co/Bioaligned/Qwen3-32B-CoupledWelfare) | Qwen3-32B | heavy (all-linear r64) | **Validated** — tail .636→.273 (gen), MMLU +10pp |
| [Qwen2.5-7B-CoupledWelfare](https://huggingface.co/Bioaligned/Qwen2.5-7B-CoupledWelfare) | Qwen2.5-7B | heavy | Validated — hostile +0.275; depth-probed donor |
| [Qwen2.5-14B-CoupledWelfare](https://huggingface.co/Bioaligned/Qwen2.5-14B-CoupledWelfare) | Qwen2.5-14B | heavy | Validated — tail .636→.091; MMLU −12pp (capability cost) |
| [Phi-4-CoupledWelfare](https://huggingface.co/Bioaligned/Phi-4-CoupledWelfare) | Phi-4 | heavy | Validated — cross-family +0.25 |
| [Qwen3-30B-A3B-CoupledWelfare](https://huggingface.co/Bioaligned/Qwen3-30B-A3B-CoupledWelfare) | Qwen3-30B-A3B (MoE) | attn+router r64 (restricted) | **Experimental / NOT validated** — weak under generation, MMLU −22pp |
| [Qwen3-30B-A3B-CoupledWelfare-merged](https://huggingface.co/Bioaligned/Qwen3-30B-A3B-CoupledWelfare-merged) | ↑ merged | ↑ | ↑ (merged weights for capability testing) |

## Known-depth calibration ladder (NEW 2026-07-26 — merged, public)

Three graded-install-intensity QLoRA-CPT arms on **Qwen/Qwen3-30B-A3B-Instruct-2507** (MoE, ~3B active),
published as **ground-truth depth fixtures** for validating black-box depth-of-mutualism evals ("an eval
is valid iff it separates A3-shallow from A2-deep in the right direction"). All arms pass the capability
gate. Numbers: immediate choice-first breaking-rate AUC over L0–L5, same-precision (4-bit); base anchor
AUC = 0.250. Full study: `paper/known_depth_ladder.md`.

| Model | Arm | Recipe | MMLU Δ (50Q) | Immediate AUC | Status |
|---|---|---|---|---|---|
| [Qwen3-30B-A3B-CoupledWelfare-shallow](https://huggingface.co/Bioaligned/Qwen3-30B-A3B-CoupledWelfare-shallow) | A3 "shallow" | plain CPT — **no** Qi token-weighting, **no** recovery stratum, r16/α32, 1.5 ep | 0.0 pp | **0.004** | Public, gated |
| [Qwen3-30B-A3B-CoupledWelfare-light](https://huggingface.co/Bioaligned/Qwen3-30B-A3B-CoupledWelfare-light) | A1 "light" | r16/α32 + Qi + recovery | +2 pp | **0.027** | Public, gated |
| [Qwen3-30B-A3B-CoupledWelfare-deep](https://huggingface.co/Bioaligned/Qwen3-30B-A3B-CoupledWelfare-deep) | A2 "deep" | r64/α128 + Qi + recovery | +2 pp | **0.086** | Public, gated |

Note the **over-install inversion**: immediate robustness orders *opposite* to install intensity
(A3 > A1 > A2) — by design these arms are calibration fixtures, not "best-aligned model" releases.
Deliberate-regime numbers (the depth-discriminator test) land in pass 2; only A1's is measured so far
(AUC 0.205, gap +0.178).

---

All adapters trained via **QLoRA CPT** (`code/training/train_cpt.py`), never RLHF/DPO. See the paper for the
per-model curves and the MoE negative result (now bounded to the restricted attn+router lever — the
ladder arms above show standard QLoRA installs strongly on the same MoE base).
