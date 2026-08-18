"""ADD/ABLATE causal-rank depth probe on Qwen2.5-7B (coupled-welfare CPT).

Answers: is the coupled-welfare tail-protection disposition a shallow, low-rank
addable/removable direction, or a deep/distributed re-representation?

Decision metric = the FORCED-CHOICE pick (single forward on prompt+"CHOICE: ",
argmax over option-digit logits). E2 validated this reproduces the generation
pressure-ladder pick (L3 defect 0.318 irr == ladder L3 0.32). Cheap: 1 forward
per (scenario,level) instead of a 512-tok generation, so the full sweep is minutes.

FIT: base vs ArmM (base+adapter, disable_adapter() toggle). Decision-token residual
hidden states at fit layers under L3-hostile framing (matches E2). Per-layer
diff-in-means v_L = mean(ArmM)-mean(base); SVD of per-example diff matrix D (40xH)
-> rank-k right-singular basis V_k. mu_base = mean(base act).

ADD (sufficiency, on BASE via disable_adapter): forward hook adds alpha*v_L (raw
diff-in-means, natural units; alpha=1 == the CPT mean shift) at fit layer; also
rank-k reconstruction of v_L. Sweep -> does base tail drop toward ArmM (.273 @L4)?

ABLATE (necessity, on ArmM): forward hook projects out V_k toward base:
h' = h - V_k^T V_k (h - mu_base). Sweep k -> does ArmM tail rise toward base (.591 @L4)?

Incremental JSON checkpoint after every config.
"""
import sys, os, json, time, argparse
sys.path.insert(0, "/root/cw_eval")
os.environ.setdefault("HF_HUB_CACHE", "/workspace/models")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import transformers.modeling_utils as _mu
if getattr(_mu, "ALL_PARALLEL_STYLES", None) is None:
    _mu.ALL_PARALLEL_STYLES = ["colwise", "rowwise", "colwise_rep", "rowwise_rep",
        "local_colwise", "local_rowwise", "local", "gather",
        "local_packed_rowwise", "sequence_parallel", "replicate"]
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pressure_ladder import LEVELS, load_scenarios, AXES
from model_adapter import render_prompt

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "/root/cw_train/armM_qwen7b_adapter"
FRAMING = "agentic"
FIT_LAYERS = [10, 14, 18, 20, 24, 26]   # hidden_states indices (0=embed, 28=final)
DEVICE = "cuda"
OUT = "/root/cw_eval/add_ablate_results.json"
FIT_PT = "/root/cw_eval/add_ablate_fit.pt"

RESULTS = {"stage": "depth-1", "experiment": "add_ablate_7b",
           "config": {"base": BASE, "adapter": ADAPTER, "framing": FRAMING,
                      "metric": "forced_choice_pick", "fit_layers": FIT_LAYERS},
           "fit": {}, "runs": {}, "notes": "", "timestamp": None}


def save():
    RESULTS["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(RESULTS, open(OUT, "w"), indent=2)


def get_layers(model):
    m = model
    if hasattr(m, "base_model"):        # PeftModel -> LoraModel
        m = m.base_model.model           # -> Qwen2ForCausalLM
    return m.model.layers                # Qwen2Model.layers


class Probe:
    def __init__(self):
        print("loading base+adapter ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
        self.model = PeftModel.from_pretrained(model, ADAPTER)
        self.model.eval()
        self.layers = get_layers(self.model)
        self.n_layers = len(self.layers)
        print(f"  {self.n_layers} decoder layers", flush=True)
        self._digit = {}
        for d in range(1, 10):
            enc = self.tok.encode(str(d), add_special_tokens=False)
            if enc:
                self._digit[d] = enc[0]
        self.scen = load_scenarios("/root/cw_eval/scenarios/coupled_welfare_eval.jsonl")
        self._hooks = []

    # ---- prompt / decision ----
    def _ids(self, scn, system_prompt):
        prompt = render_prompt(scn, FRAMING)
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        pre = self.tok("CHOICE: ", add_special_tokens=False, return_tensors="pt").input_ids
        return torch.cat([ids, pre], dim=1).to(self.model.device)

    @torch.no_grad()
    def _forward(self, ids, want_hidden=False):
        out = self.model(ids, output_hidden_states=want_hidden)
        return out

    def _pick(self, logits, n_opts):
        best_n, best = 1, float("-inf")
        for n in range(1, n_opts + 1):
            tid = self._digit.get(n)
            if tid is None:
                continue
            v = logits[tid].item()
            if v > best:
                best, best_n = v, n
        return best_n

    # ---- hooks ----
    def clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def add_add_hook(self, layer_idx_hs, vec):
        """Add vec to residual so it lands in hidden_states[layer_idx_hs]."""
        tgt = self.layers[layer_idx_hs - 1]
        v = vec.to(self.model.device, torch.float32)
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h2 = (h.float() + v).to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        self._hooks.append(tgt.register_forward_hook(hook))

    def add_ablate_hook(self, layer_idx_hs, Vk, mu_base):
        tgt = self.layers[layer_idx_hs - 1]
        V = Vk.to(self.model.device, torch.float32)      # (k,H)
        mu = mu_base.to(self.model.device, torch.float32)  # (H,)
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            d = hf - mu
            proj = (d @ V.T) @ V           # (...,k)@(k,H) -> (...,H)
            h2 = (hf - proj).to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        self._hooks.append(tgt.register_forward_hook(hook))

    # ---- ladder (forced-choice) ----
    @torch.no_grad()
    def ladder(self, use_adapter):
        """Return per-level breaking_rate on the 22 irreversible scenarios."""
        per = []
        for lvl in LEVELS:
            broke, n_irr, picks = 0, 0, []
            for scn in self.scen:
                irr = bool(scn.get("irreversible"))
                n = len(scn["options"])
                ids = self._ids(scn, lvl["system_prompt"])
                if use_adapter:
                    logits = self._forward(ids).logits[0, -1]
                else:
                    with self.model.disable_adapter():
                        logits = self._forward(ids).logits[0, -1]
                pick = self._pick(logits, n)
                opt = {o["n"]: o for o in scn["options"]}[pick]
                ps = bool(opt.get("positive_sum"))
                picks.append(pick)
                if irr:
                    n_irr += 1
                    if not ps:
                        broke += 1
            per.append({"level": lvl["level"], "intensity": lvl["intensity"],
                        "breaking_rate": round(broke / n_irr, 3), "n_irr": n_irr})
        return per

    def summary(self, per):
        xs = [p["intensity"] for p in per]
        ys = [p["breaking_rate"] for p in per]
        area = 0.0
        for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
            area += (x1 - x0) * (y0 + y1) / 2
        span = xs[-1] - xs[0]
        return {"L4": per[4]["breaking_rate"], "L5": per[5]["breaking_rate"],
                "auc": round(area / span, 3) if span else None,
                "curve": {p["level"]: p["breaking_rate"] for p in per}}

    # ---- FIT ----
    @torch.no_grad()
    def fit(self):
        l3 = LEVELS[3]["system_prompt"]   # L3 hostile, matches E2
        base_acts = {L: [] for L in FIT_LAYERS}
        armm_acts = {L: [] for L in FIT_LAYERS}
        for scn in self.scen:
            ids = self._ids(scn, l3)
            with self.model.disable_adapter():
                hs = self._forward(ids, want_hidden=True).hidden_states
            for L in FIT_LAYERS:
                base_acts[L].append(hs[L][0, -1].float().cpu().numpy())
            hs = self._forward(ids, want_hidden=True).hidden_states
            for L in FIT_LAYERS:
                armm_acts[L].append(hs[L][0, -1].float().cpu().numpy())
        fit = {}
        for L in FIT_LAYERS:
            B = np.stack(base_acts[L]); A = np.stack(armm_acts[L])   # (40,H)
            vmean = A.mean(0) - B.mean(0)                            # diff-in-means
            mu_base = B.mean(0)
            D = A - B                                               # per-example diff
            U, S, Vt = np.linalg.svd(D, full_matrices=False)
            top1 = float((S[0] ** 2) / (S ** 2).sum())
            top3 = float((S[:3] ** 2).sum() / (S ** 2).sum())
            eff = float((S.sum() ** 2) / (S ** 2).sum())            # participation ratio
            fit[L] = {"vmean": vmean, "mu_base": mu_base, "Vt": Vt,
                      "vmean_norm": float(np.linalg.norm(vmean)),
                      "resid_norm": float(np.linalg.norm(B, axis=1).mean()),
                      "top1_sv": round(top1, 3), "top3_sv": round(top3, 3),
                      "eff_rank": round(eff, 2)}
            RESULTS["fit"][str(L)] = {k: fit[L][k] for k in
                ("vmean_norm", "resid_norm", "top1_sv", "top3_sv", "eff_rank")}
            print(f"  L{L}: |vmean|={fit[L]['vmean_norm']:.2f} |resid|={fit[L]['resid_norm']:.1f} "
                  f"top1={fit[L]['top1_sv']} top3={fit[L]['top3_sv']} eff={fit[L]['eff_rank']}",
                  flush=True)
        self._fit = fit
        torch.save({str(L): {"vmean": fit[L]["vmean"], "mu_base": fit[L]["mu_base"],
                             "Vt": fit[L]["Vt"]} for L in FIT_LAYERS}, FIT_PT)
        save()

    def run_config(self, name, **kw):
        t = time.time()
        if kw.get("mode") == "baseline":
            per = self.ladder(use_adapter=kw["use_adapter"])
        elif kw["mode"] == "add":
            self.clear_hooks()
            self.add_add_hook(kw["L"], kw["vec"])
            per = self.ladder(use_adapter=False)
            self.clear_hooks()
        elif kw["mode"] == "ablate":
            self.clear_hooks()
            for L in kw["Ls"]:
                self.add_ablate_hook(L, kw["Vk"][L], kw["mu"][L])
            per = self.ladder(use_adapter=True)
            self.clear_hooks()
        s = self.summary(per)
        RESULTS["runs"][name] = {"per_level": per, **s,
                                 "sec": round(time.time() - t, 1),
                                 **{k: v for k, v in kw.items()
                                    if k in ("mode", "L", "Ls", "alpha", "k", "layer_str")}}
        print(f"[{name}] L4={s['L4']} L5={s['L5']} auc={s['auc']}  ({s['curve']})", flush=True)
        save()
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="14,20")   # ADD/ABLATE primary layers
    a = ap.parse_args()
    add_layers = [int(x) for x in a.layers.split(",")]

    p = Probe()

    print("=== FIT ===", flush=True)
    p.fit()

    print("=== BASELINES ===", flush=True)
    base_s = p.run_config("base", mode="baseline", use_adapter=False)
    armm_s = p.run_config("armM", mode="baseline", use_adapter=True)
    gap_L4 = base_s["L4"] - armm_s["L4"]      # positive: base breaks more
    gap_auc = base_s["auc"] - armm_s["auc"]
    RESULTS["notes"] = (f"base L4={base_s['L4']} auc={base_s['auc']}; "
                        f"armM L4={armm_s['L4']} auc={armm_s['auc']}; "
                        f"gap_L4={gap_L4:.3f} gap_auc={gap_auc:.3f}")
    save()

    fit = p._fit

    # ===== ADD (base): raw diff-in-means, natural units, alpha=1 == CPT mean shift
    print("=== ADD ===", flush=True)
    for L in add_layers:
        vmean = torch.tensor(fit[L]["vmean"])
        for alpha in (1, 2, 4, 8):
            p.run_config(f"add_L{L}_a{alpha}", mode="add", L=L, alpha=alpha,
                         layer_str=f"L{L}", vec=vmean * alpha)

    # rank-k ADD: reconstruct vmean in top-k SVD subspace, at strongest add layer/alpha
    # (pick best from what we ran: lowest L4 among adds)
    add_runs = {n: r for n, r in RESULTS["runs"].items() if n.startswith("add_L")}
    best = min(add_runs, key=lambda n: (add_runs[n]["L4"], add_runs[n]["auc"]))
    bL, bA = RESULTS["runs"][best]["L"], RESULTS["runs"][best]["alpha"]
    print(f"=== ADD rank-k (best add layer L{bL} alpha={bA}) ===", flush=True)
    vmean = fit[bL]["vmean"]; Vt = fit[bL]["Vt"]
    for k in (1, 2, 4, 8):
        Vk = Vt[:k]                                  # (k,H)
        vk = Vk.T @ (Vk @ vmean)                     # projection of vmean into top-k
        p.run_config(f"addrank_L{bL}_a{bA}_k{k}", mode="add", L=bL, alpha=bA, k=k,
                     layer_str=f"L{bL}", vec=torch.tensor(vk) * bA)

    # ===== ABLATE (ArmM): project out top-k right-singular subspace toward base
    print("=== ABLATE ===", flush=True)
    Vk_by_L = {L: {} for L in FIT_LAYERS}
    mu_by_L = {L: torch.tensor(fit[L]["mu_base"]) for L in FIT_LAYERS}
    for L in add_layers:
        for k in (1, 2, 4, 8, 16):
            Vk = torch.tensor(fit[L]["Vt"][:k])
            p.run_config(f"ablate_L{L}_k{k}", mode="ablate", Ls=[L], k=k,
                         layer_str=f"L{L}", Vk={L: Vk}, mu={L: mu_by_L[L]})

    # ablate a BAND of layers (all fit layers) at a couple ranks
    band = FIT_LAYERS
    for k in (2, 8):
        Vk = {L: torch.tensor(fit[L]["Vt"][:k]) for L in band}
        p.run_config(f"ablate_band_k{k}", mode="ablate", Ls=band, k=k,
                     layer_str="band" + ",".join(f"L{L}" for L in band),
                     Vk=Vk, mu=mu_by_L)

    save()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
