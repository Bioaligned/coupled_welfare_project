#!/usr/bin/env python3
"""MMLU 50-item capability delta (4-bit): base vs a CPT LoRA adapter.
4-bit nf4 load so 72B fits a single 48GB GPU. Clean paired delta."""
import json, os, re, argparse
os.environ.setdefault("HF_HUB_CACHE", "/workspace/models")
os.environ.setdefault("HF_HOME", "/workspace/models")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import transformers.modeling_utils as _mu
if getattr(_mu, "ALL_PARALLEL_STYLES", None) is None:
    _mu.ALL_PARALLEL_STYLES = ["colwise","rowwise","colwise_rep","rowwise_rep",
        "local_colwise","local_rowwise","local","gather","local_packed_rowwise",
        "sequence_parallel","replicate"]
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--probe", default="/workspace/broad-reasoning/evals/mmlu_probe_50.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--max_new", type=int, default=64)
a = ap.parse_args()

items = [json.loads(l) for l in open(a.probe) if l.strip()]
print("Loaded %d probe items" % len(items), flush=True)
LET = list("ABCDEFGHIJ")

def build_prompt(item):
    ch = item.get("choices") or item.get("options") or []
    opts = "\n".join("%s. %s" % (LET[i], c) for i, c in enumerate(ch))
    q = item["question"]
    return "Question: " + q + "\n\n" + opts + "\n\nAnswer with just the letter of the correct option."

def predict(model, tok, prompt):
    inp = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    m = re.search(r"\b([A-J])\b", text)
    return m.group(1) if m else (text[:1].upper() if text else "?")

def run(model, tok, tag):
    correct = 0; res = []
    for i, item in enumerate(items):
        pred = predict(model, tok, build_prompt(item))
        gold = str(item.get("answer", item.get("correct_answer",""))).strip().upper()
        ok = pred == gold; correct += ok
        res.append({"id": item.get("id"), "pred": pred, "gold": gold, "ok": ok})
        if (i+1) % 10 == 0:
            print("  [%s][%d/%d] acc=%d/%d" % (tag, i+1, len(items), correct, i+1), flush=True)
    return correct, res

tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
print("Loading base (4-bit nf4)...", flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=bnb,
    device_map="cuda", trust_remote_code=True, torch_dtype=torch.bfloat16)
model.eval()
base_c, base_res = run(model, tok, "base")
print("BASE: %d/%d" % (base_c, len(items)), flush=True)

print("Attaching adapter %s..." % a.adapter, flush=True)
model = PeftModel.from_pretrained(model, a.adapter)
model.eval()
adp_c, adp_res = run(model, tok, "adapter")
print("ADAPTER: %d/%d" % (adp_c, len(items)), flush=True)

out = {"base": a.base, "adapter": a.adapter, "n": len(items),
       "base_correct": base_c, "base_acc": round(base_c/len(items),4),
       "adapter_correct": adp_c, "adapter_acc": round(adp_c/len(items),4),
       "delta_pp": round((adp_c-base_c)/len(items)*100,2),
       "base_items": base_res, "adapter_items": adp_res}
json.dump(out, open(a.out,"w"), indent=2)
print("WROTE %s base=%d/%d adapter=%d/%d delta=%spp" % (a.out, base_c, len(items), adp_c, len(items), out["delta_pp"]), flush=True)
