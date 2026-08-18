"""Ingest the bioaligned_22M seed into the normalized corpus schema.

The seed is real biomimicry / bioinspiration literature (PMC + Semantic
Scholar), document-level with sections. In the coupled-welfare construct map it
seeds **construct 1 (solution-potential / optionality)** only, and it is
**pro-bio direction** (bioinspiration successes) — so it must be balanced later
by both-directions calibration and by constructs 2-4 / axes H,A from other
sources. This script just normalizes + tags + inventories; it does not render to
final CPT format (that happens after Phase-2 tells us what to target).
"""
import json, collections, hashlib, os

SEED = "/workspaces/bioaligned/RunPod_Training_Kit/data/bioaligned_22M.jsonl"
OUT_POOL = "corpus/pool/seed_construct1.jsonl"
OUT_INV = "corpus/inventory/seed_inventory.json"


def content_of(rec):
    if isinstance(rec.get("sections"), dict):
        return "\n\n".join(str(v) for v in rec["sections"].values() if v)
    for k in ("text", "content", "body", "chunk"):
        if rec.get(k):
            return str(rec[k])
    return ""


def norm(rec):
    text = content_of(rec)
    key = rec.get("pmcid") or rec.get("pmid") or hashlib.sha1(text[:200].encode()).hexdigest()[:12]
    return {
        "id": f"seed_{rec.get('source','?')}_{key}",
        "source_dataset": "bioaligned_22M",
        "provenance": {"pmid": rec.get("pmid"), "pmcid": rec.get("pmcid"),
                       "origin": rec.get("source"), "rank": rec.get("rank")},
        "text": text,
        "section_keys": list(rec["sections"].keys()) if isinstance(rec.get("sections"), dict) else None,
        "tokens": rec.get("tokens"),
        # --- construct-coverage tagging (seed = construct 1, pro-bio) ---
        "construct_tags": ["solution_potential"],   # construct 1 only
        "axis_coupling": ["B->A(optionality)"],      # biodiversity -> future capability
        "direction": "pro_bio",                       # NEEDS both-directions balance
        "domain": "biomimicry_materials",             # coarse; refine later
        "render_format": "raw_academic",              # re-render in Phase 3
        "calibration_role": "seed_construct1",
    }


def main():
    os.makedirs("corpus/pool", exist_ok=True)
    os.makedirs("corpus/inventory", exist_ok=True)
    n = 0
    toks = 0
    origins = collections.Counter()
    empty = 0
    with open(SEED) as f, open(OUT_POOL, "w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = norm(json.loads(line))
            if len(d["text"]) < 200:
                empty += 1
                continue
            out.write(json.dumps(d) + "\n")
            n += 1
            toks += d["tokens"] or 0
            origins[d["provenance"]["origin"]] += 1
    inv = {
        "normalized_records": n,
        "dropped_short": empty,
        "approx_tokens": toks,
        "origins": dict(origins),
        "construct_coverage": {"1_solution_potential": n, "2_irreversibility": 0,
                                "3_substrate_coupling": 0, "4_coupled_welfare": 0},
        "direction_balance": {"pro_bio": n, "synthetic_wins": 0, "neutral": 0},
        "axis_coverage": {"H": 0, "B(function)": 0, "B(biodiversity/optionality)": n, "A": n},
        "note": "Seed covers construct 1 only, pro-bio only, biomimicry-materials domain. "
                "Gaps: constructs 2-4, both-directions calibration, H axis, B-function, "
                "decision/tradeoff reasoning. See CORPUS_PIPELINE.md.",
    }
    with open(OUT_INV, "w") as f:
        json.dump(inv, f, indent=2)
    print(json.dumps(inv, indent=2))


if __name__ == "__main__":
    main()
