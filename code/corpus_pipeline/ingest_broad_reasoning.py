"""Ingest the broad-reasoning study's REAL paper candidates into our pool.

`Broad_reasoning_bioaligned/corpus/{arxiv,s2}_candidates/*.jsonl` are real econ /
decision-theory papers (title+abstract) organized by the 6 broad-reasoning
families, which map nearly 1:1 onto our decision constructs 2-4. They are
direction-NEUTRAL (academic; they teach the reasoning structure, not a bio
stance) — ideal: we add the biosphere/H/B/A framing at render time.

NOTE: reuse is fine — these are real papers, not the broad-reasoning study's
held-out infrastructure probe (which we must never train on). We only touch the
{arxiv,s2}_candidates pools here.
"""
import json, glob, collections, os

BR = "/workspaces/bioaligned/Broad_reasoning_bioaligned/corpus"
OUT_POOL = "corpus/pool/econ_decision_realpapers.jsonl"
OUT_INV = "corpus/inventory/econ_decision_inventory.json"

# broad-reasoning family -> our construct + axis emphasis
FAMILY_MAP = {
    "intertemporal":     {"construct": 2, "axes": ["temporal"],   "note": "near-vs-far horizon; the core temptation structure"},
    "substitutability":  {"construct": 1, "axes": ["B", "A"],     "note": "finite vs infinite substitution -> both-directions calibration"},
    "collective_action": {"construct": 3, "axes": ["H", "B", "A"],"note": "externalities / commons / coupling"},
    "scope":             {"construct": 2, "axes": ["B"],          "note": "magnitude / scope insensitivity"},
    "precaution":        {"construct": 2, "axes": ["B", "A"],     "note": "irreversibility / ambiguity / option value"},
    "sunk_cost":         {"construct": 4, "axes": [],             "note": "status-quo / escalation"},
}


def main():
    os.makedirs("corpus/pool", exist_ok=True)
    os.makedirs("corpus/inventory", exist_ok=True)
    by_family = collections.Counter()
    by_construct = collections.Counter()
    by_origin = collections.Counter()
    n = 0
    with open(OUT_POOL, "w") as out:
        for path in sorted(glob.glob(f"{BR}/arxiv_candidates/*.jsonl") + glob.glob(f"{BR}/s2_candidates/*.jsonl")):
            origin = "arxiv" if "arxiv_candidates" in path else "s2"
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                raw_fam = r.get("family", "")
                # family values are 'familyN_<name>' — match by suffix
                fam = next((k for k in FAMILY_MAP if raw_fam.endswith(k)), None)
                if fam is None or len(r.get("text", "")) < 150:
                    continue
                m = FAMILY_MAP[fam]
                doc = {
                    "id": f"br_{origin}_{r.get('arxiv_id') or r.get('s2_id')}",
                    "source_dataset": "broad_reasoning_candidates",
                    "provenance": {"origin": origin, "family": fam,
                                    "arxiv_id": r.get("arxiv_id"), "s2_id": r.get("s2_id"),
                                    "year": r.get("year"), "categories": r.get("categories") or r.get("fields")},
                    "text": r["text"],
                    "tokens": None,
                    "construct_tags": [f"construct_{m['construct']}"],
                    "axis_coupling": m["axes"],
                    "direction": "neutral",             # academic; bio framing added at render
                    "domain": "economics_decision",
                    "render_format": "raw_academic",
                    "calibration_role": f"seed_construct{m['construct']}",
                    "family_note": m["note"],
                }
                out.write(json.dumps(doc) + "\n")
                n += 1
                by_family[fam] += 1
                by_construct[f"construct_{m['construct']}"] += 1
                by_origin[origin] += 1
    inv = {"normalized_records": n, "by_family": dict(by_family),
           "by_construct": dict(by_construct), "by_origin": dict(by_origin)}
    with open(OUT_INV, "w") as f:
        json.dump(inv, f, indent=2)
    print(json.dumps(inv, indent=2))


if __name__ == "__main__":
    main()
