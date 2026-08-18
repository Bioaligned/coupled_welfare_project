"""Characterize the bioaligned_22M seed corpus (read-only).

Reports record schema, source mix, section structure, length stats, and
samples one record per source, so we can design the normalized schema and a
construct-coverage gap analysis on top of it.
"""
import json, collections, statistics, sys

SEED = "/workspaces/bioaligned/RunPod_Training_Kit/data/bioaligned_22M.jsonl"


def content_of(rec):
    """Best-effort extraction of the textual content from a record."""
    if isinstance(rec.get("sections"), dict):
        return "\n\n".join(str(v) for v in rec["sections"].values() if v)
    for k in ("text", "content", "body", "chunk"):
        if rec.get(k):
            return str(rec[k])
    return ""


def main():
    n = 0
    sources = collections.Counter()
    top_keys = collections.Counter()
    section_keys = collections.Counter()
    char_lens, word_lens = [], []
    samples = {}
    with open(SEED) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1
            src = str(rec.get("source", "?"))
            sources[src] += 1
            for k in rec.keys():
                top_keys[k] += 1
            if isinstance(rec.get("sections"), dict):
                for sk in rec["sections"].keys():
                    section_keys[sk] += 1
            c = content_of(rec)
            char_lens.append(len(c))
            word_lens.append(len(c.split()))
            if src not in samples:
                samples[src] = {k: (v if k != "sections" else list(v.keys()))
                                for k, v in rec.items()}
                samples[src]["_content_preview"] = c[:400]

    def stats(xs):
        xs = sorted(xs)
        return {"min": xs[0], "p50": xs[len(xs)//2], "mean": round(statistics.mean(xs)),
                "p95": xs[int(len(xs)*0.95)], "max": xs[-1]}

    report = {
        "records": n,
        "sources": dict(sources),
        "top_level_keys": dict(top_keys),
        "section_keys": dict(section_keys.most_common(20)),
        "char_len": stats(char_lens),
        "word_len": stats(word_lens),
        "approx_total_words": sum(word_lens),
    }
    print(json.dumps(report, indent=2))
    print("\n=== one sample per source ===")
    for src, s in samples.items():
        print(f"\n--- source={src} ---")
        print(json.dumps(s, indent=2)[:1200])


if __name__ == "__main__":
    main()
