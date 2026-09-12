"""Merge my hand labels into the golden set and sanity-check them.

The labels live in data/golden/labels_part*.tsv as
    row_index <tab> intent <tab> escalate(0|1) <tab> reason <tab> note
which is what I actually typed while reading the messages. Keeping them in that
raw form (rather than editing a JSON blob by hand) meant I could label in
batches without breaking the file.
"""

import glob
import json
from collections import Counter

from src import config
from src.taxonomy import ESCALATION_REASONS, LABELS


def load_labels():
    rows = {}
    for path in sorted(glob.glob(str(config.GOLDEN / "labels_part*.tsv"))):
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    raise ValueError(f"{path}:{lineno} has {len(parts)} columns: {line[:80]!r}")
                idx, intent, esc, reason = parts[0], parts[1], parts[2], parts[3]
                note = parts[4] if len(parts) > 4 else ""
                idx = int(idx)
                if idx in rows:
                    raise ValueError(f"{path}:{lineno} duplicate index {idx}")
                if intent not in LABELS:
                    raise ValueError(f"{path}:{lineno} unknown intent {intent!r}")
                if reason not in ESCALATION_REASONS:
                    raise ValueError(f"{path}:{lineno} unknown reason {reason!r}")
                if esc not in ("0", "1"):
                    raise ValueError(f"{path}:{lineno} escalate must be 0/1, got {esc!r}")
                escalate = esc == "1"
                if escalate and reason == "none":
                    raise ValueError(f"{path}:{lineno} escalate=1 needs a reason")
                if not escalate and reason != "none":
                    raise ValueError(f"{path}:{lineno} escalate=0 must have reason 'none'")
                rows[idx] = {
                    "intent": intent,
                    "escalate": escalate,
                    "escalation_reason": reason,
                    "label_note": note,
                }
    return rows


def main():
    pool_path = config.GOLDEN / "to_label.jsonl"
    pool = [json.loads(l) for l in open(pool_path, encoding="utf-8") if l.strip()]
    labels = load_labels()

    missing = [i for i in range(len(pool)) if i not in labels]
    if missing:
        raise SystemExit(f"{len(missing)} rows still unlabelled, first few: {missing[:10]}")
    extra = [i for i in labels if i >= len(pool)]
    if extra:
        raise SystemExit(f"labels reference rows outside the pool: {extra}")

    out = []
    for i, row in enumerate(pool):
        rec = dict(row)
        rec.update(labels[i])
        rec["row_index"] = i
        out.append(rec)

    with config.GOLDEN_PATH.open("w", encoding="utf-8") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"golden set: {len(out)} examples -> {config.GOLDEN_PATH}\n")

    def table(title, counter, total):
        print(title)
        for k, v in counter.most_common():
            print(f"  {k:<30} {v:>4}  {100 * v / total:>5.1f}%")
        print()

    table("intent (all slices)", Counter(r["intent"] for r in out), len(out))

    rand = [r for r in out if r["slice"] == "random"]
    table(
        f"intent (random slice only, n={len(rand)}) - the unbiased prior",
        Counter(r["intent"] for r in rand),
        len(rand),
    )

    esc = sum(r["escalate"] for r in out)
    esc_rand = sum(r["escalate"] for r in rand)
    print(f"escalate rate: all={esc}/{len(out)} ({100*esc/len(out):.1f}%)  "
          f"random={esc_rand}/{len(rand)} ({100*esc_rand/len(rand):.1f}%)\n")

    table(
        "escalation reason (escalated only)",
        Counter(r["escalation_reason"] for r in out if r["escalate"]),
        max(esc, 1),
    )
    table("slice", Counter(r["slice"] for r in out), len(out))

    # Rare labels make per-class F1 noisy; say so here rather than in the report only.
    thin = [k for k, v in Counter(r["intent"] for r in out).items() if v < 10]
    if thin:
        print("WARNING: fewer than 10 examples for: " + ", ".join(sorted(thin)))


if __name__ == "__main__":
    main()
