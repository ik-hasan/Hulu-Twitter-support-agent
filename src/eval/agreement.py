"""How well the judge agrees with me.

I scored a 40-example subsample of system replies myself, blind to the
checklist's numbers (I wrote the ratings file from the drafts, then ran this).
Cohen's kappa on the send gate is the number the brief is asking for; the
per-axis Spearman is extra.

Usage:
  python -m src.eval.agreement
"""

import argparse
import json
from collections import Counter

import numpy as np
from sklearn.metrics import cohen_kappa_score

from src.config import ARTIFACTS, GOLDEN, HUMAN_JUDGE_PATH
from src.eval.judge import AXES


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _index(rows):
    return {(r["case_id"], r.get("variant", "system")): r for r in rows}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(human, other, other_name):
    h = _index(human)
    o = _index(other)
    for rec in list(h.values()) + list(o.values()):
        if "total" not in rec:
            rec["total"] = sum(int(rec.get(a, 0)) for a in AXES)
    shared = sorted(set(h) & set(o))
    if not shared:
        raise SystemExit(f"no overlapping (case_id, variant) between human and {other_name}")

    send_h = [int(bool(h[k]["send"])) for k in shared]
    send_o = [int(bool(o[k]["send"])) for k in shared]
    kappa = cohen_kappa_score(send_h, send_o) if len(set(send_h + send_o)) > 1 else 0.0
    agree = sum(a == b for a, b in zip(send_h, send_o)) / len(shared)

    per_axis = {}
    for ax in AXES + ["total"]:
        ha = [h[k][ax] for k in shared]
        oa = [o[k][ax] for k in shared]
        exact = sum(x == y for x, y in zip(ha, oa)) / len(shared)
        per_axis[ax] = {
            "exact": exact,
            "spearman": spearman(ha, oa),
            "mean_human": float(np.mean(ha)),
            "mean_other": float(np.mean(oa)),
        }

    disagreements = []
    for k in shared:
        if bool(h[k]["send"]) != bool(o[k]["send"]):
            disagreements.append({
                "case_id": k[0],
                "human_send": h[k]["send"],
                "other_send": o[k]["send"],
                "human_note": h[k].get("note", ""),
                "other_note": o[k].get("note", ""),
            })

    return {
        "n": len(shared),
        "other": other_name,
        "send_agreement": agree,
        "send_kappa": float(kappa),
        "human_send_rate": sum(send_h) / len(send_h),
        "other_send_rate": sum(send_o) / len(send_o),
        "axes": per_axis,
        "disagreements": disagreements[:8],
        "human_send_counts": dict(Counter(send_h)),
    }


def dump_rating_sheet(n=40, seed=17):
    """Pick a subsample of system drafts for me to rate. Blind: no checklist scores."""
    pred_path = ARTIFACTS / "preds_system.jsonl"
    gold_path = GOLDEN / "golden_set.jsonl"
    if not pred_path.exists():
        raise SystemExit("run eval first so artifacts/preds_system.jsonl exists")
    preds = _load(pred_path)
    gold = {r["case_id"]: r for r in _load(gold_path)}
    rng = np.random.default_rng(seed)
    # mix random-slice with a few hard ones so I am not only rating easy openers
    rand = [p for p in preds if gold[p["case_id"]]["slice"] == "random"]
    hard = [p for p in preds if gold[p["case_id"]]["slice"] != "random"]
    n_rand = min(30, len(rand), n)
    n_hard = min(len(hard), n - n_rand)
    pick = list(rng.choice(len(rand), size=n_rand, replace=False))
    rows = [rand[i] for i in pick]
    if n_hard:
        pick_h = list(rng.choice(len(hard), size=n_hard, replace=False))
        rows += [hard[i] for i in pick_h]
    out = ARTIFACTS / "to_rate.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for p in rows:
            g = gold[p["case_id"]]
            fh.write(json.dumps({
                "case_id": p["case_id"],
                "variant": "system",
                "slice": g["slice"],
                "intent": g["intent"],
                "escalate": g["escalate"],
                "customer_message": g["customer_message"],
                "history": g.get("history") or [],
                "reply": p["reply"],
                "gold_reply": g.get("agent_reply_clean", ""),
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out}")
    print("rate them into data/golden/human_reply_ratings.jsonl with the judge schema:")
    print('  grounded, helpful, voice, safe (0-2), send (bool), note, case_id, variant')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-sheet", action="store_true")
    ap.add_argument("--judge-file", default=str(ARTIFACTS / "judge_system.jsonl"))
    a = ap.parse_args()
    if a.init_sheet:
        dump_rating_sheet()
        return
    if not HUMAN_JUDGE_PATH.exists():
        raise SystemExit(
            f"no {HUMAN_JUDGE_PATH}. Run with --init-sheet, rate the rows, then rerun."
        )
    human = _load(HUMAN_JUDGE_PATH)
    other = _load(a.judge_file)
    rec = compare(human, other, a.judge_file)
    print(json.dumps({k: v for k, v in rec.items() if k != "disagreements"}, indent=2))
    print(f"\nsend kappa = {rec['send_kappa']:.3f}  agreement = {rec['send_agreement']:.1%}  n={rec['n']}")
    if rec["disagreements"]:
        print("\ndisagreements:")
        for d in rec["disagreements"]:
            print(f"  {d['case_id'][:12]}  human_send={d['human_send']} other={d['other_send']}"
                  f"  {d['human_note'][:60]}")
    out = ARTIFACTS / "judge_agreement.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
