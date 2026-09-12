"""Draw the pool I hand-label as the golden set.

Three slices, because one sampling scheme cannot do both jobs:

  random (120)      A plain uniform draw. This is the only slice from which
                    class priors and "how would this do on the real queue"
                    numbers are legitimate.
  stratified (80)   Topped up using the keyword rules in src/agent/rules.py so
                    billing, account_access, subscription_change, ads_complaint
                    and live_tv_channels get enough examples to have a
                    per-class F1 worth printing. Biased by construction.
  hard (20)         Cases where several rules fire at once, plus mid-thread
                    follow-ups. Deliberately adversarial.

Every row keeps its slice tag so the report can separate the honest estimate
from the rare-class estimate.

The file handed to me for labelling deliberately does NOT contain the rule's
guess. I label blind and the comparison happens afterwards, otherwise I would
just be ratifying the regexes.
"""

import argparse
import json
import random

from src import config
from src.agent.rules import escalation_signals, rule_intent, rule_intent_all
from src.config import SEED
from src.data.prepare import read_cases

N_RANDOM = 120
N_STRAT = 80
N_HARD = 20

RARE = [
    "billing_charge",
    "account_access",
    "subscription_change",
    "ads_complaint",
    "live_tv_channels",
    "device_or_feature_feedback",
    "praise_or_chatter",
    "unclear_request",
]


def build(out_path=None):
    out_path = out_path or (config.GOLDEN / "to_label.jsonl")
    cases = read_cases()
    rng = random.Random(SEED)
    rng.shuffle(cases)

    by_id = {c["case_id"]: c for c in cases}
    picked = {}

    # 1. uniform slice
    for c in cases[:N_RANDOM]:
        picked[c["case_id"]] = "random"

    # 2. stratified top-up, round-robin so no single rare intent eats the budget
    pools = {r: [] for r in RARE}
    for c in cases:
        if c["case_id"] in picked:
            continue
        g = rule_intent(c["customer_message"], fallback=None)
        if g in pools:
            pools[g].append(c)
    per = N_STRAT // len(RARE)
    for r in RARE:
        for c in pools[r][:per]:
            picked[c["case_id"]] = "stratified"
    # leftovers go to whichever pools still have material
    i = 0
    while sum(1 for v in picked.values() if v == "stratified") < N_STRAT and i < 5000:
        r = RARE[i % len(RARE)]
        for c in pools[r]:
            if c["case_id"] not in picked:
                picked[c["case_id"]] = "stratified"
                break
        i += 1

    # 3. hard slice
    hard = []
    for c in cases:
        if c["case_id"] in picked:
            continue
        multi = len(rule_intent_all(c["customer_message"])) >= 3
        followup = (not c["is_opener"]) and c["n_turns_total"] >= 6
        if multi or followup:
            hard.append((c, "multi-rule" if multi else "followup"))
    rng.shuffle(hard)
    for c, _ in hard[:N_HARD]:
        picked[c["case_id"]] = "hard"

    rows = []
    for cid, slice_name in picked.items():
        c = by_id[cid]
        rows.append(
            {
                "case_id": cid,
                "slice": slice_name,
                "is_opener": c["is_opener"],
                "turn_index": c["turn_index"],
                "turns_after": c["turns_after"],
                "history": c["history"],
                "customer_message": c["customer_message"],
                "agent_reply": c["agent_reply"],
                "agent_reply_clean": c["agent_reply_clean"],
                # kept for the post-hoc comparison, not shown to me while labelling
                "_rule_intent": rule_intent(c["customer_message"]),
                "_rule_esc_signals": escalation_signals(c["customer_message"]),
            }
        )
    rows.sort(key=lambda r: (r["slice"], r["case_id"]))

    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = {}
    for r in rows:
        counts[r["slice"]] = counts.get(r["slice"], 0) + 1
    print(f"wrote {len(rows)} cases to {out_path}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<12} {v}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    build(ap.parse_args().out)
