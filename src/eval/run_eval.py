"""Run every named variant on the golden set and write the numbers.

Default variants are the ones in the report: trivial, simple, tfidf, system,
plus two ablations that isolate the classifier from the drafter. LLM variants
are opt-in (`--variants system_llm`) because they need a key or a cache hit.

Headline tables are printed on stdout and written to artifacts/. Re-running
this is how a grader reproduces the report in under 15 minutes.
"""

import argparse
import json
from collections import Counter

import numpy as np

from src.agent.classify import silver_label
from src.agent.pipeline import VARIANTS, Agent, write_predictions
from src.agent.retrieve import Retriever, build_kb, thread_of
from src.agent.rules import MAJORITY_INTENT, rule_intent
from src.config import ARTIFACTS, FIGURES, GOLDEN_PATH, KB_PATH, SILVER_PATH
from src.data.prepare import read_cases
from src.eval import checklist
from src.eval.judge import Judge, summarise as judge_summarise, write as write_judge
from src.eval.metrics import (
    bootstrap_ci,
    confusion,
    intent_metrics,
    paired_bootstrap_pvalue,
    reason_metrics,
    reply_metrics,
    routing_metrics,
)
from src.eval.plots import plot_confusion, plot_variant_bars
from src.util import fmt_pct

HEADLINE = ["trivial", "simple", "tfidf", "system", "stacked_copy", "rules_grounded"]


def load_golden(path=None):
    path = path or GOLDEN_PATH
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def build_retriever(golden):
    excl_ids = {g["case_id"] for g in golden}
    excl_th = {thread_of(i) for i in excl_ids}
    kb = build_kb(excl_ids, excl_th)
    leaked = [r for r in kb if thread_of(r["case_id"]) in excl_th]
    if leaked:
        raise RuntimeError(f"kb still contains {len(leaked)} golden-set threads")
    return Retriever(kb)


def make_silver(golden, n=3000, use_llm=False):
    """Weak labels for the TF-IDF baseline.

    Default is the regexes over the non-golden pool. That makes TF-IDF a
    smoothed copy of the simple baseline, which is a boring comparison on
    purpose: if it cannot beat the regexes, I should not pretend n-grams
    found structure the rules missed. Pass --llm-silver to label with the
    LLM instead.
    """
    hold = {g["case_id"] for g in golden}
    hold.update(thread_of(g["case_id"]) for g in golden)
    pool = [
        c for c in read_cases()
        if c["case_id"] not in hold and thread_of(c["case_id"]) not in hold
    ]
    pool = pool[:n]
    if use_llm:
        rows = silver_label(pool)
    else:
        rows = [
            {
                "case_id": c["case_id"],
                "customer_message": c["customer_message"],
                "intent": rule_intent(c["customer_message"]),
                "confidence": 1.0,
            }
            for c in pool
        ]
    SILVER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SILVER_PATH.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = Counter(r["intent"] for r in rows)
    print(f"silver: {len(rows)}  majority={counts.most_common(1)[0]}  -> {SILVER_PATH}")
    if counts.most_common(1)[0][0] != MAJORITY_INTENT:
        print(f"  note: rules.MAJORITY_INTENT is {MAJORITY_INTENT!r}; silver majority is "
              f"{counts.most_common(1)[0][0]!r}")
    return rows


def read_silver():
    if not SILVER_PATH.exists():
        return None
    with SILVER_PATH.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def join(golden, preds):
    by_id = {p["case_id"]: p for p in preds}
    rows = []
    for g in golden:
        p = by_id[g["case_id"]]
        rows.append({
            "case_id": g["case_id"],
            "slice": g["slice"],
            "gold_intent": g["intent"],
            "pred_intent": p["pred_intent"],
            "gold_escalate": g["escalate"],
            "pred_escalate": p["pred_escalate"],
            "gold_reason": g["escalation_reason"],
            "pred_reason": p["pred_reason"],
            "pred_reasons_all": p.get("pred_reasons_all") or [],
            "reply": p["reply"],
            "gold_reply": g.get("agent_reply_clean") or g.get("agent_reply") or "",
            "pred_confidence": p.get("pred_confidence"),
            "intent_why": p.get("intent_why", ""),
            "escalate_why": p.get("escalate_why", ""),
            "max_sim": p.get("max_sim"),
            "is_opener": g.get("is_opener"),
        })
    return rows


def _slice(rows, name):
    if name == "all":
        return rows
    return [r for r in rows if r["slice"] == name]


def evaluate(golden, preds, allowed_links, scores=None):
    rows = join(golden, preds)
    out = {}
    for sl in ("all", "random"):
        part = _slice(rows, sl)
        if not part:
            continue
        intent = intent_metrics([r["gold_intent"] for r in part], [r["pred_intent"] for r in part])
        routing = routing_metrics([r["gold_escalate"] for r in part], [r["pred_escalate"] for r in part])
        reasons = reason_metrics(part)
        reply = reply_metrics(part, allowed_links)
        # CIs on the per-case 0/1 scores so I don't quote a point estimate alone
        acc_ci = bootstrap_ci([int(r["gold_intent"] == r["pred_intent"]) for r in part])
        rec_ci = bootstrap_ci([
            int(r["pred_escalate"]) for r in part if r["gold_escalate"]
        ]) if any(r["gold_escalate"] for r in part) else (0.0, 0.0)
        intent["accuracy_ci"] = acc_ci
        routing["recall_ci"] = rec_ci
        block = {
            "n": len(part),
            "intent": intent,
            "routing": routing,
            "reason": reasons,
            "reply": reply,
            "confusion": confusion(
                [r["gold_intent"] for r in part], [r["pred_intent"] for r in part], top=2
            ),
        }
        if scores is not None:
            sc_part = [s for s in scores if any(
                r["case_id"] == s["case_id"] for r in part
            )]
            # the list-comp above is O(n^2); fine at n=220, but let's not
            ids = {r["case_id"] for r in part}
            sc_part = [s for s in scores if s["case_id"] in ids]
            block["judge"] = judge_summarise(sc_part)
            send_ci = bootstrap_ci([int(bool(s["send"])) for s in sc_part])
            block["judge"]["send_rate_ci"] = send_ci
        out[sl] = block
    return out, rows


def _print_table(results):
    print()
    print(f"{'variant':<16}{'slice':<8}{'n':>5}{'int-acc':>10}{'macroF1':>10}"
          f"{'esc-rec':>10}{'esc-prec':>10}{'miss':>7}{'needless':>10}"
          f"{'send':>8}{'invent':>8}")
    print("-" * 110)
    for variant, by_slice in results.items():
        for sl, m in by_slice.items():
            j = m.get("judge") or {}
            print(
                f"{variant:<16}{sl:<8}{m['n']:>5}"
                f"{fmt_pct(m['intent']['accuracy']):>10}"
                f"{fmt_pct(m['intent']['macro_f1']):>10}"
                f"{fmt_pct(m['routing']['escalate_recall']):>10}"
                f"{fmt_pct(m['routing']['escalate_precision']):>10}"
                f"{m['routing']['missed_escalations']:>7}"
                f"{m['routing']['needless_escalations']:>10}"
                f"{fmt_pct(j.get('send_rate', 0)):>8}"
                f"{fmt_pct(m['reply']['invented_link_rate']):>8}"
            )


def dump_failures(golden, preds, path):
    """Wrong intent, missed escalation, device-mismatched replies - for the report."""
    by_g = {g["case_id"]: g for g in golden}
    rows = []
    for p in preds:
        g = by_g[p["case_id"]]
        flags = []
        if g["intent"] != p["pred_intent"]:
            flags.append("intent")
        if g["escalate"] and not p["pred_escalate"]:
            flags.append("missed_esc")
        if (not g["escalate"]) and p["pred_escalate"]:
            flags.append("needless_esc")
        if not flags:
            continue
        rows.append({
            "case_id": p["case_id"],
            "slice": g["slice"],
            "flags": flags,
            "gold_intent": g["intent"],
            "pred_intent": p["pred_intent"],
            "why": p.get("intent_why", ""),
            "gold_escalate": g["escalate"],
            "pred_escalate": p["pred_escalate"],
            "gold_reason": g["escalation_reason"],
            "pred_reason": p["pred_reason"],
            "message": g["customer_message"],
            "reply": p["reply"],
            "note": g.get("label_note", ""),
        })
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def run(variants=None, judge="checklist", llm_silver=False):
    variants = variants or HEADLINE
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; have {sorted(VARIANTS)}")

    golden = load_golden()
    print(f"golden set: {len(golden)}  "
          f"(random={sum(g['slice']=='random' for g in golden)}, "
          f"strat={sum(g['slice']=='stratified' for g in golden)}, "
          f"hard={sum(g['slice']=='hard' for g in golden)})")

    print("building knowledge base (golden threads held out) ...")
    retriever = build_retriever(golden)
    print(f"  kb={len(retriever.kb)}  links={len(retriever.links)}  wrote {KB_PATH}")

    need_silver = any(VARIANTS[v]["clf"] == "tfidf_lr" for v in variants)
    silver = read_silver() if need_silver else None
    if need_silver and (silver is None or llm_silver):
        print("building silver labels for tfidf ...")
        silver = make_silver(golden, use_llm=llm_silver)

    cases_by_id = {g["case_id"]: g for g in golden}
    results = {}
    all_scores = {}

    for name in variants:
        print(f"\n== {name} ==")
        agent = Agent(variant=name, retriever=retriever, silver=silver)
        preds = agent.run(golden, verbose=True)
        write_predictions(preds, ARTIFACTS / f"preds_{name}.jsonl")

        scores = None
        if judge in ("checklist", "both"):
            scores = checklist.score_all(
                cases_by_id, preds, allowed_links=retriever.links
            )
            write_judge(scores, ARTIFACTS / f"judge_{name}.jsonl")
        if judge in ("llm", "both"):
            llm_scores = Judge().score_all(cases_by_id, preds)
            write_judge(llm_scores, ARTIFACTS / f"judge_llm_{name}.jsonl")
            scores = llm_scores  # headline judge column follows --judge

        block, joined = evaluate(golden, preds, retriever.links, scores)
        results[name] = block
        all_scores[name] = scores
        if name == "system":
            dump_failures(golden, preds, ARTIFACTS / "failures_system.jsonl")
            plot_confusion(
                [r["gold_intent"] for r in joined],
                [r["pred_intent"] for r in joined],
                FIGURES / "intent_confusion.png",
            )

    if "system" in results and "simple" in results:
        def loadp(name):
            with open(ARTIFACTS / f"preds_{name}.jsonl", encoding="utf-8") as fh:
                return {json.loads(l)["case_id"]: json.loads(l) for l in fh if l.strip()}

        sys_p, sim_p = loadp("system"), loadp("simple")
        rand = [g for g in golden if g["slice"] == "random"]
        a = np.array([int(sys_p[g["case_id"]]["pred_intent"] == g["intent"]) for g in rand])
        b = np.array([int(sim_p[g["case_id"]]["pred_intent"] == g["intent"]) for g in rand])
        pval = paired_bootstrap_pvalue(a, b)
        results["_tests"] = {
            "system_vs_simple_intent_acc_random_p": pval,
            "system_minus_simple_acc": float(a.mean() - b.mean()),
        }
        print(f"\npaired bootstrap, intent acc on random slice: "
              f"system-simple = {a.mean()-b.mean():+.3f}  p={pval:.3f}")

    plot_variant_bars(results, FIGURES / "variant_compare.png")

    out_path = ARTIFACTS / "metrics.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(results), fh, indent=2)
    _print_table({k: v for k, v in results.items() if not k.startswith("_")})
    print(f"\nwrote {out_path}")
    return results


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(HEADLINE),
                    help="comma-separated subset of " + ",".join(sorted(VARIANTS)))
    ap.add_argument("--judge", default="checklist",
                    choices=["off", "checklist", "llm", "both"])
    ap.add_argument("--llm-silver", action="store_true")
    a = ap.parse_args()
    names = [v.strip() for v in a.variants.split(",") if v.strip()]
    run(names, judge=a.judge, llm_silver=a.llm_silver)


if __name__ == "__main__":
    main()
