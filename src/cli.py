"""One entry point so the README can stay short.

    python -m src.cli download
    python -m src.cli prepare
    python -m src.cli explore
    python -m src.cli discover --k 16 24
    python -m src.cli sample-golden
    python -m src.cli build-golden
    python -m src.cli eval
    python -m src.cli handle "my hulu keeps buffering on roku"
"""

import argparse
import json
import sys

from src import config


def cmd_download(_):
    from src.data.download import main as m
    m()


def cmd_prepare(a):
    from src.data.prepare import build
    build(a.brand, a.max_threads)


def cmd_explore(_):
    from src.data.explore import main as m
    m()


def cmd_discover(a):
    # argparse inside discover.py reads sys.argv; rebuild it
    sys.argv = ["discover"]
    for k in a.k:
        sys.argv += ["--k", str(k)]
    from src.intents.discover import main as m
    m()


def cmd_sample_golden(a):
    from src.golden.sample import build
    build(a.out)


def cmd_build_golden(_):
    from src.golden.build import main as m
    m()


def cmd_eval(a):
    from src.eval.run_eval import run
    names = [v.strip() for v in a.variants.split(",") if v.strip()]
    run(names, judge=a.judge, llm_silver=a.llm_silver)


def cmd_agreement(a):
    from src.eval import agreement
    sys.argv = ["agreement"]
    if a.init_sheet:
        sys.argv.append("--init-sheet")
    agreement.main()


def cmd_handle(a):
    from src.agent.pipeline import Agent
    from src.agent.retrieve import load_default

    case = {
        "case_id": "demo:0",
        "customer_message": a.message,
        "history": [],
        "is_opener": True,
    }
    retriever = load_default(exclude_golden=True)
    agent = Agent(variant=a.variant, retriever=retriever)
    out = agent.handle(case)
    payload = {
        "variant": out["variant"],
        "intent": out["pred_intent"],
        "confidence": out["pred_confidence"],
        "intent_why": out["intent_why"],
        "escalate": out["pred_escalate"],
        "reason": out["pred_reason"],
        "reason_text": _reason_text(out["pred_reason"]),
        "reply": out["reply"],
        "retrieved": [
            {"sim": round(h["sim"], 3), "customer": h["customer_message"][:120],
             "reply": h["agent_reply"][:120]}
            for h in out["retrieved"][:3]
        ],
    }
    # Neighbours sometimes have emoji; Windows cp1252 consoles choke on those.
    try:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except UnicodeEncodeError:
        print(json.dumps(payload, indent=2, ensure_ascii=True))


def _reason_text(key):
    from src.taxonomy import ESCALATION_REASONS
    return ESCALATION_REASONS.get(key, key)


def main():
    p = argparse.ArgumentParser(prog="python -m src.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="fetch the corpus (optional; processed files are committed)")

    pr = sub.add_parser("prepare", help="brand filter + thread -> cases")
    pr.add_argument("--brand", default=config.BRAND)
    pr.add_argument("--max-threads", type=int, default=config.MAX_THREADS_PER_BRAND)

    sub.add_parser("explore", help="brand-selection scan")

    di = sub.add_parser("discover", help="over-cluster customer messages")
    di.add_argument("--k", type=int, nargs="+", default=[16, 24])

    sg = sub.add_parser("sample-golden")
    sg.add_argument("--out", default=None)

    sub.add_parser("build-golden", help="merge labels_part*.tsv into golden_set.jsonl")

    ev = sub.add_parser("eval", help="run variants on the golden set (the 15-minute path)")
    ev.add_argument("--variants", default="trivial,simple,tfidf,system,stacked_copy,rules_grounded")
    ev.add_argument("--judge", default="checklist", choices=["off", "checklist", "llm", "both"])
    ev.add_argument("--llm-silver", action="store_true")

    ag = sub.add_parser("agreement", help="human vs judge kappa")
    ag.add_argument("--init-sheet", action="store_true")

    hd = sub.add_parser("handle", help="run one message through the agent")
    hd.add_argument("message")
    hd.add_argument("--variant", default="system")

    a = p.parse_args()
    {
        "download": cmd_download,
        "prepare": cmd_prepare,
        "explore": cmd_explore,
        "discover": cmd_discover,
        "sample-golden": cmd_sample_golden,
        "build-golden": cmd_build_golden,
        "eval": cmd_eval,
        "agreement": cmd_agreement,
        "handle": cmd_handle,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
