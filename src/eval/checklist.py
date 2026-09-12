"""Deterministic stand-in for the LLM-as-judge rubric in judge.py.

Same four axes, same send gate. I wrote this because the 15-minute reproduce
path cannot depend on a Gemini key, and I would rather a grader rerun a picky
checklist than trust a cached LLM score they cannot inspect.

The honest caveat, which is also in the report: I wrote the drafter and I wrote
this checklist, so it is kinder to template replies than a stranger would be.
That is why data/golden/human_reply_ratings.jsonl exists.
"""

import re

from src.agent.draft import DEVICE_PATTERNS, HANDOFF, INTENT_LINKS
from src.agent.retrieve import URL_RE
from src.eval.judge import AXES
from src.eval.metrics import HANDOFF_WORDS, PII_ASK_RE, PROMISE_RE

FORMAL = re.compile(
    r"\b(dear valued|we regret to inform|pursuant to|please be advised|"
    r"do not hesitate to|at your earliest convenience)\b",
    re.I,
)
CONTRACTION = re.compile(r"\b(we're|we'd|you're|it's|that's|can't|won't|don't|isn't|i'll)\b", re.I)


def _devices(text):
    return {name for name, pat in DEVICE_PATTERNS if pat.search(text or "")}


def score(case, reply, precedents, pred_escalate=False, allowed_links=None):
    reply = (reply or "").strip()
    msg = case.get("customer_message") or ""
    rec = {a: 1 for a in AXES}

    if not reply:
        return {**{a: 0 for a in AXES}, "total": 0, "send": False, "note": "empty"}

    # --- grounded ---
    # Same contract as reply_metrics: a URL is invented if the brand registry
    # does not have it. Scoring only the top-3 neighbours punished the
    # playbook drafter for using the intent-default t.co (a real Hulu link)
    # and let copy-nearest off the hook for one-off t.co links.
    invented = False
    allowed = {HANDOFF}
    if allowed_links:
        allowed.update(u.rstrip(".,!?)") for u in allowed_links)
    else:
        allowed.update(INTENT_LINKS.values())
        for h in precedents or []:
            for u in URL_RE.findall(h.get("agent_reply") or ""):
                allowed.add(u.rstrip(".,!?)"))
    for u in URL_RE.findall(reply):
        if u.rstrip(".,!?)") not in allowed:
            invented = True
    if invented or PROMISE_RE.search(reply):
        rec["grounded"] = 0
    elif URL_RE.search(reply) or rec.get("grounded") == 1:
        # a real brand link, or a playbook with no extra facts, is a 2 unless
        # we already dinged it. Generic "we'll look into it" with no link is 1.
        rec["grounded"] = 2 if (URL_RE.search(reply) or "pass along" in reply.lower()) else 1

    # --- helpful ---
    cust_dev, reply_dev = _devices(msg), _devices(reply)
    mismatch = bool(reply_dev - cust_dev) and bool(cust_dev)
    constantish = "please reach out to us here" in reply.lower() and "closer look" in reply.lower()
    asks_already_given = bool(cust_dev and re.search(r"which device", reply, re.I))
    if mismatch or asks_already_given:
        rec["helpful"] = 0
    elif constantish:
        rec["helpful"] = 1 if pred_escalate else 0
    elif pred_escalate and (HANDOFF in reply or HANDOFF_WORDS.search(reply)):
        rec["helpful"] = 2
    elif cust_dev and (cust_dev & reply_dev):
        rec["helpful"] = 2
    elif re.search(r"\b(streaming rights|lineup|pass along|no ads plan|call/chat|call or chat)\b", reply, re.I):
        rec["helpful"] = 2
    elif "?" in reply and not constantish:
        rec["helpful"] = 2
    elif HANDOFF not in reply and len(reply.split()) <= 16:
        rec["helpful"] = 2  # short thanks / acknowledgement
    else:
        rec["helpful"] = 1

    # --- voice ---
    if FORMAL.search(reply) or len(reply) > 280:
        rec["voice"] = 0
    elif CONTRACTION.search(reply) and len(reply) < 240:
        rec["voice"] = 2
    else:
        rec["voice"] = 1

    # --- safe ---
    if PROMISE_RE.search(reply) or PII_ASK_RE.search(reply):
        rec["safe"] = 0
    elif re.search(r"\b(account number|password|card number|date of birth)\b", reply, re.I):
        rec["safe"] = 0
    elif pred_escalate and not (HANDOFF in reply or HANDOFF_WORDS.search(reply)):
        rec["safe"] = 1  # decided to escalate but the tweet still tries to solve it
    else:
        rec["safe"] = 2

    rec["total"] = sum(rec[a] for a in AXES)
    # Unedited-send is a high bar. A generic-but-safe reply is a 1 on helpful
    # and I would still want a human to look at it, so it does not pass the gate.
    rec["send"] = (
        rec["safe"] == 2
        and rec["grounded"] >= 1
        and rec["helpful"] == 2
        and rec["voice"] >= 1
    )
    rec["note"] = _note(rec, mismatch, invented)
    rec["judge"] = "checklist"
    return rec


def _note(rec, mismatch, invented):
    if invented:
        return "invented a url the brand history does not support"
    if mismatch:
        return "names a device the customer did not"
    if rec["helpful"] == 0:
        return "generic deflection or asked for info already given"
    if rec["safe"] < 2:
        return "safety/routing mismatch"
    if rec["send"]:
        return "fine"
    return "borderline"


def score_all(cases_by_id, predictions, allowed_links=None):
    out = []
    for p in predictions:
        case = cases_by_id[p["case_id"]]
        rec = score(
            case,
            p.get("reply", ""),
            p.get("retrieved") or [],
            p.get("pred_escalate"),
            allowed_links=allowed_links,
        )
        rec.update({"case_id": p["case_id"], "variant": p.get("variant", "")})
        out.append(rec)
    return out
