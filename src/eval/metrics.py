"""Automated metrics.

Three families:

intent      accuracy and macro-F1. Macro-F1 is the one to look at - accuracy is
            dominated by playback_error, which is a third of the queue.

routing     treated as a screening decision, not a balanced classification. A
            missed escalation (a billing dispute answered by a bot) and a
            needless escalation (a human handling "when is Rick and Morty back")
            cost very different amounts, so both are reported separately and
            combined in a cost-weighted figure whose weights are stated in the
            report rather than hidden here.

reply       cheap, checkable properties: does it fit in a tweet, did it invent a
            URL, did it promise something it cannot, did it ask for account
            details in public, and does it match the routing decision. Plus token
            overlap with the reply Hulu actually sent, which I treat as a weak
            similarity signal and not as correctness - there are many good
            replies to any tweet.

Confidence intervals are bootstrap percentile intervals over cases. With n=120 in
the unbiased slice they are wide, and the report says so instead of quoting point
estimates alone.
"""

import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from src.agent.retrieve import URL_RE
from src.taxonomy import LABELS

SEED = 17

# Costs used for the routing summary. Ratio matters, not the units: letting a
# money/lockout case through to a bot is treated as 5x worse than paying a human
# to answer a question a tweet could have handled. Justified in the report.
COST_MISSED_ESCALATION = 5.0
COST_NEEDLESS_ESCALATION = 1.0

PROMISE_RE = re.compile(
    r"\b(we(?:'ve| have)? (?:refunded|issued|credited)|you(?:'ll| will) (?:be )?"
    r"(?:refunded|receive a refund|get a credit)|refund(?:ed)? (?:has been|will be)|"
    r"(?:will be|arrives?|available) (?:on|by) (?:january|february|march|april|may|"
    r"june|july|august|september|october|november|december|\d{1,2}/\d{1,2})|"
    r"guarantee|we promise|definitely (?:will|be) (?:fixed|added|available))",
    re.I,
)
PII_ASK_RE = re.compile(
    r"\b(?:(?:send|share|reply with|provide|dm us|give us)[^.?!]{0,40}"
    r"(?:account number|acct #|password|card number|credit card|full name|"
    r"social security|date of birth))\b",
    re.I,
)
# @hulu_support's own phone/chat handoff link, seen 600+ times in the corpus.
HANDOFF_LINKS = {"https://t.co/6YdK7bvQN7", "https://t.co/LkhXOWWq74"}
HANDOFF_WORDS = re.compile(
    r"\b(call|phone|chat|reach out|contact us|get in touch|specialist|team)\b", re.I
)

TOKEN_RE = re.compile(r"[a-z0-9']+")
STOP = set(
    "a an the and or but if to of for on in at is are was were be been we you your "
    "our us it this that with as so we'd we're you're i i'm not no do does did can "
    "could will would have has had here there please thanks thank hi hey".split()
)


def _tok(text):
    return [t for t in TOKEN_RE.findall((text or "").lower()) if t not in STOP]


def token_f1(pred, ref):
    p, r = Counter(_tok(pred)), Counter(_tok(ref))
    if not p or not r:
        return 0.0
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# intent
# ---------------------------------------------------------------------------


def intent_metrics(gold, pred):
    gold, pred = list(gold), list(pred)
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0
    macro = f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)
    weighted = f1_score(gold, pred, labels=LABELS, average="weighted", zero_division=0)
    per = {}
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[label] = {"support": tp + fn, "precision": prec, "recall": rec, "f1": f1}
    return {"n": n, "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted,
            "per_class": per}


def confusion(gold, pred, top=None):
    m = defaultdict(Counter)
    for g, p in zip(gold, pred):
        m[g][p] += 1
    rows = []
    for g in LABELS:
        if g not in m:
            continue
        total = sum(m[g].values())
        worst = [(p, c) for p, c in m[g].most_common() if p != g]
        rows.append({"gold": g, "n": total, "correct": m[g][g],
                     "top_confusions": worst[: top or 3]})
    return rows


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


def routing_metrics(gold, pred):
    gold = [bool(g) for g in gold]
    pred = [bool(p) for p in pred]
    n = len(gold)
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    tn = sum((not g) and (not p) for g, p in zip(gold, pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    kappa = cohen_kappa_score(gold, pred) if len(set(gold)) > 1 and len(set(pred)) > 1 else 0.0
    cost = (COST_MISSED_ESCALATION * fn + COST_NEEDLESS_ESCALATION * fp) / n if n else 0.0
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "escalate_precision": prec,
        "escalate_recall": rec,
        "escalate_f1": f1,
        "kappa": kappa,
        "missed_escalations": fn,
        "needless_escalations": fp,
        "auto_rate": (tn + fn) / n if n else 0.0,
        "cost_per_case": cost,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def reason_metrics(records):
    """Reason accuracy, scored only where both sides agreed to escalate.

    This is the softest of the three numbers. Real messages carry several valid
    reasons at once ("charged twice AND I'll cancel"), so exact match understates
    agreement; the any-overlap figure is the fairer one.
    """
    both = [r for r in records if r["gold_escalate"] and r["pred_escalate"]]
    if not both:
        return {"n": 0, "exact": 0.0, "any_overlap": 0.0}
    exact = sum(r["gold_reason"] == r["pred_reason"] for r in both) / len(both)
    overlap = sum(
        r["gold_reason"] in set(r.get("pred_reasons_all") or [r["pred_reason"]])
        for r in both
    ) / len(both)
    by_gold = Counter(
        (r["gold_reason"], r["pred_reason"]) for r in both if r["gold_reason"] != r["pred_reason"]
    )
    return {"n": len(both), "exact": exact, "any_overlap": overlap,
            "top_mismatches": by_gold.most_common(5)}


# ---------------------------------------------------------------------------
# reply
# ---------------------------------------------------------------------------


def reply_metrics(records, allowed_links):
    allowed = {u.rstrip(".,!?)") for u in allowed_links}
    n = len(records)
    if not n:
        return {}
    invented, overlong, promises, pii, handoff_ok, overlaps, empties = 0, 0, 0, 0, 0, [], 0
    for r in records:
        reply = r.get("reply_raw") or r.get("reply") or ""
        if not reply.strip():
            empties += 1
        urls = [u.rstrip(".,!?)") for u in URL_RE.findall(reply)]
        if any(u not in allowed for u in urls):
            invented += 1
        if len(reply) > 280:
            overlong += 1
        if PROMISE_RE.search(reply):
            promises += 1
        if PII_ASK_RE.search(reply):
            pii += 1
        # Routing consistency: an escalated case should visibly hand off.
        if r.get("pred_escalate"):
            if any(u in HANDOFF_LINKS for u in urls) or HANDOFF_WORDS.search(reply):
                handoff_ok += 1
        overlaps.append(token_f1(reply, r.get("gold_reply", "")))
    n_esc = sum(1 for r in records if r.get("pred_escalate")) or 1
    return {
        "n": n,
        "invented_link_rate": invented / n,
        "over_280_rate": overlong / n,
        "unsafe_promise_rate": promises / n,
        "public_pii_request_rate": pii / n,
        "empty_rate": empties / n,
        "handoff_consistency": handoff_ok / n_esc,
        "token_f1_vs_actual": float(np.mean(overlaps)),
        "mean_chars": float(np.mean([len(r.get("reply") or "") for r in records])),
    }


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------


def bootstrap_ci(values, stat=np.mean, n_boot=4000, alpha=0.05, seed=SEED):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    draws = stat(v[idx], axis=1)
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def paired_bootstrap_pvalue(a, b, n_boot=4000, seed=SEED):
    """Two-sided paired bootstrap on the mean difference of per-case scores.

    Used to check whether "system beats baseline" survives n=120, which for
    several of my numbers it does not.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b) or not len(a):
        return 1.0
    d = a - b
    obs = d.mean()
    rng = np.random.default_rng(seed)
    centred = d - obs
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    draws = centred[idx].mean(axis=1)
    return float((np.abs(draws) >= abs(obs)).mean())
