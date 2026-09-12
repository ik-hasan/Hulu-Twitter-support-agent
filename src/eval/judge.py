"""LLM-as-judge for reply quality.

Design choices that matter for whether the scores mean anything:

* The judge is blind to which system produced the reply. Nothing in the prompt
  names a variant, and the replies are scored one at a time, so it cannot infer
  a ranking from the batch.
* The judge does NOT see the reply @hulu_support actually sent. If it did, it
  would collapse into a similarity metric and punish good replies that differ
  from the historical one. It sees the same retrieved precedents the drafter saw,
  which is what "grounded" has to be checked against.
* Four axes scored 0/1/2, plus a single would-you-send-this gate. The gate is the
  number I report, because an average of four rubric scores hides the difference
  between "bland" and "actively wrong".
* Temperature 0 and a committed response cache, so re-running the eval gives the
  same scores.

The honest caveat, repeated in the report: by default the judge is the same model
family that wrote the replies, which is a known self-preference bias. That is why
src/eval/judge_agreement.py exists and why --judge-model lets you swap it.
"""

import json

from src.llm import LLM

SYSTEM = (
    "You are a quality reviewer for a brand's Twitter support team. You review "
    "draft replies strictly and briefly. You care about factual grounding and "
    "safety more than politeness. Answer with JSON only."
)

PROMPT = """Review one draft reply for @hulu_support (Hulu's Twitter support account).

Customer's conversation so far:
{history}

The customer's latest message:
{message}

How this team has handled similar messages (treat this as the only evidence of
what Hulu actually offers, allows, and sounds like):
{precedents}

DRAFT REPLY TO REVIEW:
{reply}

Score each axis 0, 1 or 2.

grounded
  2 = every factual claim is supported by the precedents, or it makes no factual claims
  1 = broadly consistent but adds a detail the precedents do not support
  0 = states something the precedents contradict, or invents a policy, date, link or entitlement

helpful
  2 = answers the question, or gives the correct next step, or asks the one question actually needed
  1 = safe but generic; the customer is no further forward
  0 = irrelevant, ignores what they asked, or asks for something they already gave

voice
  2 = reads like the precedents: brief, warm, contractions, no corporate filler
  1 = serviceable but off-register (too formal, too jokey, too long)
  0 = would not pass as this brand

safe
  2 = no promise it cannot keep, no request for account or card details in public, and the
      handling matches the situation
  1 = borderline; an editor would want to change something
  0 = promises a refund/credit/date, asks for sensitive details publicly, or tries to
      resolve something that clearly needs account access

Then decide:
send = true only if a social-media lead would post this unedited, right now.

Return JSON:
{{"grounded": 0-2, "helpful": 0-2, "voice": 0-2, "safe": 0-2,
  "send": true|false, "note": "<20 words max, the single biggest problem or 'fine'>"}}"""

AXES = ["grounded", "helpful", "voice", "safe"]


class Judge:
    def __init__(self, llm=None, model=None):
        self.llm = llm or LLM(model=model)

    def score(self, case, reply, precedents):
        prompt = PROMPT.format(
            history=_history(case.get("history")),
            message=case["customer_message"],
            precedents=_precedents(precedents),
            reply=reply.strip() or "(empty reply)",
        )
        out = self.llm.complete_json(
            prompt, system=SYSTEM, max_tokens=300,
            default={a: 0 for a in AXES} | {"send": False, "note": "judge unparseable"},
        )
        rec = {}
        for a in AXES:
            try:
                rec[a] = max(0, min(2, int(round(float(out.get(a, 0))))))
            except (TypeError, ValueError):
                rec[a] = 0
        rec["total"] = sum(rec[a] for a in AXES)
        rec["send"] = bool(out.get("send"))
        rec["note"] = str(out.get("note", ""))[:160]
        return rec

    def score_all(self, cases_by_id, predictions, verbose=True):
        out = []
        for i, p in enumerate(predictions):
            case = cases_by_id[p["case_id"]]
            rec = self.score(case, p.get("reply", ""), p.get("retrieved") or [])
            rec.update({"case_id": p["case_id"], "variant": p.get("variant", "")})
            out.append(rec)
            if verbose and (i + 1) % 25 == 0:
                print(f"  judge {i + 1}/{len(predictions)}"
                      f"  cache_hits={self.llm.cache_hits} api_calls={self.llm.calls}")
        return out


def _history(history, limit=3):
    if not history:
        return "(none - this is the first message)"
    return "\n".join(
        f"{'Customer' if h['role'] == 'customer' else 'Hulu'}: {h['text']}"
        for h in history[-limit:]
    )


def _precedents(hits, k=3):
    if not hits:
        return "(no similar case was found - a grounded reply may not be possible)"
    return "\n".join(
        f'- customer: "{h["customer_message"][:180]}"\n  hulu: "{h["agent_reply"][:200]}"'
        for h in hits[:k]
    )


def summarise(scores):
    if not scores:
        return {}
    n = len(scores)
    out = {"n": n}
    for a in AXES + ["total"]:
        out[f"mean_{a}"] = sum(s[a] for s in scores) / n
    out["send_rate"] = sum(bool(s["send"]) for s in scores) / n
    return out


def write(scores, path):
    with open(path, "w", encoding="utf-8") as fh:
        for s in scores:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")


def read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]
