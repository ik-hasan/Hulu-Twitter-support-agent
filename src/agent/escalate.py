"""Auto-handle or hand to a human, with a reason from a closed vocabulary.

The reason is not decoration. "Escalate" with no reason is unauditable, and a
free-text reason cannot be measured, so every decision carries one of the keys in
src/taxonomy.ESCALATION_REASONS and the eval scores which one fired.

Three policies:

  NeverEscalate   trivial baseline. Worth reporting because ~69% of the random
                  slice is auto-handleable, so "never escalate" already looks
                  decent on plain accuracy. That is the headline-number trap.
  RuleEscalate    simple baseline, and genuinely hard to beat. Intent-driven
                  capability check, plus situational regexes, plus the
                  retrieval-confidence floor.
RuleEscalate is the headline router. LLMEscalate is the same policy as a
prompt, used only on the optional LLM path.
"""

import re

from src.agent.rules import escalation_signals
from src.agent.retrieve import NO_PRECEDENT_SIM
from src.llm import LLM
from src.taxonomy import ACCOUNT_BOUND, ESCALATION_REASONS

# Playbook gap: ads_complaint is not account-bound in the taxonomy, but
# "I pay for No-Ads and still see ads" is a billing-grade claim. Caught
# this while labelling rows 112/192/197 and promoted it to a rule so the
# simple baseline gets it too.
NO_ADS_CLAIM = re.compile(
    r"(pay|paying|paid|purchased|have|got|on).{0,28}"
    r"(no-?ads?|no commercials|ad-?free|without ads)",
    re.I,
)

CONF_FLOOR = 0.45

# Order decides which reason is reported when several apply. Safety and law come
# first because they change who must read the message; capability
# (account_bound) beats churn because it is a hard blocker rather than a
# preference.
REASON_PRIORITY = [
    "vulnerable_or_safety",
    "legal_or_regulatory",
    "unsupported_language",
    "account_bound",
    "churn_risk",
    "self_service_exhausted",
    "no_precedent",
    "low_confidence",
]

LATIN_HINTS = (
    " que ", " por ", " para ", " não ", " nao ", " está ", " estoy ", " puedo ",
    " mi cuenta", " não consigo", " pero ", " porque ", " gracias", " obrigado",
    "¿", "¡", " você ", " estão ",
)
# One of these is enough. The Spanish golden-set row only had one "function"
# word from the list above and slipped through.
STRONG_NON_EN = (
    " debería ", " mientras ", " anuncios ", " comerciales", " no consigo ",
    " mi celular", " por favor ", " ajuda ", " ajuda-me ", " não consigo",
)


def looks_non_english(text):
    """Very rough language screen.

    A proper detector is a dependency I did not want for a handful of cases, and
    a wrong-language auto-reply is bad enough that a blunt, recall-leaning check
    beats no check. It fires on ~1% of the pool.
    """
    raw = text or ""
    t = f" {raw.lower()} "
    if any(h in t for h in STRONG_NON_EN):
        return True
    if sum(h in t for h in LATIN_HINTS) >= 2:
        return True
    letters = [c for c in raw if c.isalpha()]
    if len(letters) >= 20:
        odd = sum(1 for c in letters if ord(c) > 127)
        if odd / len(letters) >= 0.08:
            return True
    return False


def pick_reason(reasons):
    for r in REASON_PRIORITY:
        if r in reasons:
            return r
    return "none"


class NeverEscalate:
    name = "never"

    def decide(self, case, intent, confidence=1.0, hits=None, max_sim=1.0):
        return {"escalate": False, "reason": "none", "reasons": []}


class RuleEscalate:
    name = "rules"

    def decide(self, case, intent, confidence=1.0, hits=None, max_sim=1.0):
        text = case["customer_message"]
        reasons = list(escalation_signals(text))
        if intent in ACCOUNT_BOUND:
            reasons.append("account_bound")
        if NO_ADS_CLAIM.search(text or ""):
            reasons.append("account_bound")
        if looks_non_english(text):
            reasons.append("unsupported_language")
        if max_sim < NO_PRECEDENT_SIM:
            reasons.append("no_precedent")
        if confidence < CONF_FLOOR:
            reasons.append("low_confidence")
        reasons = sorted(set(reasons))
        return {
            "escalate": bool(reasons),
            "reason": pick_reason(reasons),
            "reasons": reasons,
        }


class ThreadEscalate(RuleEscalate):
    """Same rules, run on the last tweet plus the previous customer turn.

    While labelling I kept seeing 'still happening' as the latest message
    with 'I already rebooted everything' sitting one turn back. The simple
    baseline is not allowed to look there; this one is. That is the whole
    difference, on purpose.
    """

    name = "thread"

    def decide(self, case, intent, confidence=1.0, hits=None, max_sim=1.0):
        bits = [case.get("customer_message") or ""]
        for h in reversed(case.get("history") or []):
            if h.get("role") == "customer" and h.get("text"):
                bits.append(h["text"])
                break
        glued = dict(case)
        glued["customer_message"] = " ".join(bits)
        return super().decide(glued, intent, confidence, hits, max_sim)


SYSTEM = (
    "You are the routing layer for @hulu_support. You decide whether a public "
    "Twitter reply can resolve a customer's message, or whether it must go to a "
    "human agent. You are cautious with money, accounts, safety and law, and you "
    "do not escalate things a tweet can genuinely answer. Answer with JSON only."
)

PROMPT = """Decide: can a public tweet resolve this, or does it need a human?

Escalate if any of these hold:
- account_bound: resolving it needs this subscriber's account or moves money
  (charges, refunds, plan changes, identity-verified lockouts).
- self_service_exhausted: they say they already tried the standard steps, or
  already contacted support, or the problem has persisted for days.
- legal_or_regulatory: legal threats, chargebacks, fraud, privacy, regulation,
  discrimination, or an accessibility-law complaint.
- vulnerable_or_safety: distress, harassment, a child exposed to unsuitable
  content, or an accessibility need being denied.
- churn_risk: a concrete cancellation threat, not vague grumbling about value.
- no_precedent: nothing similar in the brand's history, so any reply would be
  improvised.
- unsupported_language: the message is not in English.

Auto-handle otherwise. In particular these are normally auto:
- explaining what is and is not in the catalogue, or on which plan
- first-line troubleshooting when nothing has been tried yet
- channel carriage and regional availability questions
- feature requests and interface complaints
- asking one clarifying question when the message is vague
- thanks, praise and chit-chat

Classified intent: {intent} (classifier confidence {confidence:.2f})
Closest precedent similarity: {max_sim:.2f}

Precedents:
{precedents}

Conversation so far:
{history}

LAST customer message:
{message}

Return JSON:
{{"escalate": true|false, "reason": "<one of: {reasons}>", "why": "<15 words max>"}}"""


class LLMEscalate:
    name = "llm"

    def __init__(self, llm=None):
        self.llm = llm or LLM()

    def decide(self, case, intent, confidence=1.0, hits=None, max_sim=1.0):
        precedents = "(none)"
        if hits:
            precedents = "\n".join(
                f'- "{h["customer_message"][:140]}" -> "{h["agent_reply"][:160]}"'
                for h in hits[:3]
            )
        prompt = PROMPT.format(
            intent=intent,
            confidence=confidence,
            max_sim=max_sim,
            precedents=precedents,
            history=_history(case.get("history")),
            message=case["customer_message"],
            reasons=", ".join(k for k in ESCALATION_REASONS if k != "none"),
        )
        out = self.llm.complete_json(
            prompt, system=SYSTEM, max_tokens=200,
            default={"escalate": True, "reason": "low_confidence", "why": "unparseable"},
        )
        escalate = bool(out.get("escalate"))
        reason = str(out.get("reason", "")).strip()
        if escalate and reason not in ESCALATION_REASONS:
            reason = "low_confidence"
        if not escalate:
            reason = "none"
        return {
            "escalate": escalate,
            "reason": reason,
            "reasons": [reason] if escalate else [],
            "why": str(out.get("why", ""))[:120],
        }


def _history(history, limit=3):
    if not history:
        return "(none - this is the first message)"
    return "\n".join(
        f"{'Customer' if h['role'] == 'customer' else 'Hulu'}: {h['text']}"
        for h in history[-limit:]
    )
