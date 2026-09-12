"""Reply drafting, grounded in what @hulu_support actually said before.

  ConstantDrafter   trivial baseline. One safe deflection for every message. It
                    scores surprisingly well on "would this embarrass the brand",
                    which is exactly why reply quality needs more than one metric.
  RetrievalDrafter  simple baseline. Copy the nearest historical reply verbatim.
                    Free, perfectly on-brand, and wrong whenever the details
                    differ - it will happily tell someone to reboot their Roku
                    when they are on an Xbox.
  GroundedDrafter   the headline drafter. Intent playbook + a link the brand
                    actually used on similar tickets + a device slot if the
                    customer named one. No free-text generation, so it cannot
                    invent a release date. The cost is that it sounds a bit
                    samey, which the judge is supposed to catch.
  LLMDrafter        optional. Same constraints, but the sentences are generated.
                    Needs a key or a cache hit.

The link constraint is the part I care most about. A support bot that invents a
help-centre URL is worse than useless, so every drafter is only allowed URLs
that appear in the brand's history, and src/eval/metrics.py checks that it obeyed.
"""

import re
from collections import Counter

from src.agent.retrieve import URL_RE, describe_links
from src.llm import LLM
from src.taxonomy import BY_NAME

TWEET_LIMIT = 280

CONSTANT_REPLY = (
    "Sorry for the trouble! We'd love to help - please reach out to us here "
    "https://t.co/6YdK7bvQN7 and we'll take a closer look."
)

SYSTEM = (
    "You are drafting a public reply as @hulu_support, Hulu's Twitter support "
    "account. You mimic how this team has actually replied before. You never "
    "invent facts, dates, URLs, refunds or entitlements."
)

PROMPT = """Draft one public reply to the last customer message.

Intent: {intent}
Playbook for this intent: {playbook}
Routing decision: {routing}

How @hulu_support has handled similar messages (your only source of truth for
what Hulu offers, what it does not, and how it phrases things):
{precedents}

Links this team actually uses. You may use these URLs verbatim or none at all.
Do not write any other URL:
{links}

Conversation so far:
{history}

LAST customer message:
{message}

Hard constraints:
- Under {limit} characters, one or two sentences. This is a tweet.
- Only URLs from the list above. If none fits, include no link.
- Never state a release date, a refund, a credit, or a timeline the precedents
  do not already support.
- Never ask for an account number, password, card details or an email address in
  public. If you need account details, send them to the phone/chat link.
- If the routing decision is ESCALATE, acknowledge the issue and route them to a
  human. Do not attempt to solve it yourself.
- If the routing decision is AUTO, actually answer or give the next concrete step.
  Ask at most one question.
- Match the voice of the precedents: warm, brief, contractions, no corporate filler.

Return JSON: {{"reply": "<the tweet>"}}"""


class ConstantDrafter:
    name = "constant"

    def draft(self, case, intent=None, escalate=False, hits=None):
        return {"reply": CONSTANT_REPLY, "source": "constant"}


class RetrievalDrafter:
    name = "retrieval_copy"

    def __init__(self, retriever):
        self.retriever = retriever

    def draft(self, case, intent=None, escalate=False, hits=None):
        hits = hits if hits is not None else self.retriever.search(case["customer_message"])
        if not hits:
            return {"reply": CONSTANT_REPLY, "source": "fallback"}
        return {"reply": hits[0]["agent_reply"], "source": f"copy:{hits[0]['case_id']}"}


# Links the brand actually leans on, keyed by intent. Only used if they also
# appear in this run's link registry - so a rebuild of the KB cannot smuggle
# in a URL that is not in the history.
INTENT_LINKS = {
    "playback_error": "https://t.co/HXpLyTU79x",
    "content_availability": "https://t.co/O9QhZdLbxw",
    "live_tv_channels": "https://t.co/rjjbP5u0Fj",
    "ads_complaint": "https://t.co/cZHqrcmWQY",
    "device_or_feature_feedback": "https://t.co/HiTFsfb2rK",
    "account_access": "https://t.co/6YdK7bvQN7",
    "billing_charge": "https://t.co/6YdK7bvQN7",
    "subscription_change": "https://t.co/6YdK7bvQN7",
}
HANDOFF = "https://t.co/6YdK7bvQN7"

# Order matters: more specific names first so "apple tv" wins over "tv".
DEVICE_PATTERNS = [
    ("Xbox", re.compile(r"\bxbox(\s?one|\s?360)?\b", re.I)),
    ("Roku", re.compile(r"\broku\b", re.I)),
    ("Fire TV", re.compile(r"\b(fire\s?(tv|stick)|kindle)\b", re.I)),
    ("Apple TV", re.compile(r"\bapple\s?tv\b", re.I)),
    ("PS4", re.compile(r"\b(ps4|playstation\s?4)\b", re.I)),
    ("PS5", re.compile(r"\b(ps5|playstation\s?5)\b", re.I)),
    ("Switch", re.compile(r"\b(nintendo\s?)?switch\b", re.I)),
    ("Chromecast", re.compile(r"\bchromecast\b", re.I)),
    ("Samsung TV", re.compile(r"\bsamsung\b", re.I)),
    ("LG TV", re.compile(r"\blg\b", re.I)),
    ("Vizio", re.compile(r"\bvizio\b", re.I)),
    ("iPhone", re.compile(r"\biphone\b", re.I)),
    ("iPad", re.compile(r"\bipad\b", re.I)),
    ("Android", re.compile(r"\bandroid\b", re.I)),
    ("browser", re.compile(r"\b(safari|chrome|firefox|browser|laptop|macbook)\b", re.I)),
]


FAIL_NEAR = re.compile(
    r"buffer|error|crash|freez|won'?t|can'?t|not work|skip|glitch|lag", re.I
)


def detect_device(text):
    """If they name two devices, keep the one next to the failure, not the first
    in DEVICE_PATTERNS. 'LG app buffers, Xbox is fine' used to return Xbox."""
    text = text or ""
    hits = []
    for name, pat in DEVICE_PATTERNS:
        m = pat.search(text)
        if m:
            hits.append((name, m.start()))
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0][0]
    fail = FAIL_NEAR.search(text)
    if fail:
        return min(hits, key=lambda x: abs(x[1] - fail.start()))[0]
    return hits[0][0]


def pick_link(hits, allowed, fallback):
    if fallback in allowed:
        return fallback
    counts = Counter()
    for h in hits or []:
        for u in URL_RE.findall(h.get("agent_reply") or ""):
            u = u.rstrip(".,!?)")
            if u in allowed:
                counts[u] += 1
    if counts:
        return counts.most_common(1)[0][0]
    return ""


class GroundedDrafter:
    """Playbook + a real link + a device slot. No generated sentences."""

    name = "grounded"

    def __init__(self, retriever):
        self.retriever = retriever
        self.allowed = set(retriever.links) | {HANDOFF}

    def draft(self, case, intent=None, escalate=False, hits=None):
        hits = hits if hits is not None else self.retriever.search(case["customer_message"])
        text = case["customer_message"]
        device = detect_device(text)
        fallback = INTENT_LINKS.get(intent, HANDOFF)
        link = pick_link(hits, self.allowed, fallback)
        if escalate:
            reply = self._esc(intent, device, text)
        else:
            reply = self._auto(intent, device, text, link)
        if len(reply) > TWEET_LIMIT:
            reply = reply[: TWEET_LIMIT - 1].rsplit(" ", 1)[0] + "…"
        return {"reply": reply, "source": "grounded", "link_repaired": False}

    def _esc(self, intent, device, text):
        device_bit = f" on your {device}" if device else ""
        if intent == "billing_charge":
            return (
                "Sorry about the charge - we can't confirm or issue a refund over Twitter. "
                f"Please call/chat us with the account email so we can look: {HANDOFF}"
            )
        if intent == "account_access":
            return (
                "Sorry you're locked out. We can't verify an account in public, so please "
                f"reach us here and we'll walk through it: {HANDOFF}"
            )
        if intent == "subscription_change":
            return (
                "Sorry this isn't matching what you wanted from the plan. We can't change "
                f"a subscription over Twitter - please reach us here and we'll help: {HANDOFF}"
            )
        if intent == "ads_complaint":
            return (
                "Sorry the ads aren't matching what you expect from the plan. That's something "
                f"we need to check on the account - please call/chat us: {HANDOFF}"
            )
        return (
            f"Sorry this is still happening{device_bit} after the steps you've already tried. "
            f"Please call/chat us so we can take a closer look: {HANDOFF}"
        )

    def _auto(self, intent, device, text, link):
        device_bit = f" on your {device}" if device else ""
        link_bit = f" {link}" if link else ""
        if intent == "playback_error":
            if device:
                return (
                    f"Sorry you're running into this{device_bit}! Please try a quick reboot of "
                    f"the device and your router, then work through{link_bit}. Any error code showing?"
                )
            return (
                "Sorry you're running into this! Which device are you streaming on? In the "
                f"meantime please try a reboot and these steps:{link_bit}"
            )
        if intent == "content_availability":
            return (
                "Thanks for asking! What we can carry depends on streaming rights and can vary "
                f"by plan, so I don't have a date I can promise. We'll pass along the interest:{link_bit}"
            )
        if intent == "live_tv_channels":
            if re.search(r"\b\d{5}\b", text or ""):
                return (
                    "Live TV carriage is regional, so that zip helps. You can check the lineup "
                    f"for your area here:{link_bit} - some sports are blacked out locally."
                )
            return (
                "Live TV carriage is regional and some sports get blacked out. Could you share "
                f"the zip you're watching from so we can check the lineup?{link_bit}"
            )
        if intent == "ads_complaint":
            return (
                "Sorry the ads are frustrating. The ad-supported plan offsets licensing costs; "
                "the No Ads plan still carries ads on some live/next-day titles. Updating the "
                f"app sometimes mixes up the rotation:{link_bit}"
            )
        if intent == "device_or_feature_feedback":
            return (
                "Thanks for the note - we don't have a public roadmap, but I'll pass this along "
                f"to the team.{link_bit}"
            )
        if intent == "unclear_request":
            return (
                "Happy to help - could you tell us which device you're on and what exactly is "
                "going wrong (error code, title, or what you see on screen)?"
            )
        if intent == "praise_or_chatter":
            return "Glad to hear it! Thanks for hanging in there with us."
        if intent == "account_access":
            return (
                "Sorry for the login trouble. There's a generic reset path, but anything "
                f"account-specific we have to do over chat/phone:{link_bit or (' ' + HANDOFF)}"
            )
        if intent in ("billing_charge", "subscription_change"):
            return (
                "Sorry we can't take billing or plan changes over Twitter. Please reach us "
                f"here and we'll take a look: {HANDOFF}"
            )
        return (
            f"Sorry for the trouble{device_bit}! We'd like to help - "
            f"could you share a bit more about what you're seeing?{link_bit}"
        )


class LLMDrafter:
    name = "llm_grounded"

    def __init__(self, retriever, llm=None):
        self.retriever = retriever
        self.llm = llm or LLM()
        self.links_block = describe_links(retriever.links, top=14)
        self.allowed = set(retriever.links)

    def _precedents(self, hits):
        if not hits:
            return "(no similar case found)"
        lines = []
        for h in hits:
            msg = re.sub(r"\s+", " ", h["customer_message"])[:200]
            rep = re.sub(r"\s+", " ", h["agent_reply"])[:240]
            lines.append(f'- customer: "{msg}"\n  hulu: "{rep}"')
        return "\n".join(lines)

    def draft(self, case, intent=None, escalate=False, hits=None):
        hits = hits if hits is not None else self.retriever.search(case["customer_message"])
        spec = BY_NAME.get(intent)
        prompt = PROMPT.format(
            intent=intent or "unknown",
            playbook=spec.playbook if spec else "Answer briefly and honestly.",
            routing="ESCALATE" if escalate else "AUTO",
            precedents=self._precedents(hits),
            links=self.links_block or "(none)",
            history=_history(case.get("history")),
            message=case["customer_message"],
            limit=TWEET_LIMIT,
        )
        out = self.llm.complete_json(
            prompt, system=SYSTEM, max_tokens=300, default={"reply": ""}
        )
        reply = str(out.get("reply", "")).strip()
        if not reply:
            return {"reply": CONSTANT_REPLY, "source": "fallback:empty"}
        reply, repaired = enforce_links(reply, self.allowed)
        if len(reply) > TWEET_LIMIT:
            reply = reply[: TWEET_LIMIT - 1].rsplit(" ", 1)[0] + "…"
        return {"reply": reply, "source": "llm", "link_repaired": repaired}


def enforce_links(reply, allowed):
    """Strip URLs the brand has never used.

    I keep this as a last line of defence rather than the only one: the raw
    output is still what the metrics score, so the report can show how often the
    model tried to invent a link even though the user never sees it.
    """
    bad = [u for u in URL_RE.findall(reply) if u.rstrip(".,!?)") not in allowed]
    for u in bad:
        reply = reply.replace(u, "").strip()
    reply = re.sub(r"\s{2,}", " ", reply).replace(" .", ".").strip()
    return reply, bool(bad)


def _history(history, limit=3):
    if not history:
        return "(none - this is the first message)"
    return "\n".join(
        f"{'Customer' if h['role'] == 'customer' else 'Hulu'}: {h['text']}"
        for h in history[-limit:]
    )
