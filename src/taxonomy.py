"""The intent set for @hulu_support, plus the policy attached to each intent.

Where this came from: `python -m src.intents.discover` over-clusters the customer
messages at k=16 and k=24; I read the term lists and prototype messages and
merged clusters by hand. The raw cluster dumps are in reports/cluster_dump_*.txt.

Merges I made and why (also in reports/DECISIONS.md):
  * "buffering", "error code", "isn't working", "is Hulu down" were four separate
    clusters. They get one intent, because Hulu answers all of them with the same
    first-line troubleshooting playbook.
  * "missing episode", "when is season N up", "please add <title>" were three
    clusters. One intent: every answer is a content-rights statement.
  * I kept ads_complaint apart from subscription_change even though they both
    touch the No-Ads plan, because "there are too many drug ads" is feedback and
    "I pay for No-Ads and still see ads" is a billing-grade complaint. Collapsing
    them hid a real difference in what the agent should do.
  * praise_or_chatter and unclear_request are not support intents, but they are
    ~8% of the inbound queue and each has its own correct action, so they are
    labels rather than a dropped "other".

`needs_account_access` is the field that does most of the escalation work: it
marks intents the bot structurally cannot finish, because resolving them requires
looking at a subscriber record it has no access to.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Intent:
    name: str
    definition: str
    cues: list = field(default_factory=list)
    playbook: str = ""
    needs_account_access: bool = False
    default_action: str = "auto"


INTENTS = [
    Intent(
        name="playback_error",
        definition=(
            "Something will not play or plays badly: buffering, freezing, blank or "
            "black screen, a named error code, app crashing mid-stream, or a broad "
            "'Hulu is not working / is Hulu down' report."
        ),
        cues=["buffering", "error code", "won't load", "keeps crashing", "is hulu down"],
        playbook=(
            "Acknowledge, ask for the device and the specific title if not given, and "
            "point at the troubleshooting guide. Do not promise a fix or a credit."
        ),
        default_action="auto",
    ),
    Intent(
        name="content_availability",
        definition=(
            "Asking where a title, season or episode is, when it will arrive, why it "
            "disappeared, or requesting that Hulu add something to the catalogue."
        ),
        cues=["when will", "why isn't ... on hulu", "missing episode", "please add"],
        playbook=(
            "Answer from streaming rights: what is and is not licensed, and on which "
            "plan. If it is a request, say the interest will be passed on. Never "
            "invent a release date."
        ),
        default_action="auto",
    ),
    Intent(
        name="live_tv_channels",
        definition=(
            "Hulu + Live TV specifically: whether a channel or a live sports event is "
            "carried, regional availability and blackouts, the channel lineup, or DVR "
            "and recording behaviour."
        ),
        cues=["live tv", "is the game on", "channel lineup", "in my area", "dvr"],
        playbook=(
            "State carriage and regional limits, link the lineup or the live guide. "
            "Regional rights vary, so avoid absolute claims about a viewer's market."
        ),
        default_action="auto",
    ),
    Intent(
        name="billing_charge",
        definition=(
            "Money already moved or is about to: an unexpected, duplicated or "
            "pro-rated charge, the wrong price, a refund request, a failed payment, or "
            "updating a payment method."
        ),
        cues=["charged twice", "refund", "why am i paying", "pro-rated", "wrong price"],
        playbook=(
            "Explain the general billing mechanic if one clearly applies, then hand "
            "off. Never confirm, promise or deny a refund in this channel."
        ),
        needs_account_access=True,
        default_action="escalate",
    ),
    Intent(
        name="account_access",
        definition=(
            "Locked out or on the wrong account: cannot log in, password or email "
            "problems, being asked to start a trial despite paying, home-location and "
            "device-limit lockouts, suspected unauthorised access."
        ),
        cues=["can't log in", "password", "not at home error", "someone else's account"],
        playbook=(
            "Offer the generic self-service reset path, then hand off, because "
            "anything account-specific needs identity verification."
        ),
        needs_account_access=True,
        default_action="escalate",
    ),
    Intent(
        name="subscription_change",
        definition=(
            "Wanting to start, stop, pause, upgrade or downgrade a plan, including "
            "cancellation threats and 'I'm done with you' churn messages."
        ),
        cues=["cancel", "unsubscribe", "switch plan", "downgrade", "how do i pause"],
        playbook=(
            "Point at the plan-management page for the mechanics, acknowledge the "
            "reason, and hand off rather than trying to save the account by tweet."
        ),
        needs_account_access=True,
        default_action="escalate",
    ),
    Intent(
        name="ads_complaint",
        definition=(
            "Complaints about advertising itself: volume, repetition, irrelevant or "
            "distressing ad content, ad targeting, or ads appearing on a plan the "
            "viewer believes should not have them."
        ),
        cues=["too many commercials", "paying for no ads", "same ad over and over"],
        playbook=(
            "Explain the difference between the ad-supported and No-Ads plans and "
            "which content still carries ads. Pass on feedback about ad content. If "
            "they say they pay for No-Ads and still see them, that is a billing-grade "
            "claim and goes to a human."
        ),
        default_action="auto",
    ),
    Intent(
        name="device_or_feature_feedback",
        definition=(
            "The product, not a fault: device or platform support questions, "
            "complaints about the interface or a redesign, and feature requests such "
            "as 5.1 audio, 60fps, profiles or captions."
        ),
        cues=["new interface", "not supported on", "please add 5.1", "feature request"],
        playbook=(
            "Say plainly whether the thing is supported, thank them and commit only to "
            "passing the feedback on. Do not hint at a roadmap."
        ),
        default_action="auto",
    ),
    Intent(
        name="unclear_request",
        definition=(
            "Reaches out but gives nothing to act on: 'help me', 'DM me', a bare link "
            "or screenshot, or undirected venting with no identifiable problem."
        ),
        cues=["can you dm me", "help please", "<bare link>", "wtf hulu"],
        playbook="Ask one specific, cheap-to-answer clarifying question. Do not guess the issue.",
        default_action="auto",
    ),
    Intent(
        name="praise_or_chatter",
        definition=(
            "Not a support request: compliments, thanks, jokes, show talk, or replies "
            "addressed at other users that happen to mention the brand."
        ),
        cues=["love hulu", "thanks!", "best show ever"],
        playbook="Reply briefly and warmly. Do not open a ticket or ask for details.",
        default_action="auto",
    ),
]

BY_NAME = {i.name: i for i in INTENTS}
LABELS = [i.name for i in INTENTS]
ACCOUNT_BOUND = {i.name for i in INTENTS if i.needs_account_access}

# ---------------------------------------------------------------------------
# Escalation reasons. Kept as an explicit vocabulary so that a reason string is
# never free-text prose - the brief asks for a stated reason, and a closed set
# means I can actually measure which reason fires and whether it was right.
# ---------------------------------------------------------------------------

ESCALATION_REASONS = {
    "account_bound": "Resolving this needs access to the subscriber's account or money movement.",
    "self_service_exhausted": "The customer says they already tried the standard steps or could not reach support.",
    "legal_or_regulatory": "Mentions legal action, chargeback, data privacy, discrimination or accessibility law.",
    "vulnerable_or_safety": "Distress, harassment or a safety concern that a human should read.",
    "churn_risk": "A concrete cancellation threat where retention should own the reply.",
    "no_precedent": "No sufficiently similar case in the brand's history, so a grounded reply is not possible.",
    "low_confidence": "The classifier itself is unsure which intent this is.",
    # Added while hand-labelling, not designed up front: a non-trivial number of
    # inbound messages are Spanish or Portuguese and @hulu_support answers in
    # English. Auto-replying in the wrong language is worse than queueing.
    "unsupported_language": "The message is not in English and needs an agent who can answer in it.",
    "none": "Handled automatically.",
}


def prompt_taxonomy_block():
    """The taxonomy as the classifier prompt sees it."""
    lines = []
    for i in INTENTS:
        lines.append(f"- {i.name}: {i.definition}")
    return "\n".join(lines)


def labelling_guide():
    """Longer form, used as my own reference while hand-labelling the golden set."""
    out = []
    for i in INTENTS:
        out.append(f"## {i.name}")
        out.append(i.definition)
        out.append(f"cues: {', '.join(i.cues)}")
        out.append(f"needs account access: {i.needs_account_access}")
        out.append(f"default: {i.default_action}")
        out.append(f"playbook: {i.playbook}")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print(labelling_guide())
