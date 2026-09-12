"""Keyword rules.

These wear two hats, which is worth flagging up front:

1. They are the "simple baseline" the brief asks for. Anyone building this for
   real would write these first, and an LLM that cannot beat them is not earning
   its latency or its per-call cost.
2. They provide the provisional strata used to sample the golden set, so that
   billing/account/ads cases are not missed by a pure random draw.

Using them for both jobs biases the stratified part of the golden set towards
cases these rules fire on, which flatters them. That is why every headline
number is also reported on the random-only slice. See reports/REPORT.md.

Patterns are lists of alternatives rather than one re.VERBOSE blob. I wrote them
as a VERBOSE blob first and every multi-word pattern silently stopped matching,
because VERBOSE strips literal spaces - "black screen" had become "blackscreen".
Lists are harder to get wrong and diff better.
"""

import re

from src.taxonomy import LABELS

# Order matters: first intent whose list matches wins. Intents that are expensive
# to get wrong (money, lockouts) sit above the chatty ones deliberately.
INTENT_RULES = [
    ("billing_charge", [
        r"refund", r"charg(ed|e|ing|es)\b", r"double.?(bill|charg)", r"overcharg",
        r"bill(ing|ed)?\b", r"invoice", r"pro.?rat", r"my card", r"credit card",
        r"debit", r"payment (method|fail|declin)", r"why (am i|are you) (paying|charging)",
        r"\$\d", r"price (went|increase|chang)", r"free trial.{0,30}charg",
        r"charg.{0,30}free trial", r"money back", r"receipt",
    ]),
    ("account_access", [
        r"can.?t (log|sign) ?in", r"log ?in (issue|problem|error|trouble)",
        r"logged out", r"password", r"reset my (account|password)", r"wrong account",
        r"(my|the) account (is )?(lock|suspend|disabl|hack)", r"hacked",
        r"someone (else )?(is )?using my", r"verif(y|ication) (code|email)",
        r"not at home", r"home (location|network)", r"location change",
        r"device limit", r"already (have|had) an? (paying )?account",
        r"asking me to (sign up|start a trial)", r"(sign|log) me out",
    ]),
    ("subscription_change", [
        r"cancel(l?(ed|ing|ation))?\b", r"unsubscrib", r"un.?subscribe", r"downgrad",
        r"upgrad(e|ing) (my )?plan", r"switch (my )?plan", r"chang(e|ing) (my )?plan",
        r"pause my", r"end my subscription", r"cut the cord", r"resubscrib",
        r"re.?activate my", r"(i.?m|im) (done|leaving|out)\b",
    ]),
    ("ads_complaint", [
        r"(too many|so many|all these|all those|these) (ad|advert|commercial)",
        r"(ad|advert|commercial)s? (are|is) (too|so|really|way)",
        r"no.?ads? (plan|tier|subscription|version)",
        r"(pay|paying|paid) for no.?ads", r"same (ad|commercial) (over|again)",
        r"drug ad", r"ad (break|targeting)", r"commercials?\b", r"advertisements?\b",
    ]),
    ("live_tv_channels", [
        r"live ?tv", r"hulu live", r"live stream(ing)?", r"channel (lineup|list|guide)",
        r"(add|carry|have|get) .{0,20}(channel|network)\b", r"blackout",
        r"(is|will|when).{0,30}(game|match|race|fight|series|fight) (on|be on|air)",
        r"local channel", r"\bdvr\b", r"record(ing)? (the|my|it)", r"cloud dvr",
        r"in my (area|market|region|state)", r"fox sports", r"espn", r"\bnfl\b",
        r"\bnba\b", r"world series",
    ]),
    ("playback_error", [
        r"buffer(ing)?", r"freez(e|ing|es)", r"black screen", r"blank screen",
        r"won.?t (load|play|start)", r"keeps? (load|buffer|crash|freez|stopping)",
        r"error( code| message)?", r"\berr\d", r"\bp-?dev\d", r"\bdrm-?\d",
        r"\b\d{3} error", r"loading (error|issue|forever)",
        r"(not|isn.?t|ain.?t) (working|loading|playing)", r"crash(es|ing|ed)?\b",
        r"is (hulu|it|the app) down", r"(app|hulu) (is )?down\b", r"lag(gy|ging)",
        r"stuck (on|at)", r"glitch", r"pixelat", r"out of sync",
        r"audio (is )?(off|out|delay)", r"no sound", r"keeps kicking me",
    ]),
    ("content_availability", [
        r"when (will|is|are|does|do).{0,40}(be )?(on|available|added|coming|air|up)\b",
        r"why (isn.?t|is|aren.?t|are|no).{0,40}(on hulu|available|gone|removed)",
        r"(missing|where.?s|where is|where are).{0,30}(episode|season|show|movie|series)",
        r"(add|bring|put).{0,30}(back|on hulu|to hulu)",
        r"(season|episode|ep\.?) ?\d+", r"\bs\d+ ?e\d+", r"new (episode|season)s?",
        r"rights? (expire|expired)", r"(take|took|taken) (it |them )?(off|down)",
        r"licens(e|ing)", r"full seasons?", r"all seasons?",
    ]),
    ("device_or_feature_feedback", [
        r"new (interface|layout|design|ui|app|update)",
        r"old (interface|layout|app|version)",
        r"(interface|ui|layout|redesign) (is|sucks|stinks|sux)",
        r"(not|isn.?t) (supported|compatible)",
        r"support(ed)? on (my )?(roku|apple ?tv|fire|xbox|ps\d|switch|chromecast|vizio|samsung|lg)",
        r"(please )?(add|support) (5\.1|dolby|surround|4k|hdr|60 ?fps|profiles?|captions?|subtitles?)",
        r"closed caption", r"subtitle", r"feature request", r"bring back the old",
    ]),
    ("unclear_request", [
        r"^\W*(help|help me|help please|halp)\W*$",
        r"(can|could|will) you (please )?dm me", r"\bdm me\b",
        r"^\s*@?\w*\s*https?://\S+\s*$", r"^\W*(hello|hi|hey|yo)\W*$",
        r"^\W*\?+\W*$", r"what.?s going on\W*$", r"(fix|sort) (it|this|your shit)\W*$",
    ]),
    ("praise_or_chatter", [
        r"(love|loving|adore|thank you|thanks|ty) (you|hulu|so much|for)",
        r"(you|hulu|y.?all) (are|is) (the best|awesome|great|amazing)",
        r"best (show|streaming)", r"binge", r"obsessed", r"appreciate (it|you)",
        r"^\W*(lol|lmao|haha)",
        r"^\W*(thanks|thank you|ty)\b",
        r"\b(thanks|thank you|ty)[!.,]*\s*$",
        r"(it )?(totally )?worked",
        r"(it'?s|its) back( up)?", r"back up and running",
    ]),
]

COMPILED = [
    (name, re.compile("|".join(f"(?:{p})" for p in pats), re.I))
    for name, pats in INTENT_RULES
]

# Escalation overrides. Deliberately about the situation rather than the intent,
# because the riskiest messages cut across every intent.
ESCALATION_PATTERNS = {
    "legal_or_regulatory": [
        r"\bsue\b", r"suing", r"lawsuit", r"lawyer", r"attorney", r"legal action",
        r"small claims", r"charge ?back", r"dispute (the|this|my) charge",
        r"\bbbb\b", r"better business bureau", r"fraud", r"unauthoriz",
        r"identity theft", r"\bgdpr\b", r"\bccpa\b", r"my data", r"privacy",
        r"\bada\b", r"americans with disabilities", r"discriminat", r"class action",
        r"report(ing)? (you|this) to",
    ],
    "self_service_exhausted": [
        r"(already|i.?ve) (tried|done|did|rebooted|restarted|reinstalled) (all|everything|that|this|those|the steps)?",
        r"(tried|did) (that|everything|all of that)( already| too)?",
        r"(rebooted|restarted|reinstalled|power.?cycl).{0,40}(twice|again|everything|multiple|already)",
        r"(still|nothing) (not working|doesn.?t work|didn.?t work|hasn.?t worked)",
        r"(called|chatted|contacted|emailed|dm.?d) (you|hulu|support).{0,40}(times|already|weeks|days|months)",
        r"(no|nobody|no one) (has )?(response|responded|replied|answer)",
        r"(your|the) (support|chat|phone)( chat| line)? (is |just )?(doesn.?t|won.?t|never|useless|broken|endless)",
        r"third time", r"\d+(st|nd|rd|th) time", r"for weeks", r"for months",
        r"still (having|getting|seeing) (the same|this|issues)",
    ],
    "vulnerable_or_safety": [
        r"harass", r"threat(en|ening)?\b", r"abus(e|ive)", r"stalk",
        r"(my|a) (child|kid|son|daughter).{0,30}(saw|watch)", r"seizure", r"epilep",
        r"suicid", r"self.?harm", r"passed away", r"funeral", r"in the hospital",
        r"\bdeaf\b", r"\bblind\b", r"hard of hearing", r"disabilit", r"accessib",
    ],
    "churn_risk": [
        r"(am|i.?m|im|going to|gonna|will|about to) (cancel|unsubscrib|switch(ing)? to|leav)",
        r"cancel(l)?ing (my|our|this)", r"(done|finished|through) with (hulu|you|this)",
        r"(switch|switching|moving) (to|back to) (netflix|disney|youtube|sling|prime)",
        r"take my money elsewhere", r"last (chance|straw)",
    ],
}

COMPILED_ESC = {
    k: re.compile("|".join(f"(?:{p})" for p in pats), re.I)
    for k, pats in ESCALATION_PATTERNS.items()
}

# Measured on the silver pool rather than guessed - see src/eval/run_eval.py,
# which re-derives it and warns if this constant has drifted.
MAJORITY_INTENT = "content_availability"


def rule_intent(text, fallback=MAJORITY_INTENT):
    for name, pat in COMPILED:
        if pat.search(text or ""):
            return name
    return fallback


def rule_intent_all(text):
    """Every rule that fires. Used to find genuinely ambiguous cases to label."""
    return [name for name, pat in COMPILED if pat.search(text or "")]


def escalation_signals(text):
    return [k for k, pat in COMPILED_ESC.items() if pat.search(text or "")]


assert {n for n, _ in INTENT_RULES} <= set(LABELS), "rule names must match the taxonomy"
