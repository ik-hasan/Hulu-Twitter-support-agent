"""Intent classifiers, same predict() interface.

  MajorityClassifier  trivial baseline - always the most common class.
  RuleClassifier      simple baseline - regexes from src/agent/rules.py.
  TfidfClassifier     logistic regression over TF-IDF, trained on weak labels
                      from those same regexes (or LLM silver if you have a key).
                      If it cannot beat the regexes, n-gram smoothing is not
                      doing anything the rules did not already do.
  StackedClassifier   the headline classifier. Regexes where they are precise,
                      nearest-neighbour vote over the weakly-labelled knowledge
                      base everywhere else, and history-aware lookup on short
                      follow-ups ("Roku", "still happening").
  LLMClassifier       taxonomy in the prompt plus a handful of few-shot
                      examples. Optional; needs an API key or a cache hit.

Confidence is not calibrated. It is only used as a `low_confidence` floor.
"""

import json
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.agent.rules import MAJORITY_INTENT, rule_intent, rule_intent_all
from src.config import SEED
from src.llm import LLM
from src.taxonomy import ACCOUNT_BOUND, LABELS, prompt_taxonomy_block

PLAYBACK_CUES = re.compile(
    r"buffer|freez|\berror|skip|crash|lag|glitch|won'?t (load|play)|"
    r"keeps? (load|crash|freez|stopping)|not (working|loading|playing)|"
    r"restarted|rebooted|cutting out",
    re.I,
)
def _break_ties(fired, text, knn_intent):
    """Labelling leftover: live-tv regexes sit above playback, so a buffering
    football game becomes a zip-code question. If a playback cue is in the
    message, that wins. The simple baseline still first-matches; this is the
    stack's job."""
    if "live_tv_channels" in fired and "playback_error" in fired:
        return "playback_error" if PLAYBACK_CUES.search(text or "") else "live_tv_channels"
    if knn_intent in fired:
        return knn_intent
    return fired[0]

SYSTEM = (
    "You triage inbound customer messages for @hulu_support, the Twitter support "
    "account for Hulu. You only classify; you never write replies here. "
    "Answer with JSON only."
)

PROMPT = """Classify the LAST customer message into exactly one intent.

Intents:
{taxonomy}

Rules:
- Label the customer's underlying problem, not the action they threaten. "It keeps
  buffering so I'm cancelling" is playback_error, not subscription_change.
- If the message is only an acknowledgement or thanks, it is praise_or_chatter even
  when the thread was about a fault.
- unclear_request is for messages with no identifiable topic AND no symptom.
- Use the conversation so far to resolve short replies like "Roku" or "it worked".
{examples}
Conversation so far:
{history}

LAST customer message:
{message}

Return JSON: {{"intent": "<one label>", "confidence": <0.0-1.0>, "why": "<12 words max>"}}"""


def format_history(history, limit=4):
    if not history:
        return "(none - this is the first message)"
    lines = []
    for h in history[-limit:]:
        who = "Customer" if h["role"] == "customer" else "Hulu"
        lines.append(f"{who}: {h['text']}")
    return "\n".join(lines)


class MajorityClassifier:
    name = "majority"

    def __init__(self, label=MAJORITY_INTENT):
        self.label = label

    def predict(self, case):
        return {"intent": self.label, "confidence": 1.0, "why": "constant baseline"}


class RuleClassifier:
    name = "rules"

    def predict(self, case):
        return {
            "intent": rule_intent(case["customer_message"]),
            "confidence": 1.0,
            "why": "first matching regex",
        }


class TfidfClassifier:
    name = "tfidf_lr"

    def __init__(self):
        self.model = make_pipeline(
            TfidfVectorizer(
                sublinear_tf=True, ngram_range=(1, 2), min_df=2, stop_words="english"
            ),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                               random_state=SEED),
        )
        self.fitted = False

    def fit(self, texts, labels):
        self.model.fit(texts, labels)
        self.fitted = True
        return self

    def predict(self, case):
        if not self.fitted:
            raise RuntimeError("TfidfClassifier used before fit()")
        text = case["customer_message"]
        label = self.model.predict([text])[0]
        conf = float(self.model.predict_proba([text]).max())
        return {"intent": label, "confidence": conf, "why": "tfidf+lr on silver labels"}


class StackedClassifier:
    """Regex + retrieval vote.

    The regexes are high-precision on money and lockouts and I do not want a
    neighbour vote talking me out of those. They are also first-match, so a
    message that hits three intents gets whatever happened to be listed first
    in rules.py - that is where the knowledge-base vote earns its keep.

    Short follow-ups are the other gap. "Phone or laptop" is not an intent;
    it is an answer to a question Hulu already asked. Looking up the last
    couple of turns with the message finds the original problem.
    """

    name = "stacked"

    def __init__(self, retriever, weak=None):
        self.retriever = retriever
        self.weak = weak if weak is not None else {
            row["case_id"]: rule_intent(row["customer_message"]) for row in retriever.kb
        }

    def _query(self, case):
        bits = []
        for h in (case.get("history") or [])[-2:]:
            bits.append(h.get("text") or "")
        bits.append(case["customer_message"])
        return " ".join(bits)

    def _knn(self, case):
        hits = self.retriever.search(self._query(case))
        scores = Counter()
        for h in hits:
            lab = self.weak.get(h["case_id"]) or rule_intent(h["customer_message"])
            scores[lab] += max(h.get("sim") or 0.0, 0.01)
        if not scores:
            return MAJORITY_INTENT, 0.0
        label, sc = scores.most_common(1)[0]
        return label, sc / (sum(scores.values()) or 1.0)

    def predict(self, case):
        text = case["customer_message"]
        fired = rule_intent_all(text)
        knn_intent, knn_conf = self._knn(case)

        if len(fired) == 1 and fired[0] in ACCOUNT_BOUND:
            return {
                "intent": fired[0],
                "confidence": 0.92,
                "why": "account-bound regex, not to be talked out of",
            }
        if len(fired) == 1 and fired[0] == "live_tv_channels" and PLAYBACK_CUES.search(text):
            return {
                "intent": "playback_error",
                "confidence": 0.8,
                "why": "live-tv regex, but the complaint is a symptom",
            }
        if len(fired) == 1 and fired[0] == knn_intent:
            return {
                "intent": fired[0],
                "confidence": max(knn_conf, 0.75),
                "why": "regex and neighbours agree",
            }
        if len(fired) == 0:
            thanks = (
                len(text.split()) <= 12
                and bool(re.search(r"\b(thanks|thank you|ty|worked|back up)\b", text, re.I))
            )
            if thanks:
                return {
                    "intent": "praise_or_chatter",
                    "confidence": 0.8,
                    "why": "short thanks, regex silent",
                }
            if knn_conf < 0.4:
                return {
                    "intent": "unclear_request",
                    "confidence": knn_conf,
                    "why": "knn weak, regex silent",
                }
            return {
                "intent": knn_intent,
                "confidence": knn_conf,
                "why": "neighbours; regex silent",
            }
        if len(fired) >= 2:
            pick = _break_ties(fired, text, knn_intent)
            return {
                "intent": pick,
                "confidence": knn_conf if pick == knn_intent else 0.7,
                "why": "playback cue beat live-tv" if pick == "playback_error" and "live_tv_channels" in fired
                else ("knn broke a regex tie" if pick == knn_intent else "regex first-match, knn disagreed"),
            }
        # one regex, neighbours disagree
        short_followup = (not case.get("is_opener", True)) and len(text.split()) <= 10
        if short_followup:
            return {
                "intent": knn_intent,
                "confidence": knn_conf,
                "why": "short follow-up, neighbours+history over regex",
            }
        return {
            "intent": fired[0],
            "confidence": 0.65,
            "why": "regex, neighbours disagreed",
        }


class LLMClassifier:
    name = "llm"

    def __init__(self, llm=None, retriever=None, n_examples=6):
        self.llm = llm or LLM()
        self.retriever = retriever
        self.n_examples = n_examples
        self._examples = []

    def set_examples(self, examples):
        """Few-shot examples. These come from the dev slice, never the golden set."""
        self._examples = examples[: self.n_examples]

    def _examples_block(self):
        if not self._examples:
            return ""
        lines = ["\nLabelled examples from this brand's queue:"]
        for ex in self._examples:
            msg = ex["customer_message"].replace("\n", " ")[:160]
            lines.append(f'- "{msg}" -> {ex["intent"]}')
        return "\n".join(lines) + "\n"

    def predict(self, case):
        prompt = PROMPT.format(
            taxonomy=prompt_taxonomy_block(),
            examples=self._examples_block(),
            history=format_history(case.get("history")),
            message=case["customer_message"],
        )
        out = self.llm.complete_json(
            prompt, system=SYSTEM, max_tokens=200,
            default={"intent": MAJORITY_INTENT, "confidence": 0.0, "why": "unparseable"},
        )
        intent = str(out.get("intent", "")).strip()
        if intent not in LABELS:
            # Cheap repair: models occasionally return a near-miss like
            # "playback_errors". Anything else falls back and is logged as such.
            match = [l for l in LABELS if l.startswith(intent[:12])] if intent else []
            intent = match[0] if match else MAJORITY_INTENT
            out["why"] = f"off-taxonomy output repaired ({out.get('intent')!r})"
        try:
            conf = float(out.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        return {
            "intent": intent,
            "confidence": max(0.0, min(1.0, conf)),
            "why": str(out.get("why", ""))[:120],
        }


def silver_label(cases, llm=None, verbose=True):
    """Label a pool with the LLM classifier to train the TF-IDF baseline on."""
    clf = LLMClassifier(llm=llm)
    out = []
    for i, c in enumerate(cases):
        pred = clf.predict(c)
        out.append({"case_id": c["case_id"], "customer_message": c["customer_message"],
                    "intent": pred["intent"], "confidence": pred["confidence"]})
        if verbose and (i + 1) % 100 == 0:
            print(f"  silver {i + 1}/{len(cases)} (cache hits {clf.llm.cache_hits})")
    return out


def read_silver(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]
