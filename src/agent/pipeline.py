"""Assemble classifier + router + drafter into a named system.

Each case goes through the same four steps so the variants are comparable:
retrieve -> classify -> route -> draft. Retrieval happens first because both the
router (precedent similarity) and the drafter (grounding) need it, and running it
once keeps the baselines and the LLM on identical inputs.
"""

import json
import time

from src.agent import classify as C
from src.agent import draft as D
from src.agent import escalate as E
from src.agent.retrieve import load_default
from src.llm import LLM

# The named systems the report compares. "trivial" and "simple" are the two
# baselines the brief asks for. "system" is the thing `make eval` actually
# reruns with no API key. The llm_* variants are the same pipeline with the
# optional LLM plugged in; they no-op without a key or a cache hit.
VARIANTS = {
    "trivial": {"clf": "majority", "route": "never", "draft": "constant"},
    "simple": {"clf": "rules", "route": "rules", "draft": "retrieval_copy"},
    "tfidf": {"clf": "tfidf_lr", "route": "rules", "draft": "retrieval_copy"},
    "system": {"clf": "stacked", "route": "thread", "draft": "grounded"},
    # ablations: which piece of "system" is doing the work
    "stacked_copy": {"clf": "stacked", "route": "thread", "draft": "retrieval_copy"},
    "rules_grounded": {"clf": "rules", "route": "rules", "draft": "grounded"},
    # optional LLM path - same four steps, different guts
    "system_llm": {"clf": "llm", "route": "llm", "draft": "llm_grounded"},
    "llm_clf_only": {"clf": "llm", "route": "rules", "draft": "retrieval_copy"},
    "rules_clf_llm_draft": {"clf": "rules", "route": "rules", "draft": "llm_grounded"},
    "system_no_ground": {"clf": "llm", "route": "llm", "draft": "llm_grounded",
                         "no_precedents": True},
}


class Agent:
    def __init__(self, variant="system", retriever=None, llm=None, silver=None,
                 fewshot=None):
        if variant not in VARIANTS:
            raise KeyError(f"unknown variant {variant!r}; have {sorted(VARIANTS)}")
        self.variant = variant
        cfg = VARIANTS[variant]
        self.cfg = cfg
        self.retriever = retriever or load_default()
        self.llm = llm or LLM()

        if cfg["clf"] == "majority":
            self.clf = C.MajorityClassifier()
        elif cfg["clf"] == "rules":
            self.clf = C.RuleClassifier()
        elif cfg["clf"] == "tfidf_lr":
            if not silver:
                raise ValueError("tfidf_lr needs silver labels; run `cli silver` first")
            self.clf = C.TfidfClassifier().fit(
                [s["customer_message"] for s in silver], [s["intent"] for s in silver]
            )
        elif cfg["clf"] == "stacked":
            self.clf = C.StackedClassifier(self.retriever)
        else:
            self.clf = C.LLMClassifier(llm=self.llm)
            if fewshot:
                self.clf.set_examples(fewshot)

        self.router = {
            "never": E.NeverEscalate,
            "rules": E.RuleEscalate,
            "thread": E.ThreadEscalate,
            "llm": E.LLMEscalate,
        }[cfg["route"]]()
        if cfg["route"] == "llm":
            self.router = E.LLMEscalate(llm=self.llm)

        if cfg["draft"] == "constant":
            self.drafter = D.ConstantDrafter()
        elif cfg["draft"] == "retrieval_copy":
            self.drafter = D.RetrievalDrafter(self.retriever)
        elif cfg["draft"] == "grounded":
            self.drafter = D.GroundedDrafter(self.retriever)
        else:
            self.drafter = D.LLMDrafter(self.retriever, llm=self.llm)

    def handle(self, case):
        t0 = time.time()
        hits = self.retriever.search(case["customer_message"])
        max_sim = hits[0]["sim"] if hits else 0.0

        pred = self.clf.predict(case)
        route = self.router.decide(
            case, pred["intent"], confidence=pred["confidence"],
            hits=hits, max_sim=max_sim,
        )
        draft_hits = [] if self.cfg.get("no_precedents") else hits
        drafted = self.drafter.draft(
            case, intent=pred["intent"], escalate=route["escalate"], hits=draft_hits
        )
        return {
            "case_id": case["case_id"],
            "variant": self.variant,
            "pred_intent": pred["intent"],
            "pred_confidence": pred["confidence"],
            "intent_why": pred.get("why", ""),
            "pred_escalate": route["escalate"],
            "pred_reason": route["reason"],
            "pred_reasons_all": route.get("reasons", []),
            "escalate_why": route.get("why", ""),
            "reply": drafted["reply"],
            "reply_source": drafted.get("source", ""),
            "link_repaired": drafted.get("link_repaired", False),
            "max_sim": max_sim,
            "retrieved": [
                {"case_id": h["case_id"], "sim": h["sim"],
                 "customer_message": h["customer_message"], "agent_reply": h["agent_reply"]}
                for h in hits
            ],
            "latency_s": round(time.time() - t0, 3),
        }

    def run(self, cases, verbose=True):
        out = []
        for i, c in enumerate(cases):
            out.append(self.handle(c))
            if verbose and (i + 1) % 25 == 0:
                print(f"  [{self.variant}] {i + 1}/{len(cases)}"
                      f"  cache_hits={self.llm.cache_hits} api_calls={self.llm.calls}")
        return out


def write_predictions(rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_predictions(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]
