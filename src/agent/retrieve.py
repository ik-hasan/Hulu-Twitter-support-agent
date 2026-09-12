"""The brand's memory: find how @hulu_support has answered messages like this one.

TF-IDF over character and word n-grams with nearest-neighbour lookup. No dense
embeddings, on purpose - see reports/DECISIONS.md #7. Short support tweets are
dominated by product nouns, error codes and device names, which is exactly what
sparse lexical retrieval is good at, and it costs nothing to run and re-run.

Two things this module also produces, which matter more than the retrieval
itself:

* a `max_sim` score, which is how the agent knows it has no precedent and should
  escalate rather than improvise;
* a registry of the links Hulu actually sends, so a drafted reply can be checked
  for invented URLs instead of being trusted.

Leakage: every case in the golden set is excluded from the knowledge base, and so
is every other turn of the same conversation. Without the second part, a
golden-set follow-up could retrieve its own thread and the reply would be handed
to the model almost verbatim.
"""

import json
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src import config
from src.data.prepare import read_cases

URL_RE = re.compile(r"https?://\S+")
NO_PRECEDENT_SIM = 0.18  # tuned on the dev slice only; see DECISIONS.md #10


def thread_of(case_id):
    return case_id.split(":")[0]


def build_kb(exclude_case_ids=(), exclude_threads=(), max_cases=None):
    max_cases = max_cases or config.KB_MAX_CASES
    exclude_case_ids = set(exclude_case_ids)
    exclude_threads = set(exclude_threads)

    kb = []
    for c in read_cases():
        if c["case_id"] in exclude_case_ids or thread_of(c["case_id"]) in exclude_threads:
            continue
        reply = c["agent_reply_clean"]
        if len(reply.split()) < 4:
            continue
        kb.append(
            {
                "case_id": c["case_id"],
                "customer_message": c["customer_message"],
                "agent_reply": reply,
                "is_opener": c["is_opener"],
                "turns_after": c["turns_after"],
            }
        )
        if len(kb) >= max_cases:
            break

    with config.KB_PATH.open("w", encoding="utf-8") as fh:
        for row in kb:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return kb


def read_kb():
    with config.KB_PATH.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


class Retriever:
    def __init__(self, kb):
        self.kb = kb
        texts = [r["customer_message"] for r in kb]
        # Word grams catch the topic, char grams survive the typos and the
        # error codes ("drm-4-68abfb9b", "error97") that word grams throw away.
        self.word_vec = TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), min_df=2, stop_words="english"
        )
        self.char_vec = TfidfVectorizer(
            sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3
        )
        self.Xw = normalize(self.word_vec.fit_transform(texts))
        self.Xc = normalize(self.char_vec.fit_transform(texts))
        self.links = link_registry(kb)

    def _sim(self, query):
        qw = normalize(self.word_vec.transform([query]))
        qc = normalize(self.char_vec.transform([query]))
        # 0.7/0.3 in favour of word grams; char grams are a tie-breaker, not the
        # signal. Picked on the dev slice.
        return 0.7 * np.asarray((self.Xw @ qw.T).todense()).ravel() + 0.3 * np.asarray(
            (self.Xc @ qc.T).todense()
        ).ravel()

    def search(self, query, k=None):
        k = k or config.RETRIEVE_K
        sims = self._sim(query)
        order = np.argsort(sims)[::-1][:k]
        out = []
        for i in order:
            row = dict(self.kb[i])
            row["sim"] = float(sims[i])
            out.append(row)
        return out

    def max_sim(self, query):
        return float(self._sim(query).max())


def link_registry(kb, min_count=3):
    """Links @hulu_support actually sends, with how often and a sample context.

    Used two ways: handed to the drafter so it reuses a real URL, and used by the
    evaluator to detect invented ones. t.co links are opaque, so the "purpose" is
    inferred from the words Hulu puts around them.
    """
    counts = Counter()
    contexts = {}
    for row in kb:
        for url in URL_RE.findall(row["agent_reply"]):
            url = url.rstrip(".,!?)")
            counts[url] += 1
            contexts.setdefault(url, []).append(row["agent_reply"])
    out = {}
    for url, n in counts.most_common():
        if n < min_count:
            continue
        out[url] = {"count": n, "examples": contexts[url][:3]}
    return out


def describe_links(links, top=12):
    """Compact block for the prompt: the handful of links Hulu leans on."""
    lines = []
    for url, meta in list(links.items())[:top]:
        sample = re.sub(r"\s+", " ", meta["examples"][0])[:110]
        lines.append(f"{url}  (used {meta['count']}x, e.g. \"{sample}\")")
    return "\n".join(lines)


def load_default(exclude_golden=True):
    """Retriever with the golden set and its threads held out."""
    excl_ids, excl_threads = set(), set()
    if exclude_golden and config.GOLDEN_PATH.exists():
        with config.GOLDEN_PATH.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    cid = json.loads(line)["case_id"]
                    excl_ids.add(cid)
                    excl_threads.add(thread_of(cid))
    if not config.KB_PATH.exists():
        build_kb(excl_ids, excl_threads)
    kb = read_kb()
    leaked = [r for r in kb if thread_of(r["case_id"]) in excl_threads]
    if leaked:
        raise RuntimeError(
            f"{len(leaked)} knowledge-base rows come from golden-set threads; "
            "delete data/processed/*_kb.jsonl and rebuild"
        )
    return Retriever(kb)


if __name__ == "__main__":
    r = load_default()
    print(f"kb size: {len(r.kb)}")
    print(f"distinct links seen 3+ times: {len(r.links)}\n")
    print(describe_links(r.links, top=8))
    print("\n--- sample lookups ---")
    for q in [
        "my hulu keeps buffering on roku during the football game",
        "why did you charge me twice this month",
        "when is season 3 of rick and morty coming",
        "asdkjh qwe zzz",
    ]:
        print(f"\nQ: {q}   (max_sim={r.max_sim(q):.3f})")
        for hit in r.search(q, k=2):
            print(f"   [{hit['sim']:.3f}] {hit['customer_message'][:90]}")
            print(f"           -> {hit['agent_reply'][:110]}")
