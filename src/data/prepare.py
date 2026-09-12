"""Turn the raw dump into per-brand threads, then into prediction units.

Two input shapes are supported:

* `data/raw/twcs.csv`   - the original Kaggle file (tweet-level, threads have to
  be stitched together through in_response_to_tweet_id).
* `data/raw/twcs_conversations.parquet` - an ungated HuggingFace mirror of the
  same corpus, already collapsed into conversations. This is the default because
  it needs no Kaggle credentials, which makes the repo reproducible for someone
  who just clones it. See reports/DECISIONS.md #2.

Both paths end up in the same schema so nothing downstream cares which was used.
"""

import argparse
import html
import json
import random
import re
from collections import Counter

import pandas as pd

from src import config
from src.config import BRAND, MAX_THREADS_PER_BRAND, SEED

# The mirror went through a lossy encode/decode round trip somewhere upstream, so
# curly quotes arrive as this soup. Left alone it leaks into prompts and into
# every string metric, e.g. "it�?Ts" instead of "it's".
MOJIBAKE = {
    "\ufffd?T": "'",
    "\ufffd?t": "'",
    "\ufffd?\u009d": '"',
    "\ufffd?\u009c": '"',
    "\ufffd?M": "'",
    "\ufffd?s": "'",
    "\ufffd?": "'",
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\u009d": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "Ã©": "e",
}

ANON_HANDLE = re.compile(r"@\d{4,}")
URL = re.compile(r"https?://\S+")
# Support staff sign their tweets: "... -becki", "... ^JK", "... ~AB".
SIGNATURE = re.compile(r"[\s]*[\^~\-–—]\s?[A-Za-z]{1,12}\s*$")
WS = re.compile(r"\s+")

ROLE_LINE = re.compile(r"^(Customer|Support):\s?(.*)$")


def clean_text(text, drop_urls=False):
    if not isinstance(text, str):
        return ""
    for bad, good in MOJIBAKE.items():
        text = text.replace(bad, good)
    text = html.unescape(text)
    text = ANON_HANDLE.sub("", text)
    if drop_urls:
        text = URL.sub("<link>", text)
    return WS.sub(" ", text).strip()


def strip_signature(text):
    """Remove the agent's initials from the end of a support reply.

    I strip these before any text-similarity scoring: "-becki" vs "^JK" is a
    rota artefact, not a difference in reply quality. The raw text is kept
    alongside so brand voice is still visible to the drafter.
    """
    prev = None
    while prev != text:
        prev = text
        text = SIGNATURE.sub("", text).strip()
    return text


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _threads_from_mirror(path):
    df = pd.read_parquet(path, columns=["conversation_id", "company", "conversation"])
    for row in df.itertuples(index=False):
        turns = []
        for line in str(row.conversation).split("\n"):
            m = ROLE_LINE.match(line)
            if m:
                turns.append({"role": m.group(1).lower(), "text": m.group(2)})
            elif turns:
                # continuation of a multi-line tweet
                turns[-1]["text"] += " " + line
        if turns:
            yield {"thread_id": row.conversation_id, "brand": row.company, "turns": turns}


def _threads_from_kaggle(path):
    df = pd.read_csv(path, dtype={"tweet_id": "int64"}, low_memory=False)
    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"], errors="coerce"
    )
    by_id = {int(r.tweet_id): r for r in df.itertuples(index=False)}
    children = {}
    for r in df.itertuples(index=False):
        parent = r.in_response_to_tweet_id
        if pd.notna(parent):
            children.setdefault(int(parent), []).append(int(r.tweet_id))

    roots = df[df["in_response_to_tweet_id"].isna()]["tweet_id"].astype(int).tolist()
    for root in roots:
        chain, node = [], root
        # Follow the single longest reply chain. Twitter threads do branch, but
        # in this corpus branching is rare and support answers linearly.
        while node is not None:
            chain.append(node)
            kids = children.get(node, [])
            node = min(kids) if kids else None
        if len(chain) < 2:
            continue
        turns, brand = [], None
        for tid in chain:
            row = by_id[tid]
            inbound = bool(row.inbound)
            if not inbound and brand is None:
                brand = str(row.author_id)
            turns.append(
                {
                    "role": "customer" if inbound else "support",
                    "text": str(row.text),
                    "created_at": str(row.created_at),
                }
            )
        if brand:
            yield {"thread_id": f"k{root}", "brand": brand, "turns": turns}


def load_threads():
    if config.RAW_KAGGLE_CSV.exists():
        return _threads_from_kaggle(config.RAW_KAGGLE_CSV), "kaggle:twcs.csv"
    if config.RAW_CONVERSATIONS.exists():
        return _threads_from_mirror(config.RAW_CONVERSATIONS), "hf-mirror:parquet"
    raise FileNotFoundError(
        "No raw data found. Run `python -m src.cli download` first."
    )


# --------------------------------------------------------------------------
# cleaning + prediction units
# --------------------------------------------------------------------------


def normalise_thread(thread):
    turns = []
    for t in thread["turns"]:
        text = clean_text(t["text"])
        if not text:
            continue
        role = t["role"]
        # Role labels are not trustworthy: brand broadcasts occasionally land in
        # the corpus tagged as Customer. An agent signature is a good tell.
        if role == "customer" and SIGNATURE.search(t["text"]) and len(text) > 40:
            role = "support"
        turns.append({"role": role, "text": text, "clean": strip_signature(text)})
    # Collapse consecutive same-role turns - people tweet in bursts.
    merged = []
    for t in turns:
        if merged and merged[-1]["role"] == t["role"]:
            merged[-1]["text"] += " " + t["text"]
            merged[-1]["clean"] += " " + t["clean"]
        else:
            merged.append(dict(t))
    return {**thread, "turns": merged}


def to_cases(thread):
    """A case = one customer message the brand actually answered.

    Keeping mid-thread turns (not just openers) makes this closer to a real
    triage queue, where plenty of inbound messages are follow-ups on an open
    issue. `turn_index` is carried through so I can slice on it later.
    """
    turns = thread["turns"]
    cases = []
    for i, t in enumerate(turns):
        if t["role"] != "customer":
            continue
        if i + 1 >= len(turns) or turns[i + 1]["role"] != "support":
            continue
        history = [
            {"role": h["role"], "text": h["text"]} for h in turns[max(0, i - 4) : i]
        ]
        cases.append(
            {
                "case_id": f"{thread['thread_id']}:{i}",
                "brand": thread["brand"],
                "turn_index": i,
                "is_opener": i == 0,
                "history": history,
                "customer_message": t["text"],
                "agent_reply": turns[i + 1]["text"],
                "agent_reply_clean": turns[i + 1]["clean"],
                "n_turns_total": len(turns),
                "turns_after": len(turns) - (i + 1),
            }
        )
    return cases


def brand_counts(limit=None):
    threads, source = load_threads()
    counts = Counter()
    for n, th in enumerate(threads):
        counts[th["brand"]] += 1
        if limit and n >= limit:
            break
    return counts, source


def build(brand=BRAND, max_threads=MAX_THREADS_PER_BRAND):
    threads, source = load_threads()
    target = brand.lower()
    kept = []
    for th in threads:
        if str(th["brand"]).lower() != target:
            continue
        th = normalise_thread(th)
        if len(th["turns"]) < 2:
            continue
        kept.append(th)

    rng = random.Random(SEED)
    rng.shuffle(kept)
    if len(kept) > max_threads:
        kept = kept[:max_threads]

    cases = []
    for th in kept:
        cases.extend(to_cases(th))

    # Drop degenerate cases: one-word messages carry no intent signal and would
    # silently inflate every metric.
    cases = [c for c in cases if len(c["customer_message"].split()) >= 3]

    # Exact-duplicate customer messages are common (bots, copy-paste rants).
    seen, deduped = set(), []
    for c in cases:
        k = c["customer_message"].lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(c)

    config.THREADS.parent.mkdir(parents=True, exist_ok=True)
    with config.THREADS.open("w", encoding="utf-8") as fh:
        for c in deduped:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"source          : {source}")
    print(f"brand           : {brand}")
    print(f"threads kept    : {len(kept)}")
    print(f"cases written   : {len(deduped)}  -> {config.THREADS}")
    print(f"openers         : {sum(c['is_opener'] for c in deduped)}")
    return deduped


def read_cases(path=None):
    path = path or config.THREADS
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default=BRAND)
    ap.add_argument("--max-threads", type=int, default=MAX_THREADS_PER_BRAND)
    ap.add_argument("--counts", action="store_true", help="just show brand volumes")
    a = ap.parse_args()
    if a.counts:
        counts, src = brand_counts()
        print(f"source: {src}\n")
        for name, n in counts.most_common(40):
            print(f"{n:7d}  {name}")
    else:
        build(a.brand, a.max_threads)
