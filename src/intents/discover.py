"""Bottom-up intent discovery.

The brief says the intent set has to come out of the data, so I did not want to
write down the eight labels I would have guessed. Instead: TF-IDF the customer
messages, over-cluster with k-means at a few values of k, and print the
distinctive terms plus real examples per cluster. I then read the output and
merged clusters by hand into the taxonomy in src/taxonomy.py.

This script is kept in the repo because the taxonomy is a judgement call and the
evidence behind it should be re-runnable, not just asserted.

Run: python -m src.intents.discover --k 16
"""

import argparse
import re
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import SEED
from src.data.prepare import read_cases

# Brand/product nouns dominate TF-IDF otherwise and every cluster looks the same.
EXTRA_STOP = [
    "hulu", "hulu_support", "support", "just", "like", "im", "ive", "dont", "doesnt",
    "cant", "got", "get", "know", "really", "guys", "pls", "please", "thanks", "hey",
    "hi", "yall", "does", "did", "day", "time", "watch", "watching", "link",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[14])
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--openers-only", action="store_true", default=True)
    a = ap.parse_args()

    cases = read_cases()
    if a.openers_only:
        cases = [c for c in cases if c["is_opener"]]
    texts = [c["customer_message"] for c in cases]
    print(f"clustering {len(texts)} opener messages\n")

    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words=list(TfidfVectorizer(stop_words="english").get_stop_words()) + EXTRA_STOP,
        token_pattern=r"[a-zA-Z][a-zA-Z']{2,}",
        ngram_range=(1, 2),
        min_df=8,
        max_df=0.4,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())

    for k in a.k:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X)
        print("=" * 78)
        print(f"k = {k}")
        print("=" * 78)
        order = Counter(labels).most_common()
        for cid, n in order:
            top = terms[np.argsort(km.cluster_centers_[cid])[::-1][:12]]
            print(f"\n[c{cid}] n={n} ({100 * n / len(texts):.1f}%)  {', '.join(top)}")
            idx = np.where(labels == cid)[0]
            # Pick the examples closest to the centroid - they read as prototypes.
            d = np.asarray(X[idx] @ km.cluster_centers_[cid]).ravel()
            for j in idx[np.argsort(d)[::-1][: a.examples]]:
                print("     .", re.sub(r"\s+", " ", texts[j])[:170])


if __name__ == "__main__":
    main()
