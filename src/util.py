"""Small shared helpers."""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def pmap(fn, items, workers=6, label="", every=25):
    """Ordered parallel map with a progress line.

    Modest parallelism because the free tiers of every provider here rate-limit
    hard; src.llm retries 429s with backoff, so 6 workers keeps the pipe full
    without spending the whole run in retry. workers=1 gives plain sequential.
    """
    items = list(items)
    n = len(items)
    if workers <= 1:
        out = []
        for i, it in enumerate(items):
            out.append(fn(it))
            _tick(i + 1, n, label, every)
        return out

    results = [None] * n
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fmap = {pool.submit(fn, it): i for i, it in enumerate(items)}
        for fut in as_completed(fmap):
            i = fmap[fut]
            results[i] = fut.result()
            done += 1
            _tick(done, n, label, every)
    return results


def _tick(done, n, label, every):
    if done % every == 0 or done == n:
        sys.stdout.write(f"\r  {label} {done}/{n}   ")
        sys.stdout.flush()
        if done == n:
            sys.stdout.write("\n")


def fmt_pct(x):
    return f"{100 * x:.1f}%"
