"""Brand-selection scan.

The thing that decides whether this assignment is even possible for a brand is
not volume, it is whether the brand ever answers in-channel. A brand whose every
reply is "please DM us" gives the drafter nothing to ground on and makes the
escalation decision trivially always-escalate. So I score candidates on
deflection rate and reply substance, not on tweet count.

Run: python -m src.data.explore
"""

import re
from collections import Counter

from src.data.prepare import load_threads, normalise_thread, to_cases

CANDIDATES = [
    "AmazonHelp", "AppleSupport", "Uber_Support", "SpotifyCares", "Delta",
    "AmericanAir", "TMobileHelp", "hulu_support", "XboxSupport", "AskPlayStation",
    "AskPayPal", "AirbnbHelp",
]

DEFLECT = re.compile(
    r"\b(dm|d\.m\.|direct message|private message|pm us|send us a message|"
    r"follow and dm|meet us in the dm)", re.I
)
LINKY = re.compile(r"https?://|<link>")
ASKS_INFO = re.compile(r"\?\s*$|\bcan you (tell|confirm|send|share)|\bwhich\b.*\?", re.I)
# Words that suggest an actual instruction rather than a holding pattern.
ACTIONABLE = re.compile(
    r"\b(try|tap|click|go to|settings|reinstall|log ?out|log ?in|restart|update|"
    r"toggle|clear|reset|check|swipe|select|enable|disable|uninstall)\b", re.I
)

SAMPLE_LIMIT = 4000  # cases per brand is plenty for these rates


def main():
    wanted = {c.lower(): c for c in CANDIDATES}
    buckets = {c: [] for c in CANDIDATES}

    threads, source = load_threads()
    for th in threads:
        name = wanted.get(str(th["brand"]).lower())
        if not name or len(buckets[name]) >= SAMPLE_LIMIT:
            continue
        cases = to_cases(normalise_thread(th))
        buckets[name].extend(cases[: SAMPLE_LIMIT - len(buckets[name])])
        if all(len(v) >= SAMPLE_LIMIT for v in buckets.values()):
            break

    print(f"source: {source}\n")
    header = f"{'brand':<16}{'cases':>7}{'deflect%':>10}{'action%':>9}{'asks%':>7}{'link%':>7}{'med_len':>9}{'openers%':>10}"
    print(header)
    print("-" * len(header))
    rows = []
    for name, cases in buckets.items():
        if not cases:
            continue
        replies = [c["agent_reply_clean"] for c in cases]
        n = len(replies)
        deflect = 100 * sum(bool(DEFLECT.search(r)) for r in replies) / n
        action = 100 * sum(bool(ACTIONABLE.search(r)) for r in replies) / n
        asks = 100 * sum(bool(ASKS_INFO.search(r)) for r in replies) / n
        link = 100 * sum(bool(LINKY.search(r)) for r in replies) / n
        lens = sorted(len(r.split()) for r in replies)
        med = lens[n // 2]
        openers = 100 * sum(c["is_opener"] for c in cases) / n
        rows.append((name, n, deflect, action, asks, link, med, openers))

    for r in sorted(rows, key=lambda x: x[3] - x[2], reverse=True):
        print(f"{r[0]:<16}{r[1]:>7}{r[2]:>10.1f}{r[3]:>9.1f}{r[4]:>7.1f}{r[5]:>7.1f}{r[6]:>9}{r[7]:>10.1f}")

    print("\nhigher action% and lower deflect% = more to ground a draft on")


if __name__ == "__main__":
    main()
