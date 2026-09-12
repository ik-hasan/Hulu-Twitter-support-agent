# Golden set — how I sampled and labelled

220 examples, `data/golden/golden_set.jsonl`. Built from `to_label.jsonl` (the pool, no regex guesses shown) plus `labels_part{1-4}.tsv` (what I actually typed).

## Sampling

`python -m src.golden.sample`, seed 17, from the 8,169 Hulu cases.

- **random (120).** Uniform. This is the only slice whose class mix I treat as the live queue. Playback_error is 43/120 (36%), content_availability 21/120, then a long tail. Escalate rate 37/120 (31%).
- **stratified (80).** Round-robin top-up using the regexes in `src/agent/rules.py`, so billing / account / subscription / ads / live TV / device / praise / unclear aren't empty. Biased toward cases those regexes fire on. I do not quote this slice as an unbiased estimate.
- **hard (20).** Either ≥3 regexes fire, or it's a mid-thread turn in a conversation of 6+ turns. Deliberately adversarial.

I dropped exact-duplicate customer messages at prepare-time, and dropped messages shorter than 3 tokens. I did **not** drop non-English, screenshots-only, or concatenated-agent-text rows — those showed up, and I labelled them.

## Labelling

I used `scratch/show.py` to print a window of rows with two turns of history, and typed TSV lines:

```
row_index <tab> intent <tab> escalate(0|1) <tab> reason <tab> note
```

Four files because I did it in sittings, not because the scheme changed. `python -m src.golden.build` merges them and refuses unknown intents, escalate/reason mismatches, and missing rows.

Rules I held myself to, written down in `src/taxonomy.py` (`python -m src.taxonomy` prints the sheet):

- Label the customer's **underlying problem**, not the threat. "It keeps buffering so I'm cancelling" is `playback_error` + `churn_risk`, not `subscription_change`.
- A thanks, even in a thread about a fault, is `praise_or_chatter`.
- `unclear_request` only when there is no topic *and* no symptom.
- Escalate if finishing the job needs the subscriber record, money movement, identity, a language I can't answer in, or they've already done the standard steps. Vague "this sucks" is not churn; "cancelling as soon as this game is over" is.
- When two reasons apply, I picked the one in `REASON_PRIORITY` (safety > law > language > account > churn > exhausted). The note field records the other.

I did not relabel after seeing model errors, except for two regex patches (thanks-at-end-of-tweet; "restarted everything twice") that I applied to the *rules*, not to the gold.

## What I noticed while labelling, and left as gold

- Watch-progress / "up next" / continue-watching bugs have no intent. I put them in `playback_error` and wrote `TAXONOMY GAP` in the note.
- A few threads are Spotify or Verizon text concatenated onto a Hulu customer. I labelled the Hulu-relevant speech.
- One Spanish row (`unsupported_language`). `@hulu_support` sometimes answers in Spanish in the corpus; my agent does not.
- Sailor Moon under "LGBTQ Programming" — I marked `no_precedent` rather than auto-handle with a rights template. Wrong-tone here is worse than a queue delay.

## Human reply ratings

`data/golden/human_reply_ratings.jsonl` — 40 system drafts, mixed random/strat/hard, scored against the same 0–1–2 rubric as `src/eval/judge.py`, after the eval produced the drafts, without looking at the checklist scores. I updated three rows after the device/language/playback patches (LG now names LG; Spanish now queues; Xbox error97 uses the playback playbook). Send rate 12/40. Compared in `python -m src.cli agreement`.
