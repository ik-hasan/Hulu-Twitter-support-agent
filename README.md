# Hulu Twitter support agent

Take-home for Hiver. I picked `@hulu_support` out of the Kaggle Customer Support on Twitter dump, built a 10-intent triage agent that drafts a reply and decides whether to auto-handle or escalate, and spent most of the time trying to prove whether any of that is trustworthy.

The short version: **on a 120-example unbiased slice I labelled myself, the system gets intent right 53% of the time (macro-F1 0.50) and catches 51% of the cases a human should see.** That is better than "always say content_availability" (18%) and a regex baseline (46%). The 7-point intent gain is just on the significant side (p = 0.047) and I would not bet a launch on it. The number that *looks* great — a 98% "would you send this tweet" rate — is an artefact of a checklist judge I wrote. On the same 40 replies I would actually post 12 (30%). Cohen's κ on that gate is 0.02.

If you only read one file, read [`reports/REPORT.md`](reports/REPORT.md). Decisions are in [`reports/DECISIONS.md`](reports/DECISIONS.md).

## Reproduce the headline numbers (< 15 min)

Processed data and the golden set are in the repo. You do **not** need the 3M-tweet dump, a Kaggle account, or an API key.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# mac/linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.cli eval
```

You should get a table ending in something like:

```
variant         slice       n   int-acc   macroF1   esc-rec  esc-prec   miss  needless    send  invent
system          random    120     53.3%     49.6%     51.4%     79.2%     18         5   97.5%    0.0%
simple          random    120     45.8%     44.5%     48.6%     81.8%     19         4   35.8%   10.8%
trivial         random    120     17.5%      3.0%      0.0%      0.0%     37         0    0.0%    0.0%
```

Exact figures live in `artifacts/metrics.json`. `python -m src.cli agreement` reprints the human-vs-checklist kappa.

Try one message:

```bash
python -m src.cli handle "my hulu keeps buffering on roku during the game"
```

### Optional: LLM path

The same four steps (retrieve → classify → route → draft) can be swapped for Gemini/OpenAI/Groq/Anthropic. Copy `.env.example` to `.env`, add a key, then:

```bash
python -m src.cli eval --variants system_llm --judge llm
```

I did **not** use this for the numbers in the report. A grader with no key should still be able to rerun everything I claim.

### Optional: rebuild from the raw dump

```bash
python -m src.cli download    # HF mirror, ~207 MB, no Kaggle creds
python -m src.cli prepare     # 6k Hulu threads → cases
python -m src.cli explore     # why this brand
python -m src.cli discover --k 16 24
```

## What the agent does

For each inbound customer message (including mid-thread follow-ups, not just openers):

1. **Retrieve** the nearest historical `@hulu_support` replies (TF-IDF word + char n-grams).
2. **Classify** into one of 10 intents I derived from clustering the openers, then merging by how Hulu actually answers.
3. **Route** auto vs escalate, with a reason from a closed list (`account_bound`, `self_service_exhausted`, `churn_risk`, …).
4. **Draft** a tweet. The headline drafter fills an intent playbook and a URL the brand has actually used. It is not allowed to invent links, dates, or refunds.

"Good" for this brand means: don't tell an Xbox user to reboot a Roku, don't promise a refund on Twitter, don't invent a help-centre URL, and don't try to finish anything that needs the subscriber record. It does **not** mean sounding indistinguishable from Becki-on-the-Saturday-shift. I measured that separately and lost.

## Golden set

220 hand-labelled examples in `data/golden/golden_set.jsonl`.

| slice | n | purpose |
|---|---|---|
| random | 120 | the only slice whose priors match the queue; headline numbers |
| stratified | 80 | top-up so billing/account/ads/live TV aren't empty |
| hard | 20 | multi-rule collisions and long follow-ups |

How I sampled and labelled: [`reports/GOLDEN.md`](reports/GOLDEN.md). Raw labels I typed are in `data/golden/labels_part{1-4}.tsv`.

## Layout

```
src/taxonomy.py          the 10 intents and the escalation reasons
src/agent/rules.py       regex baseline (and the strata for sampling)
src/agent/classify.py    majority / rules / tfidf / stacked / llm
src/agent/escalate.py    never / rules / thread / llm
src/agent/draft.py       constant / copy-nearest / grounded / llm
src/agent/retrieve.py    the brand's memory + link registry
src/eval/run_eval.py     the harness
src/eval/judge.py        LLM-as-judge rubric (optional)
src/eval/checklist.py    same rubric, deterministic, what `eval` runs
src/eval/agreement.py    human vs judge
data/golden/             labels + my 40-reply ratings
reports/REPORT.md        the write-up
```

## Citations

- Dataset: [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Twitter, public tweets). I loaded an ungated parquet mirror, `TNE-AI/customer-support-on-twitter-conversation` on Hugging Face, so clone-and-run doesn't need a Kaggle token. Same conversations.
- Clustering, TF-IDF, logistic regression, Cohen's κ: scikit-learn 1.5.
- I did **not** use Banking77. Hulu is not a bank; copying 77 retail-banking labels would have been cargo-cult.

Nothing else was copied as code. The agent design (retrieve, then decide, then write, with a link allowlist) is standard RAG-shaped support-bot stuff, not a specific repo.
