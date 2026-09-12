# Decision log

The non-obvious calls, in roughly the order I made them. Numbered so the code comments can point here.

1. **Brand = `@hulu_support`, not Amazon/Apple/Uber.** Those three have more tweets. They also deflect to DMs at a rate that makes "draft a grounded reply" a non-problem: the grounded reply is always "please DM us". Hulu actually troubleshoots in public and reuses a handful of t.co links. Measured in `src/data/explore.py`.

2. **HuggingFace parquet mirror, not the Kaggle CSV, as the default download.** Same conversations; no token. A clone that requires `kaggle.json` is not reproducible. If the CSV is present it still wins (`src/data/prepare.py`).

3. **Intents from over-clustering, then merged by playbook, not by topic.** k=16 and k=24. Four "it doesn't work" clusters became one intent because Hulu's first reply is the same troubleshooting walkthrough. Three "where's the show" clusters became one because every answer is a rights statement. Dumps: `reports/cluster_dump_k16.txt`, `_k24.txt`.

4. **Kept `ads_complaint` separate from `subscription_change`.** Both touch the No-Ads plan. "There are too many drug ads" is feedback. "I pay for No-Ads and still see ads" is a billing-grade claim. Collapsing them hid that.

5. **`needs_account_access` is a field on the intent, not a vibe.** Billing, lockouts, and plan changes cannot be finished on Twitter. This one boolean does most of the escalation work. Situational regexes (legal, exhausted, churn, language) sit on top.

6. **Closed escalation-reason vocabulary.** A free-text reason cannot be scored. The brief asked for a stated reason; a closed set is the only way I could measure which reason fired. Priority order is safety/law > language > account > churn > exhausted > no-precedent > low-confidence.

7. **Prediction unit = a customer message the brand actually answered, including mid-thread.** Openers-only would have been cleaner and would have flattered every number. The real queue is full of "Roku" and "still happening". `turn_index` is on every row so I can slice later.

8. **Sparse TF-IDF retrieval, not embeddings.** Short tweets are dominated by product nouns, error codes, device names. Char grams keep `drm-4-68abfb9b`. No model download, reruns in seconds. Neighbour similarity is also the `no_precedent` signal.

9. **Golden-set threads held out of the KB, including other turns of the same conversation.** Holding out the row but keeping the rest of the thread is how a follow-up "retrieves" the answer Hulu already gave.

10. **120 / 80 / 20 sampling (random / stratified / hard), and headline numbers only on random.** One scheme cannot both estimate the live prior and give rare classes a per-class F1. Stratified used the regexes, which biases that slice toward the simple baseline — so I don't quote it as the live number.

11. **Labelled blind to the regex guess.** `to_label.jsonl` does not show `_rule_intent`. I would just have ratified my own patterns.

12. **Headline system is offline.** I wrote the LLM classify/route/draft path (`system_llm`). I did not use it for reported numbers. A 15-minute reproduce that needs my Gemini key is not a reproduce. The LLM path is there for a live demo if a key is present.

13. **TF-IDF baseline trained on regex silver, not LLM silver.** Training on LLM labels makes TF-IDF a student of the LLM. Training on the regexes makes it a smoothed copy of the simple baseline, which is the comparison I actually wanted: do n-grams generalise past the patterns? Mostly no.

14. **Thread router looks at the previous customer turn; the simple baseline does not.** Labelling: "still happening" is the latest tweet, "I already rebooted" is one turn back. Giving the system an input the baseline can't see is normally cheating. Here it is the intervention being tested, and I say so.

15. **Link allowlist from URLs Hulu used ≥3 times.** A bot that invents a help-centre URL is worse than a bot that sends no link. Copy-nearest is allowed to fail this; the system is not.

16. **Did not use Banking77.** Hulu is not a bank. Importing 77 retail-banking intents would have been looking like I used the optional dataset, not using it.

17. **Routing cost 5:1 (missed:needless).** Arbitrary but stated. Letting a refund request through to a template is several times worse than a human answering "when is Rick and Morty back". Changing the ratio changes who "wins" between system and TF-IDF; I would still not ship TF-IDF's 21 needless escalations.

18. **`no_precedent` similarity floor 0.18, set on a peek at the pool, not on the golden set.** Still too low: Sailor Moon retrieved *something*. I left it. Tuning it on the 220 would have been leaking.

19. **Four patches after the first eval, then I stopped.** Playback cue beats live-TV in the stack only; device slot picks the name nearest a failure verb; language screen got a short strong-marker list plus an accented-letter ratio; `pick_link` prefers the intent-default brand URL over neighbour-frequency. I also made the checklist use the same link registry as the invented-URL metric — scoring only the top-3 neighbours was calling the playbook's real t.co "invented". I did not retune on the full 220 after that.
