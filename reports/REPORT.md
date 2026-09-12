# Report: an auto-reply agent for `@hulu_support`

I built a triage-and-draft agent on real Hulu Twitter threads and tried to find out whether I would let it post. The answer is: not unsupervised, and the number that says otherwise is lying.

## 1. Problem framing

Hulu's public Twitter queue is a mix of "when is season 3", "error drm-4-68abfb9b on Roku", "you charged me twice", and "thanks, it's back". A useful bot does three things: name the problem, write a tweet that could have come from this team, and know when it is structurally unable to finish.

**What "good" means here**

- *Intent.* Macro-F1 on a 10-label set I derived from the data, reported on a **random** slice so the prior isn't cooked. Accuracy is a vanity metric — playback_error is 36% of the unbiased queue.
- *Routing.* This is a screen, not a balanced classifier. Missing a refund request (bot "handles" it) is worse than sending a catalogue question to a human. I use escalate-recall as the primary number, escalate-precision next to it, and a cost of **5:1** (missed vs needless) so the tradeoff is explicit. ~31% of the random slice should escalate (37/120), so "never escalate" already gets 69% routing accuracy. That is the trap.
- *Reply.* Two checks that do not need a judge: no invented URLs, no promised refunds/dates, no public asks for passwords. Plus a send/no-send gate. The gate is the one I would actually use.

**What I chose not to build**

- A multi-turn bot. One inbound message → one decision. Follow-ups are in the eval, but the agent does not manage a ticket.
- Account actions (refunds, plan changes, password resets). Twitter is the wrong channel and I don't have the subscriber record. Those intents exist so the router can refuse them.
- Dense embeddings, fine-tuning, or a production LLM in the headline path. See §5 and DECISIONS.md.
- Banking77. Wrong domain.

**Brand.** I scored a dozen support accounts on whether they ever answer in-channel (`src.data.explore`). Amazon/Apple/Uber mostly say "please DM us", which makes drafting vacuous and escalation always-on. `@hulu_support` actually troubleshoots, explains rights, and has a small set of t.co links they reuse hundreds of times. That is enough to ground a draft, and enough to catch a draft that invents a new one.

## 2. The system

Intents came out of k-means over opener messages (k=16 and k=24; dumps in `reports/cluster_dump_k*.txt`), then I merged clusters that Hulu answers the same way. Buffering, named error codes, and "is Hulu down" became `playback_error`. Missing episodes, "when is season N", and "please add X" became `content_availability`. I kept `ads_complaint` apart from `subscription_change` because "there are too many drug ads" is feedback and "I pay for No-Ads and still see ads" is a billing-grade claim.

| intent | default |
|---|---|
| playback_error, content_availability, live_tv_channels, ads_complaint, device_or_feature_feedback, unclear_request, praise_or_chatter | auto, with a playbook |
| billing_charge, account_access, subscription_change | escalate (`account_bound`) |

Pipeline, same four steps for every variant so the comparison is fair: retrieve → classify → route → draft. Retrieval is TF-IDF over word 1–2 grams and char 3–5 grams. Golden-set threads are held out of the knowledge base, including other turns of the same conversation — otherwise a follow-up retrieves its own thread and the eval is a joke.

The headline **system** is not an LLM. Classifying is a stack: high-stakes regexes (money, lockouts) win; when several regexes fire, neighbours vote; short follow-ups look up the last two turns. Routing is the same regex policy as the simple baseline, except it also reads the previous customer turn — labelling taught me that "still happening" is the latest tweet and "I already rebooted everything" sits one turn back. Drafting fills a playbook plus a URL that appears in Hulu's history. No free-text generation, so it cannot invent a release date. It can still pick the wrong playbook, which it does constantly.

Trivial baseline: always `content_availability` (the majority of the weakly-labelled pool), never escalate, one constant deflection.

Simple baseline: first matching regex, same-message routing rules, copy the nearest historical reply verbatim.

TF-IDF + logistic regression is a third point: trained on those regex labels over 3,000 non-golden cases. If n-gram smoothing found structure the rules missed, it would beat them. Mostly it didn't, and it over-escalates.

## 3. Results

All numbers below are from `python -m src.cli eval`. The **random** slice (n=120) is the one I am willing to quote; "all" includes the stratified top-up and is biased toward rare intents.

| variant | slice | intent acc | macro-F1 | esc recall | esc prec | missed / needless | send* | invented URL |
|---|---|---|---|---|---|---|---|---|
| trivial | random | 17.5% [11–24] | 0.03 | 0% | — | 37 / 0 | 0% | 0% |
| simple | random | 45.8% [37–55] | 0.45 | 49% | 82% | 19 / 4 | 36% | 11% |
| tfidf | random | 46.7% [38–56] | 0.46 | 65% | 53% | 13 / 21 | 34% | 11% |
| **system** | random | **53.3% [45–63]** | **0.50** | **51%** | 79% | 18 / 5 | 98% | **0%** |
| system | all | 50.5% | 0.50 | 55% | 74% | 38 / 16 | 95% | 0% |

\*send = checklist judge, see §5. Brackets are 95% bootstrap CIs.

Paired bootstrap on per-case intent correctness, system vs simple, random slice: **+7.5 pp, p = 0.047**. That is just under the usual 0.05 line. I will call it a real but small gain, not a launch criterion. What else I can claim:

- The system invents **zero** URLs. Copy-nearest invents 11%, because it pastes one-off t.co links that aren't in the 3+ registry.
- Escalation handoff is consistent (100% of system-escalate tweets contain the phone/chat link). Copy-nearest is at 48% — it will copy a reboot guide onto a case we decided to escalate.
- Routing cost per case (5·miss + 1·needless)/n: trivial 1.54, simple 0.83, system 0.79, tfidf 0.72. TF-IDF "wins" on cost by catching more exhausted-self-service cases and paying for it with 21 needless escalations. I would not ship that; it just moves work onto humans.
- Reason match on the random slice, where both sides escalated: 84% exact, 95% any-overlap (all-slice 74% / 87%). The common mismatch is gold `churn_risk` vs pred `account_bound` — both are "give this to a human", and `account_bound` is what the priority list reports when a cancel-threat is also a plan-change.

Per-class F1 on the random slice, system: account_access 0.80, ads_complaint 0.80, content_availability 0.55, playback_error 0.58 (recall 0.44 — it still leaks into live_tv and content), praise 0.63, billing 0.60, live_tv 0.42, device_feedback 0.25, subscription_change 0.33 (n=3), **unclear_request 0.00**. Unclear is the hole. Nothing in the regexes, and neighbours vote for whatever the thread used to be about.

Ablations (same random slice): `stacked_copy` keeps the classifier and copies replies — send drops from 98% to 36%, invented URLs return (11%). `rules_grounded` keeps the regex classifier and the playbook drafter — intent acc drops back to 45.8%, send stays high. So the send-rate jump is the drafter, and the intent bump is the stack (playback-vs-live-tv, history on follow-ups). Neither is a miracle.

## 4. Failure analysis

Five modes, with real golden-set cases. I did not pick the cute ones.

**1. Regex-silent → the catalogue speech.** When no regex fires, neighbours are weakly labelled by those same regexes, so they vote `content_availability` (61% of the silver pool). The drafter then recites streaming rights. This is how "Thank you very much, Hulu", "Awesome its back up!", "GOT, Arrow, Brooklyn 99…", and "I have sent you several Tweets without getting a response" all get the same paragraph about licensing. I patched obvious thanks; the rest still fall through. Hypothesis: the fallback should be `unclear_request` (ask one question) unless neighbour similarity is high *and* the neighbour labels agree. I have a 0.4 confidence floor; it is not enough.

**2. "Live TV" hijacks playback.** `live_tv_channels` is listed above `playback_error` in the regexes, on purpose — carriage questions are real. The first version turned "restarted my Xbox during live tv" and "NBA kept freezing" into zip questions. I added a playback-cue override in the stack (not in the simple baseline). Playback recall went from 0.35 to 0.44. Eight of 43 still leak, mostly game-quality complaints with no buffer/error/reboot word. The opposite error is still there: CBS "this stream is unavailable" is gold `live_tv_channels`, pred `playback_error`.

**3. Exhaustion that isn't in the last tweet's regex.** Gold escalate-reason `self_service_exhausted` is 18/37 of the random slice. I added "restarted everything twice" after missing it, and the thread router looks one turn back. Still missed: "why is my hulu giving a loading error on xbox? No access for a solid 24 hours and ive restarted everything twice" — wait, that one should fire now; the remaining misses are vaguer: "In the bottom of the 9th?!" (history says they already reported the TBS feed), a screenshot-only tweet, "Some shows updated overnight, but others did not." Hypothesis: exhaustion is often *pragmatic* (turn index ≥ 4, or Hulu already opened an investigation) rather than lexical. A turn-index feature would help and would also overfire on chatty sports fans.

**4. Two devices, one slot.** "LG webos Hulu smart app keeps buffering. … Works fine through the Xbox one." The first version of the drafter returned Xbox (first match in DEVICE_PATTERNS). I changed it to pick the device nearest a failure verb; that case now says LG. It will still fail when both devices are in the same clause.

**5. Anything not in English, and anything the taxonomy doesn't have.** Spanish "me debería dejar navegar…" used to get an English Live-TV zip ask — the language screen wanted two function-word hits and only got one. I added a short list of strong markers (`debería`, `anuncios`, …); that row now escalates as `unsupported_language`. The *tweet* is still English, so I still would not send it. Taxonomy gaps I wrote down while labelling and then ignored: watch-progress not saving (I dumped it in `playback_error`), "what's your email support address", a Verizon agent's text concatenated onto a customer tweet, Sailor Moon filed under LGBTQ Programming (gold: `no_precedent`). The similarity floor is 0.18; Sailor Moon still retrieved something.

Honourable mention, because it is funny and structural: row 11 of the golden set is a Spotify+Hulu bundle question sitting in a Hulu thread. Cross-brand contamination. The agent cannot win.

## 5. What is misleading about my headline number

There isn't one headline number; there are several, and each one is doing a different kind of lying.

**"53% intent accuracy."** The CI is 45–63%. Trivial is 18% only because I used the *silver* majority (`content_availability`), not the true inbound majority (`playback_error`, 36%). If I had set the constant baseline to playback_error, trivial would look like 36% and the jump to 53% would look smaller. The 7.5 pp over the regex baseline has p = 0.047 — significant if you like 0.05, not a finding I would take to a launch review. Macro-F1 (0.50 vs 0.45) is the less-wrong summary.

**"98% send rate."** This is the checklist implementation of my own rubric, run on my own templates. Of course it likes them: they contain contractions, a real URL, and a question mark, which is most of what the checklist treats as "helpful = 2". I scored 40 of the same replies myself, blind to the checklist scores. I would send **12/40 (30%)**. Agreement on the send gate: 32.5%, **Cohen's κ = 0.02**. That is chance. We do agree on *safe* (Spearman 0.78) — neither of us wants a public refund. The headline send rate is a measurement of template-shapedness, not of whether a social-media lead would hit tweet.

The LLM-as-judge in `src/eval/judge.py` is the same rubric, as a prompt, blinded to variant. I did not run it for the reported numbers because the 15-minute path cannot depend on a key, and a cached Gemini score you cannot regenerate is just a more expensive checklist. If I had reported the checklist send rate without this section I would have been cheating.

**"80% routing accuracy."** Never-escalate gets 69% on the same slice. The 11-point gap is real; quoting 80% without the 69% is not.

**"All-slice macro-F1."** The stratified 80 were sampled with the same regexes that the simple baseline uses. That flatters the regexes (and anything that leans on them, including the system). I put the random-only column in the table for a reason. Per-class F1 on billing/account in the all-slice is not a claim about the live queue.

**I am the only rater.** I wrote the taxonomy, I labelled the 220, I wrote the templates, I wrote the checklist, I rated the 40. There is no independent human. Judge-agreement here is "me vs a script I wrote", which is why κ = 0.02 is the result that should make you trust me more, not less: when I actually read the tweets I did not rubber-stamp my own bot.

**Token-F1 vs Hulu's real reply is 0.15 for the system and 0.18 for copy-nearest.** Copying the nearest tweet is closer to what they said, as you'd expect, and still low, because there are many good replies. I am not using it as a quality metric. If I had, copy-nearest would "win" at reply quality while telling Xbox users to reboot a Roku.

## 6. One more week

1. Split the 220: 60 for few-shot / threshold tuning, 160 frozen. I burned the whole set on reporting, so I cannot honestly retune.
2. Put an LLM on classification only, with the taxonomy block and six examples from the 60. Keep the router and the playbook drafter. That's the ablation `llm_clf_only` already wired; I didn't spend the key.
3. A second rater on 80 of the golden set, and a different-family model as judge (the prompt is ready). Until then I would not quote a send rate in a meeting.
4. A real language detector (the strong-marker list is a hack that happens to catch the one Spanish row); a `watch_progress` label so I stop stuffing it into playback; don't ask for an error code they already typed.
5. Cost-curve for the router: plot missed vs needless as I move the "look at history / turn index" knobs, pick a point with the 5:1 weights, freeze it on the 60.

I would not spend the week making the templates sound more like Becki. Voice is the axis I already pass and the one that doesn't hurt anyone when I fail.
