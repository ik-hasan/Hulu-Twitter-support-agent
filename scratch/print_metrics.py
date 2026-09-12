import json
from collections import Counter

m = json.load(open("artifacts/metrics.json", encoding="utf-8"))
print("tests", m.get("_tests"))
for v in ["trivial", "simple", "tfidf", "system"]:
    r = m[v]["random"]
    print("==", v, "==")
    print("  acc", round(r["intent"]["accuracy"], 3), "ci", [round(x, 3) for x in r["intent"]["accuracy_ci"]])
    print("  f1 ", round(r["intent"]["macro_f1"], 3))
    rt = r["routing"]
    print("  esc rec", round(rt["escalate_recall"], 3), "prec", round(rt["escalate_precision"], 3),
          "cost", round(rt["cost_per_case"], 3), "miss", rt["missed_escalations"], "needless", rt["needless_escalations"])
    print("  send", round((r.get("judge") or {}).get("send_rate", 0), 3),
          "invent", round(r["reply"]["invented_link_rate"], 3),
          "handoff", round(r["reply"]["handoff_consistency"], 3))
    print("  per-class:")
    for k, val in r["intent"]["per_class"].items():
        if val["support"]:
            print(f"    {k:<28} n={val['support']:<3} p={val['precision']:.2f} r={val['recall']:.2f} f1={val['f1']:.2f}")
    print()

# how many of the 40 rated replies changed
old = {json.loads(l)["case_id"]: json.loads(l)["reply"]
       for l in open("artifacts/to_rate.jsonl", encoding="utf-8") if l.strip()}
# to_rate is from previous run - preds_system is new
newp = {json.loads(l)["case_id"]: json.loads(l)["reply"]
        for l in open("artifacts/preds_system.jsonl", encoding="utf-8") if l.strip()}
changed = [cid for cid in old if old[cid] != newp.get(cid)]
print("rated replies changed", len(changed), "/", len(old))
for cid in changed[:8]:
    print("---", cid)
    print("OLD", old[cid][:140])
    print("NEW", newp.get(cid, "")[:140])
