import json
import sys
import textwrap

rows = [json.loads(l) for l in open("data/golden/to_label.jsonl", encoding="utf-8") if l.strip()]
start = int(sys.argv[1])
end = int(sys.argv[2])
for i, r in enumerate(rows):
    if not (start <= i < end):
        continue
    tag = "OP" if r["is_opener"] else "T%d" % r["turn_index"]
    print(f"[{i:3d}] {r['slice'][:4]:<4} {tag:<3}")
    for h in r["history"][-2:]:
        print("      ctx." + h["role"][:4] + ": " + h["text"][:170])
    for line in textwrap.wrap(r["customer_message"], 150)[:4]:
        print("      MSG: " + line if line == textwrap.wrap(r["customer_message"], 150)[0] else "           " + line)
