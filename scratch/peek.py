import random
import sys

from src.data.prepare import read_cases

cs = read_cases()
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 18
for c in random.sample(cs, n):
    tag = "opener" if c["is_opener"] else "turn%d" % c["turn_index"]
    print("---", c["case_id"], tag, "after=%d" % c["turns_after"])
    for h in c["history"][-2:]:
        print("   ctx[%s]: %s" % (h["role"][:4], h["text"][:150]))
    print("   CUST:", c["customer_message"][:280])
    print("   HULU:", c["agent_reply_clean"][:280])
