#!/usr/bin/env python3
"""Convention-leak analysis: of the positives our SHIPPING vote gets wrong,
how many are *transferable* convention misses (recoverable by training the
model to emit the canonical form — helps blind too) vs *irreducible* noise
(free-text dates, ID miscopy)?

For each wrong positive, find the argument key(s) that fail value_match and
bucket them. Transferable = the model's value is the same concept but a
surface/convention the evaluator does not bridge (esp. termination_type
Arabic<->English enum, and closed-class values missing from the alias table).
"""
import json, sys, collections
sys.path.insert(0, "baselines/leaderboard-code-v1_1")
from normalize import value_match, canon_value, FIELD_CLASS, ID_FIELDS, _as_number  # noqa

def load(p):
    o = {}
    for l in open(p, encoding="utf-8"):
        l = l.strip()
        if l:
            x = json.loads(l); o[x["id"]] = x
    return o

gold = load("data/processed_v11/dev.jsonl")
vote = load("results/dev_vote7_locked.jsonl")

DATEKEYS = {"date", "check_in", "check_out", "appointment_date", "departure_date", "return_date"}

buckets = collections.Counter()
examples = collections.defaultdict(list)
fail_keys = collections.Counter()

for gid, g in gold.items():
    if not g.get("requires_function"):
        continue
    ga = {k: v for k, v in (g.get("arguments") or {}).items() if v not in (None, "")}
    p = vote.get(gid, {})
    pa = {k: v for k, v in (p.get("arguments") or {}).items() if v not in (None, "")}
    # whole-row correct? skip
    if set(pa) == set(ga) and all(value_match(pa[k], ga[k], k) for k in ga):
        continue
    # key-set mismatch (added/dropped arg) — separate bucket
    if set(pa) != set(ga):
        buckets["key_set_mismatch"] += 1
        continue
    # same keys: find the failing value(s)
    for k in ga:
        if value_match(pa[k], ga[k], k):
            continue
        fail_keys[k] += 1
        pv, gv = str(pa[k]), str(ga[k])
        cls = FIELD_CLASS.get(k)
        if k == "termination_type":
            b = "termination_type (Ar<->En enum, NOT aliased = TRANSFERABLE if we train canonical)"
        elif k in DATEKEYS:
            b = "date/time (gold internally inconsistent = IRREDUCIBLE)"
        elif k in ID_FIELDS or (_as_number(pv) is not None and _as_number(gv) is not None and k in ("recipient_iban","insurance_number","id_number")):
            b = "ID/number miscopy (IRREDUCIBLE)"
        elif cls:
            # closed-class: is the value the same concept but missing from alias table?
            b = f"closed-class[{cls}] leak (pred!=gold canon = TRANSFERABLE: add alias / train canonical)"
        else:
            b = "free-text entity (copy/translate)"
        buckets[b] += 1
        if len(examples[b]) < 6:
            examples[b].append(f"{g['tool_called']}.{k}: pred={pv!r} | gold={gv!r}")

print("== where our shipping 7-vote loses ArgEM (per failing argument) ==\n")
total = sum(buckets.values())
for b, c in buckets.most_common():
    tag = "🟢 RECOVERABLE" if "TRANSFERABLE" in b else ("🔴 irreducible" if "IRREDUCIBLE" in b else "⚪")
    print(f"{tag}  {c:3d}  {b}")
print(f"\n total failing-argument incidents = {total}")
print("\n top failing keys:", dict(fail_keys.most_common(12)))
for b in examples:
    if "TRANSFERABLE" in b:
        print(f"\n--- examples: {b} ---")
        for e in examples[b]:
            print("   ", e)
