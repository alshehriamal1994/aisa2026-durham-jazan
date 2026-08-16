#!/usr/bin/env python3
"""Two gold-free diagnostics comparing the blind test split against dev.

Both exist to test one question. Nine of the seventeen final systems scored
above ArgEM 0.85 on blind, refuting the ~0.85 determinacy bound we had predicted
from our own ensemble's failures. These diagnostics ask whether the blind split
was simply more determinate than dev. They give that defence no support.
Is the blind split simply more determinate than dev?

E1  Ensemble unanimity. Over positive rows, how often do all six members emit an
    identical argument dictionary? Needs no gold. Higher unanimity = the input
    pins the answer down harder = a more determinate split.

E2  Nearest-train-twin similarity. Same difflib char-ratio metric the dev leakage
    audit used (dev: 59% of rows have a >=0.8 twin, median 0.818), now run on the
    blind inputs. Tests whether blind was drawn from the same template pool.

Run:  python3 scripts/blind_determinacy.py
"""
import json, sys, os, re, unicodedata
from difflib import SequenceMatcher
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMBERS = ["dev_allam_v10", "dev_allam_v12", "dev_qwen7b_v7b",
           "dev_qwen7b_v7", "dev_allam_clean", "dev_qwen25_clean"]

AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def canon(v):
    """Light, symmetric normalisation. Not the official scorer, but applied
    identically to dev and blind, so the comparison between them is fair."""
    s = unicodedata.normalize("NFKC", str(v)).translate(AR_DIGITS)
    s = re.sub(r"[ً-ْـ]", "", s)      # diacritics, tatweel
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def argkey(args):
    if not isinstance(args, dict):
        return None
    return tuple(sorted((canon(k), canon(v)) for k, v in args.items()))


def unanimity(pred_dir, label):
    preds = {}
    for m in MEMBERS:
        p = os.path.join(pred_dir, m + ".jsonl")
        if not os.path.exists(p):
            print(f"  [skip {label}] missing {p}")
            return None
        preds[m] = {r["id"]: r for r in load(p)}
    ids = sorted(set.intersection(*(set(d) for d in preds.values())))

    tool_unan = arg_unan = pos = 0
    pairwise = []
    for i in ids:
        tools = [preds[m][i].get("tool_called") for m in MEMBERS]
        called = [t for t in tools if t not in (None, "none", "")]
        if len(called) <= len(MEMBERS) / 2:
            continue                      # majority says no call; skip
        pos += 1
        if len(set(tools)) == 1:
            tool_unan += 1
        keys = [argkey(preds[m][i].get("arguments")) for m in MEMBERS]
        if len(set(keys)) == 1:
            arg_unan += 1
        agree = sum(1 for a in range(len(MEMBERS)) for b in range(a + 1, len(MEMBERS))
                    if keys[a] == keys[b])
        pairwise.append(agree / 15.0)

    print(f"\n  {label}  (positive rows n={pos})")
    print(f"    all six agree on the tool        : {tool_unan/pos:.4f}")
    print(f"    all six agree on the full args   : {arg_unan/pos:.4f}")
    print(f"    mean pairwise argument agreement : {sum(pairwise)/len(pairwise):.4f}")
    return {"n": pos, "tool": tool_unan / pos, "arg": arg_unan / pos,
            "pair": sum(pairwise) / len(pairwise)}


def twins(queries, train_q, label, topk=10):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    T = vec.fit_transform(train_q)
    Q = vec.transform(queries)
    sims = []
    B = 200
    for s in range(0, Q.shape[0], B):
        block = (Q[s:s + B] @ T.T).toarray()
        for r in range(block.shape[0]):
            cand = np.argpartition(block[r], -topk)[-topk:]
            best = max(SequenceMatcher(None, queries[s + r], train_q[c]).ratio()
                       for c in cand)
            sims.append(best)
    sims.sort()
    n = len(sims)
    ge80 = sum(1 for x in sims if x >= 0.8) / n
    exact = sum(1 for x in sims if x >= 0.999) / n
    print(f"\n  {label}  (n={n})")
    print(f"    median nearest-train similarity : {sims[n//2]:.4f}")
    print(f"    fraction with a >=0.8 twin      : {ge80:.4f}")
    print(f"    fraction verbatim in train      : {exact:.4f}")
    return {"n": n, "median": sims[n // 2], "ge80": ge80, "exact": exact}


def main():
    print("=" * 66)
    print("E1  ENSEMBLE UNANIMITY (no gold needed)")
    print("=" * 66)
    d = unanimity(os.path.join(ROOT, "results"), "DEV")
    b = unanimity(os.path.join(ROOT, "results", "blind_preds"), "BLIND")
    if d and b:
        print(f"\n    full-argument unanimity, blind minus dev: "
              f"{b['arg']-d['arg']:+.4f}")

    print("\n" + "=" * 66)
    print("E2  NEAREST-TRAIN-TWIN SIMILARITY (leakage audit, extended to blind)")
    print("=" * 66)
    train = load(os.path.join(ROOT, "data/processed_v13/train.jsonl"))
    train_q = [r["user"] for r in train if r.get("source") == "sharedtask"]
    if not train_q:
        train_q = [r["user"] for r in train]
    print(f"  train pool: {len(train_q)} queries")
    dev = load(os.path.join(ROOT, "data/processed_v13/dev.jsonl"))
    blind = load(os.path.join(ROOT, "data/processed_blind/test.jsonl"))
    td = twins([r["user"] for r in dev], train_q, "DEV")
    tb = twins([r["user"] for r in blind], train_q, "BLIND")
    print(f"\n    median similarity, blind minus dev: {tb['median']-td['median']:+.4f}")
    print(f"    >=0.8-twin rate, blind minus dev  : {tb['ge80']-td['ge80']:+.4f}")

    print("\n  blind dialect mix:",
          dict(Counter(r.get("dialect") for r in blind).most_common()))


if __name__ == "__main__":
    main()
