"""Score the permuted-order retraining experiment.

Design. One variable is changed: the order of the four candidate tools in the
training file. Everything else is byte-identical, verified by hashing every field
except candidate_tools. Hyperparameters, epochs, bases and inference settings are
those of the released recipe in scripts/chain_clean_zoo.sh.

Each adapter is then evaluated on both orderings of the development split, giving

                        eval canonical    eval shuffled
    trained canonical        A                 B
    trained permuted         C                 D

A and B are already measured (the submitted adapters, same-day controls).
C and D come from this run.

Readings:
  D near A  -> the ordering dependence is trainable away. The 0.202 collapse was a
               property of how the release is written, not a limit of 7B Arabic models.
  C near A  -> permuted training costs nothing on the canonical order, so the fix
               is free and the organisers can adopt it without penalising anyone.
  C well below A -> the ordering was carrying real signal for everyone, and the
               leaderboard partly ranked who absorbed it best.
"""
import glob
import importlib.util
import json

import pandas as pd

nz_spec = importlib.util.spec_from_file_location(
    "nz", "baselines/leaderboard-code-v1_3/normalize.py")
nz = importlib.util.module_from_spec(nz_spec)
nz_spec.loader.exec_module(nz)


def load_gold():
    df = pd.read_parquet(glob.glob("data/raw/aisa_v1_4/data/dev-*.parquet")[0])
    df = df.reset_index(drop=True)
    gold = {}
    for i, row in df.iterrows():
        args = {}
        if row["requires_function"]:
            for m in row["messages"]:
                if m.get("role") == "assistant" and m.get("tool_calls") is not None:
                    for tc in m["tool_calls"]:
                        args = {k: v for k, v in dict(tc["function"]["arguments"]).items()
                                if v is not None}
        gold[i] = (row["tool_called"] if row["requires_function"] else "none", args)
    return gold


def score(path, gold):
    preds = {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8")}
    fn = n = ae = npos = 0
    for i, (gt, ga) in gold.items():
        pred = (preds[i].get("tool_called") if i in preds else None) or "none"
        n += 1
        fn += (pred == gt)
        if gt in (None, "none", ""):
            continue
        npos += 1
        if pred == gt and nz.args_match(preds[i].get("arguments") or {}, ga, gt):
            ae += 1
    return fn / n, ae / npos


def slot_one_share(pred_path, input_path):
    inp = {json.loads(l)["id"]: json.loads(l) for l in open(input_path, encoding="utf-8")}
    preds = {json.loads(l)["id"]: json.loads(l) for l in open(pred_path, encoding="utf-8")}
    first = called = 0
    for i, p in preds.items():
        t = p.get("tool_called")
        if t in (None, "none", ""):
            continue
        called += 1
        cands = inp[i].get("candidate_tools") or []
        first += bool(cands and cands[0] == t)
    return first / called if called else 0.0


CANON = "data/processed_v13/dev.jsonl"
SHUF = "results/shuffle/dev_shuffled.jsonl"
CELLS = [
    ("allam",  "trained canonical", "results/shuffle/aC_control.jsonl",              CANON),
    ("allam",  "trained canonical", "results/shuffle/aC_shuffled.jsonl",             SHUF),
    ("allam",  "trained permuted",  "results/permtrain/allam_perm_canonical.jsonl",  CANON),
    ("allam",  "trained permuted",  "results/permtrain/allam_perm_shuffled.jsonl",   SHUF),
    ("qwen25", "trained canonical", "results/shuffle/qC_control.jsonl",              CANON),
    ("qwen25", "trained canonical", "results/shuffle/qC_shuffled.jsonl",             SHUF),
    ("qwen25", "trained permuted",  "results/permtrain/qwen25_perm_canonical.jsonl", CANON),
    ("qwen25", "trained permuted",  "results/permtrain/qwen25_perm_shuffled.jsonl",  SHUF),
]

if __name__ == "__main__":
    gold = load_gold()
    print(f"{'base':8s}{'training':20s}{'eval order':13s}"
          f"{'FnAcc':>8s}{'ArgEM':>8s}{'says slot 1':>13s}")
    for base, training, pred, inp in CELLS:
        evalorder = "canonical" if inp == CANON else "shuffled"
        try:
            fn, ae = score(pred, gold)
        except FileNotFoundError:
            print(f"{base:8s}{training:20s}{evalorder:13s}   pending")
            continue
        print(f"{base:8s}{training:20s}{evalorder:13s}"
              f"{fn:8.4f}{ae:8.4f}{slot_one_share(pred, inp):12.1%}")
    print("\nreference: gold sits in slot 1 on 100% of canonical rows "
          "and 26.2% of shuffled rows")
