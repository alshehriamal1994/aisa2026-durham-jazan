"""Score the candidate-order experiment.

Compares a member's predictions on the original candidate order against the same
member on a shuffled order. Same rows, same four tools, same text, same context.
Only the order in which the four schemas are listed in the prompt differs.

The reference tool is the first candidate on every positive row of the release,
so a system reading position rather than meaning should lose most of its FnAcc
when the order is shuffled, and a system reading meaning should not.
"""
import glob
import importlib.util
import json
import sys

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "nz", "baselines/leaderboard-code-v1_3/normalize.py")
nz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nz)


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
    fn_ok = n_total = arg_ok = n_pos = 0
    for i, (gt, ga) in gold.items():
        p = preds.get(i)
        pred = (p.get("tool_called") if p else None) or "none"
        n_total += 1
        if pred == gt:
            fn_ok += 1
        if gt in (None, "none", ""):
            continue
        n_pos += 1
        if p and pred == gt and nz.args_match(p.get("arguments") or {}, ga, gt):
            arg_ok += 1
    return fn_ok / n_total, arg_ok / n_pos, n_total, n_pos


def slot_profile(pred_path, input_path):
    """Where in the candidate list did the system's answer sit?"""
    inp = {json.loads(l)["id"]: json.loads(l) for l in open(input_path, encoding="utf-8")}
    preds = {json.loads(l)["id"]: json.loads(l) for l in open(pred_path, encoding="utf-8")}
    slots = [0, 0, 0, 0]
    for i, p in preds.items():
        t = p.get("tool_called")
        if t in (None, "none", ""):
            continue
        cands = inp[i].get("candidate_tools") or []
        if t in cands and cands.index(t) < 4:
            slots[cands.index(t)] += 1
    return slots


if __name__ == "__main__":
    gold = load_gold()
    rows = [
        ("original order", "results/shuffle/qC_control.jsonl", "data/processed_v13/dev.jsonl"),
        ("shuffled order", "results/shuffle/qC_shuffled.jsonl", "results/shuffle/dev_shuffled.jsonl"),
    ]
    print(f"{'run':18s}{'FnAcc':>9s}{'ArgEM':>9s}    slot chosen (1/2/3/4)")
    out = {}
    for name, pred, inp in rows:
        try:
            fn, ar, nt, npos = score(pred, gold)
        except FileNotFoundError:
            print(f"{name:18s}   not finished yet")
            continue
        slots = slot_profile(pred, inp)
        out[name] = (fn, ar)
        print(f"{name:18s}{fn:9.4f}{ar:9.4f}    {slots}")
    if len(out) == 2:
        a = out["original order"]
        b = out["shuffled order"]
        print()
        print(f"FnAcc change under shuffling : {b[0] - a[0]:+.4f}")
        print(f"ArgEM change under shuffling : {b[1] - a[1]:+.4f}")
        print()
        if b[0] < a[0] - 0.10:
            print("READING: FnAcc collapses. Position was carrying the tool decision.")
        elif abs(b[0] - a[0]) < 0.02:
            print("READING: FnAcc holds. The model reads the request, not the ordering.")
            print("         The positional artefact is redundant information the model did not need.")
        else:
            print("READING: partial drop. Position contributed but was not the whole story.")
