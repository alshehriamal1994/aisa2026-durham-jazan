"""Score the full six-member vote under shuffled candidate order.

Runs the released combiner over the six shuffled-order member predictions and
compares against the same combiner over the original order, so the comparison is
system-level rather than single-member. Convention rules are applied in both
conditions, since they operate on the user text and are unaffected by ordering.
"""
import copy
import glob
import importlib.util
import json

import pandas as pd

nz_spec = importlib.util.spec_from_file_location(
    "nz", "baselines/leaderboard-code-v1_3/normalize.py")
nz = importlib.util.module_from_spec(nz_spec)
nz_spec.loader.exec_module(nz)

ev_spec = importlib.util.spec_from_file_location("ev", "scripts/ensemble_vote.py")
ev = importlib.util.module_from_spec(ev_spec)
try:
    ev_spec.loader.exec_module(ev)
except SystemExit:
    pass

ORDER = ["v10", "v12", "v7b", "v7", "aC", "qC"]
ORIGINAL = {
    "v10": "results/dev_allam_v10.jsonl",
    "v12": "results/dev_allam_v12.jsonl",
    "v7b": "results/dev_qwen7b_v7b.jsonl",
    "v7": "results/dev_qwen7b_v7.jsonl",
    "aC": "results/dev_allam_clean.jsonl",
    "qC": "results/dev_qwen25_clean.jsonl",
}
SHUFFLED = {m: f"results/shuffle/{m}_shuffled.jsonl" for m in ORDER}


def load_gold():
    df = pd.read_parquet(glob.glob("data/raw/aisa_v1_4/data/dev-*.parquet")[0])
    df = df.reset_index(drop=True)
    gold, user = {}, {}
    for i, row in df.iterrows():
        args = {}
        for m in row["messages"]:
            if m.get("role") == "user":
                user[i] = m.get("content") or ""
            if m.get("role") == "assistant" and m.get("tool_calls") is not None:
                for tc in m["tool_calls"]:
                    args = {k: v for k, v in dict(tc["function"]["arguments"]).items()
                            if v is not None}
        gold[i] = (row["tool_called"] if row["requires_function"] else "none",
                   args if row["requires_function"] else {})
    return gold, user


def vote(members, user_text, rules=True):
    """Reproduce the submitted combiner over a dict of member predictions."""
    ids = sorted(set.intersection(*[set(m) for m in members.values()]))
    out = []
    for i in ids:
        active = [(name, members[name][i]) for name in ORDER if i in members[name]]
        out.append(ev.vote_one(i, active, ORDER))
    if rules:
        out = ev.apply_train_conventions(out, user_text, True, True)
    return {r["id"]: r for r in out}


def score(preds, gold):
    fn_ok = n = arg_ok = npos = 0
    for i, (gt, ga) in gold.items():
        p = preds.get(i)
        pred = (p.get("tool_called") if p else None) or "none"
        n += 1
        fn_ok += (pred == gt)
        if gt in (None, "none", ""):
            continue
        npos += 1
        if p and pred == gt and nz.args_match(p.get("arguments") or {}, ga, gt):
            arg_ok += 1
    return fn_ok / n, arg_ok / npos, npos


if __name__ == "__main__":
    gold, user = load_gold()
    L = lambda p: {json.loads(l)["id"]: json.loads(l)
                   for l in open(p, encoding="utf-8")}
    print(f"{'member':8s}{'FnAcc orig':>12s}{'FnAcc shuf':>12s}"
          f"{'ArgEM orig':>12s}{'ArgEM shuf':>12s}")
    orig, shuf, missing = {}, {}, []
    for m in ORDER:
        try:
            orig[m], shuf[m] = L(ORIGINAL[m]), L(SHUFFLED[m])
        except FileNotFoundError:
            missing.append(m); continue
        fo, ao, _ = score(orig[m], gold)
        fs, asf, _ = score(shuf[m], gold)
        print(f"{m:8s}{fo:12.4f}{fs:12.4f}{ao:12.4f}{asf:12.4f}")
    if missing:
        print(f"\npending: {', '.join(missing)}")
        raise SystemExit(0)
    print()
    for tag, rules in [("vote + enum", False), ("+ both rules  [SUBMITTED]", True)]:
        fo, ao, npos = score(vote(orig, user, rules), gold)
        fs, asf, _ = score(vote(shuf, user, rules), gold)
        print(f"{tag:26s} FnAcc {fo:.4f} -> {fs:.4f} ({fs-fo:+.4f})"
              f"   ArgEM {ao:.4f} -> {asf:.4f} ({asf-ao:+.4f})   n={npos}")
