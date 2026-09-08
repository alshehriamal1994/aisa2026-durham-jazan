"""Audit the position of the reference tool among the candidate tools.

For every row of a split that calls a tool, report whether the reference tool
is the first of the candidate schemas as the release stores them
(the `tools_sampled` column, which is also the order in which the schemas are
rendered into the prompt). Prints the count per split and the position
histogram.

Usage:
    python scripts/audit_candidate_order.py --local data/raw/aisa_v1_6
    python scripts/audit_candidate_order.py --hub --revision 35338790   # v1.6
    python scripts/audit_candidate_order.py --hub --revision f43e65a2   # v1.4

The blind test split ships with the gold nulled, so it cannot be audited here.
"""
import argparse
import collections
import glob
import os

import pandas as pd

DATASET = "TuwaiqAcademy/AISA-ArabicFC"


def audit(df, name):
    hist = collections.Counter()
    n = 0
    for _, r in df.iterrows():
        if not r["requires_function"]:
            continue
        cands = [t["function"]["name"] for t in r["tools_sampled"]]
        n += 1
        hist[cands.index(r["tool_called"]) + 1 if r["tool_called"] in cands else 0] += 1
    first = hist.get(1, 0)
    print(f"{name:6s} positives {n:6d}   reference first on {first:6d} ({first / n:.1%})   position histogram {dict(sorted(hist.items()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="directory holding data/<split>-*.parquet")
    ap.add_argument("--hub", action="store_true", help="download from the Hugging Face hub")
    ap.add_argument("--revision", default="main", help="hub commit or branch (main = v1.6 as of September 2026)")
    ap.add_argument("--splits", nargs="*", default=["train", "dev"])
    a = ap.parse_args()
    for split in a.splits:
        if a.hub:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(DATASET, f"data/{split}-00000-of-00001.parquet", repo_type="dataset", revision=a.revision)
        else:
            path = glob.glob(os.path.join(a.local, "data", f"{split}-*.parquet"))[0]
        audit(pd.read_parquet(path), split)


if __name__ == "__main__":
    main()
