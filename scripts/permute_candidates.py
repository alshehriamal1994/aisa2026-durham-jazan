"""Permute the candidate_tools order in a training file, changing nothing else.

Used for the retraining experiment in the paper's Appendix G. The seed is fixed
so the permutation is reproducible, and every field except candidate_tools is
byte-identical to the input, which can be verified by hashing.

    python3 scripts/permute_candidates.py \
        --input data/processed_v13/train_st_aug.jsonl \
        --output data/processed_v13/train_st_aug_permuted.jsonl --seed 20260816
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=20260816)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    moved = total = 0
    with open(args.input, encoding="utf-8") as f, \
         open(args.output, "w", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            c = list(r.get("candidate_tools") or [])
            if len(c) > 1:
                before = list(c)
                rng.shuffle(c)
                moved += (c != before)
            r["candidate_tools"] = c
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += 1
    print(f"{total} rows, order changed on {moved}")


if __name__ == "__main__":
    main()
