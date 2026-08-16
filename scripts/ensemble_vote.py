#!/usr/bin/env python3
"""Per-argument majority vote over N model prediction files -> one submission.

Locked combiner for the AISA-ArabicFC blind test. Plain majority (no dev-tuned
weights, so it generalizes). Reproduces dev ArgEM 0.822 / OverallA 0.8925 (v1.4 gold) from the 6 locked
prediction files. Usage:

  # vote existing prediction files -> submission
  python scripts/ensemble_vote.py --preds a.jsonl b.jsonl ... --out submission.jsonl
  # or pull file list + tiebreak order from the locked config
  python scripts/ensemble_vote.py --from-config scripts/ensemble_config.json \
      --pred-dir results --pred-prefix dev_ --out results/dev_vote7.jsonl
  # add --score data/processed_v11/dev.jsonl to print OverallA/ArgEM/FnAcc
"""
import argparse, json, collections, sys, re

sys.path.insert(0, "baselines/leaderboard-code-v1_3")
try:
    from normalize import canon_value  # noqa: E402
except ModuleNotFoundError as e:  # the organisers' evaluator is not vendored here
    raise SystemExit(
        "This needs the shared-task evaluator. Place it at "
        "baselines/leaderboard-code-v1_3/ (normalize.py, eval_lib.py) and re-run."
    ) from e

# Tool schemas — used for the enum-validity rule (a transferable, CV-validated
# combiner improvement: prefer a candidate that is a valid "Supported values"
# member; 0.814 -> 0.816 on v1.3 dev, OOF-confirmed => generalizes to blind).
try:
    _REG = json.load(open("data/processed_v13/tools_registry.json"))
except Exception:
    _REG = {}


def _enum_for(tool, key):
    p = ((_REG.get(tool, {}) or {}).get("parameters", {}) or {}).get("properties", {}).get(key, {}) or {}
    m = re.search(r"Supported values:\s*([^).]+)", p.get("description", "") or "")
    return [x.strip() for x in m.group(1).split(",")] if m else None


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        x = json.loads(line)
        out[x["id"]] = x
    return out


def vote_one(gid, preds, order):
    """preds: list of (name, pred dict) present for this id. order: tiebreak rank."""
    if not preds:
        return None
    rank = {n: i for i, n in enumerate(order)}
    tc = collections.Counter((p.get("tool_called") or "none") for _, p in preds)
    best_tool = tc.most_common(1)[0][0]
    active = [(n, p) for n, p in preds if (p.get("tool_called") or "none") == best_tool]
    # carry a reasoning trace (Track B ThinkRate = non-empty <think>): take the
    # best-ranked active model's trace, else any non-empty one.
    think = ""
    for n, p in sorted(active, key=lambda np: rank.get(np[0], 99)):
        if (p.get("think") or "").strip():
            think = p["think"]
            break
    if best_tool == "none":
        return {"id": gid, "tool_called": "none", "arguments": {}, "think": think}
    # which keys appear in a majority of the tool-agreeing models
    kc = collections.Counter()
    for _, p in active:
        for k in (p.get("arguments") or {}):
            kc[k] += 1
    args = {}
    for k, cnt in kc.items():
        if cnt < len(active) / 2:
            continue
        # cluster candidate values by canonical form; pick the largest cluster,
        # tie-break by best-ranked model that produced it
        clusters = collections.defaultdict(list)
        for n, p in active:
            v = (p.get("arguments") or {}).get(k)
            if v is None or not str(v).strip():
                continue
            clusters[canon_value(v, k)].append((rank.get(n, 99), v))
        if clusters:
            keyfn = lambda kv: (len(kv[1]), -min(r for r, _ in kv[1]))
            best_c = max(clusters.items(), key=keyfn)
            # enum-validity override: if the field declares supported values,
            # prefer the largest cluster that IS a valid member.
            enum = _enum_for(best_tool, k)
            if enum:
                valid_canon = {canon_value(e, k) for e in enum}
                valid = [c for c in clusters.items() if c[0] in valid_canon]
                if valid:
                    best_c = max(valid, key=keyfn)
            best_c[1].sort()  # by rank asc
            args[k] = best_c[1][0][1]
    return {"id": gid, "tool_called": best_tool, "arguments": args, "think": think}


# Train-dominant date-year convention (TRAIN-only statistic, 2026-07-04):
# search_hotels check_in/check_out gold is 80% year-2023 and book_doctor date
# is 100% year-2023 in train regardless of the context date. Rewriting a
# predicted year to 2023 when no year is stated in the user query = +0.004
# ArgEM on dev (0.818->0.822, official v1.3 evaluator). Derived from train, so
# it transfers to any split from the same generator. Opt-in via
# --train-conventions --input <jsonl with id+user>.
_YEAR_RULE_FIELDS = {"search_hotels": ("check_in", "check_out"),
                     "book_doctor_appointment": ("date",)}

# v1.4 (2026-07-04) train policy: these args are kept in gold ONLY when the
# value appears verbatim in the user text (recipient_iban 276-vs-2 in new
# train; insurance_number 371-vs-26). Dropping non-verbatim values recovers
# the 4 July gold change: dev 0.800 -> 0.822 ArgEM under v1.4.
_OMIT_RULE_FIELDS = {"transfer_money": ("recipient_iban",),
                     "check_insurance_coverage": ("insurance_number",)}


def apply_train_conventions(voted, id2user, year_rule=True, omit_rule=True):
    """Both rules default ON — that is the configuration verified at dev
    ArgEM 0.822 / OverallA 0.8925 under v1.4 gold. The toggles exist because the
    two rules carry very different transfer risk to the blind test:
      - year_rule is a property of the DATA GENERATOR (train dates 2023-pinned
        80-100% regardless of context date) -> expected to transfer.
      - omit_rule drops recipient_iban and insurance_number when the value is
        absent from the user text, matching how the training gold keeps those
        fields. Worth +0.022 on dev against the final data release.
    """
    n = m_ = 0
    for p in voted:
        q = id2user.get(p["id"], "")
        if year_rule:
            for f in _YEAR_RULE_FIELDS.get(p.get("tool_called"), ()):
                v = str((p.get("arguments") or {}).get(f, ""))
                mt = re.match(r"(20\d\d)(-.*)", v)
                if mt and mt.group(1) != "2023" and mt.group(1) not in q:
                    p["arguments"][f] = "2023" + mt.group(2)
                    n += 1
        if omit_rule:
            for f in _OMIT_RULE_FIELDS.get(p.get("tool_called"), ()):
                v = (p.get("arguments") or {}).get(f)
                if v is not None and str(v) not in q:
                    del p["arguments"][f]
                    m_ += 1
    print(f"[conventions] year-rule rewrote {n}; omit-rule dropped {m_}")
    return voted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="*", default=[])
    ap.add_argument("--names", nargs="*", default=[], help="optional labels matching --preds (for tiebreak)")
    ap.add_argument("--from-config", default=None)
    ap.add_argument("--pred-dir", default="results")
    ap.add_argument("--pred-prefix", default="dev_")
    ap.add_argument("--out", required=True)
    ap.add_argument("--score", default=None, help="gold jsonl to score against")
    ap.add_argument("--train-conventions", action="store_true",
                    help="apply train-derived convention rules (needs --input)")
    ap.add_argument("--input", default=None,
                    help="input jsonl with id+user fields (for --train-conventions)")
    ap.add_argument("--no-omit-rule", action="store_true",
                    help="with --train-conventions: apply the year rule only (hedge "
                         "against the v1.4 ungrounded-arg gold policy not holding on blind)")
    ap.add_argument("--no-year-rule", action="store_true",
                    help="with --train-conventions: apply the omit rule only")
    args = ap.parse_args()

    files, names, order = [], [], []
    if args.from_config:
        cfg = json.load(open(args.from_config))
        order = cfg.get("tiebreak_order", [])
        name2file = {
            "v7b": "dev_qwen7b_v7b.jsonl", "v7": "dev_qwen7b_v7.jsonl",
            "v8": "dev_qwen7b_v8.jsonl", "v9": "dev_allam_v9.jsonl",
            "v10": "dev_allam_v10.jsonl", "v11": "dev_silma_v11.jsonl",
            "v12": "dev_allam_v12.jsonl",
            "aC": "dev_allam_clean.jsonl", "qC": "dev_qwen25_clean.jsonl",
        }
        for m in cfg["models"]:
            n = m["name"]
            files.append(f"{args.pred_dir}/{name2file[n]}")
            names.append(n)
    else:
        files = args.preds
        names = args.names or [f"m{i}" for i in range(len(files))]
        order = names

    M = {n: load(f) for n, f in zip(names, files)}
    all_ids = sorted({i for d in M.values() for i in d})
    voted = []
    for gid in all_ids:
        present = [(n, M[n][gid]) for n in names if gid in M[n]]
        v = vote_one(gid, present, order)
        if v:
            voted.append(v)
    if args.train_conventions:
        if not args.input:
            sys.exit("--train-conventions requires --input <jsonl with id+user>")
        id2user = {}
        for line in open(args.input, encoding="utf-8"):
            line = line.strip()
            if line:
                x = json.loads(line)
                id2user[x["id"]] = x.get("user", "")
        voted = apply_train_conventions(voted, id2user,
                                        year_rule=not args.no_year_rule,
                                        omit_rule=not args.no_omit_rule)
    with open(args.out, "w", encoding="utf-8") as f:
        for v in voted:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    print(f"[vote] {len(names)} models -> {len(voted)} predictions -> {args.out}")

    if args.score:
        from data_loader import load_gold  # noqa
        from eval_lib import evaluate  # noqa
        gold = [json.loads(l) for l in open(args.score, encoding="utf-8")]
        s = evaluate(voted, gold)
        print(f"[score] OverallA={s['overall_a']:.4f}  ArgEM={s['argem']:.4f}  FnAcc={s['fnacc']:.4f}")


if __name__ == "__main__":
    main()
