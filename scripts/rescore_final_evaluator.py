"""Recompute every development-set figure in the system paper under a chosen
scorer and gold release.

    python scripts/rescore_final_evaluator.py --scorer scorer/final_20260724 --gold 35338790
    python scripts/rescore_final_evaluator.py --scorer scorer/submitted_20260623 --gold f43e65a2

The first call reproduces the camera-ready figures: the final pinned evaluator,
which is the leaderboard scorer at its last change (24 July 2026) with data
release v1.6 (dataset commit 35338790, 23 July 2026). The second reproduces the
figures of the submitted version (scorer of 23 June, release v1.4, dataset
commit f43e65a2). `--gold` is either a commit of the TuwaiqAcademy/AISA-ArabicFC
dataset, downloaded on first use, or a local directory holding data/<split>-*.parquet.
Only scoring changes between the two runs: every prediction file is the frozen
file the paper reports. `--compare-gold` (default f43e65a2, or the local
directory data/raw/aisa_v1_4 where present) is the release the row-level
transition in Appendix E is measured against.

Prediction files are looked up in the released layout (predictions/dev,
predictions/shuffled, predictions/permtrain) when it exists, else in the
working layout (results/...).
"""
import argparse, collections, glob, importlib.util, json, os, random, re, sys
from math import comb

import pandas as pd

DATASET = "TuwaiqAcademy/AISA-ArabicFC"


def parquet_path(spec, split):
    """spec: local directory with data/<split>-*.parquet, or a hub commit id."""
    if os.path.isdir(spec):
        return glob.glob(os.path.join(spec, "data", f"{split}-*.parquet"))[0]
    from huggingface_hub import hf_hub_download
    return hf_hub_download(DATASET, f"data/{split}-00000-of-00001.parquet", repo_type="dataset", revision=spec)


ap = argparse.ArgumentParser()
ap.add_argument("--scorer", required=True)
ap.add_argument("--gold", required=True, help="dataset commit id, or a local directory holding data/<split>-*.parquet")
ap.add_argument("--compare-gold", default=None, help="release for the Appendix E transition (default: v1.4)")
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
if A.compare_gold is None:
    A.compare_gold = "data/raw/aisa_v1_4" if os.path.isdir("data/raw/aisa_v1_4") else "f43e65a2"
RELEASE = os.path.isdir("predictions/dev")

sys.path.insert(0, A.scorer)
import normalize as nz          # noqa: E402
import eval_lib                 # noqa: E402

# The combiner clusters values with the normaliser it shipped with (the 23 June
# scorer). Load that copy under the name the combiner imports, so the reproduced
# vote is the submitted system whatever scorer is used for scoring.
SUBMITTED_SCORER = "scorer/submitted_20260623" if RELEASE else "baselines/leaderboard-code-v1_3"
_v13_spec = importlib.util.spec_from_file_location("normalize_v13", SUBMITTED_SCORER + "/normalize.py")
_nz_v13 = importlib.util.module_from_spec(_v13_spec)
_v13_spec.loader.exec_module(_nz_v13)
_saved = sys.modules.get("normalize")
sys.modules["normalize"] = _nz_v13
ev_spec = importlib.util.spec_from_file_location("ev", "scripts/ensemble_vote.py")
ev = importlib.util.module_from_spec(ev_spec)
try:
    ev_spec.loader.exec_module(ev)
except SystemExit:
    pass
sys.modules["normalize"] = _saved
assert ev.canon_value is _nz_v13.canon_value
if not ev._REG:
    ev._REG = json.load(open("scripts/tools_registry.json" if RELEASE else "data/processed_v13/tools_registry.json"))

ORDER = ["v10", "v12", "v7b", "v7", "aC", "qC"]
if RELEASE:
    MEMBER = {m: f"predictions/dev/members/{m}.jsonl" for m in ORDER}
    SHUF = {m: f"predictions/shuffled/{m}_shuffled.jsonl" for m in ORDER}
    V1 = "predictions/dev/V1_year_and_omit.jsonl"
    VARIANT = {"vote+enum": "predictions/dev/V3_no_rules.jsonl", "+year": "predictions/dev/V2_year_only.jsonl",
               "+omit": "predictions/dev/V4_omit_only.jsonl", "+both(V1)": V1}
    PERM = {"allam_c": "predictions/permtrain/allam_perm_canonical.jsonl", "allam_s": "predictions/permtrain/allam_perm_shuffled.jsonl",
            "qwen_c": "predictions/permtrain/qwen25_perm_canonical.jsonl", "qwen_s": "predictions/permtrain/qwen25_perm_shuffled.jsonl"}
    CONTROL = {m: f"predictions/shuffled/{m}_control.jsonl" for m in ORDER}
    BIG = "predictions/dev/qwen25_32b.jsonl"
else:
    MEMBER = {"v10": "results/dev_allam_v10.jsonl", "v12": "results/dev_allam_v12.jsonl",
              "v7b": "results/dev_qwen7b_v7b.jsonl", "v7": "results/dev_qwen7b_v7.jsonl",
              "aC": "results/dev_allam_clean.jsonl", "qC": "results/dev_qwen25_clean.jsonl"}
    SHUF = {m: f"results/shuffle/{m}_shuffled.jsonl" for m in ORDER}
    V1 = "results/dev_vote_strong6_conv_v14.jsonl"
    VARIANT = {"vote+enum": "scratchpad/dev_norules.jsonl", "+year": "scratchpad/dev_yearonly.jsonl",
               "+omit": "scratchpad/dev_omitonly.jsonl", "+both(V1)": V1}
    PERM = {"allam_c": "results/permtrain/allam_perm_canonical.jsonl", "allam_s": "results/permtrain/allam_perm_shuffled.jsonl",
            "qwen_c": "results/permtrain/qwen25_perm_canonical.jsonl", "qwen_s": "results/permtrain/qwen25_perm_shuffled.jsonl"}
    CONTROL = {m: f"results/shuffle/{m}_control.jsonl" for m in ORDER}
    BIG = "results/dev_qwen32b.jsonl"
DATE_KEYS = {"date", "check_in", "check_out", "departure_date", "return_date", "appointment_date"}
CUR_KEYS = {"currency", "from_currency", "to_currency"}


def load_gold(split):
    df = pd.read_parquet(parquet_path(A.gold, split)).reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        args, user = {}, ""
        for m in r["messages"]:
            if m.get("role") == "user":
                user = m.get("content") or ""
            if r["requires_function"] and m.get("role") == "assistant" and m.get("tool_calls") is not None:
                for tc in m["tool_calls"]:
                    args = {k: v for k, v in dict(tc["function"]["arguments"]).items() if v is not None}
        rows.append({"id": i, "dialect": (r["dialect"] or "unknown").lower(),
                     "requires_function": bool(r["requires_function"]),
                     "tool_called": r["tool_called"] if r["requires_function"] else "none",
                     "arguments": args, "user": user,
                     "cands": [t["function"]["name"] for t in r["tools_sampled"]]})
    return rows


def L(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def clean(d):
    return {str(k): str(v).strip() for k, v in (d or {}).items() if v is not None and v != ""}


def row_ok(p, g):
    if not p:
        return False
    return (p.get("tool_called") or "none") == g["tool_called"] and \
        nz.args_match(clean(p.get("arguments")), clean(g["arguments"]), g["tool_called"])


def argem(preds, pos):
    return sum(row_ok(preds.get(g["id"]), g) for g in pos) / len(pos)


def fnacc(preds, gold):
    return sum(((preds.get(g["id"]) or {}).get("tool_called") or "none") == g["tool_called"] for g in gold) / len(gold)


def vec(preds, pos):
    return [int(row_ok(preds.get(g["id"]), g)) for g in pos]


def mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n, k = b01 + b10, min(b01, b10)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return p, b01, b10


def boot_diff(a, b, B=2000):
    rng = random.Random(A.seed)
    n = len(a)
    d = [y - x for x, y in zip(a, b)]
    res = []
    for _ in range(B):
        res.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    res.sort()
    return res[int(0.025 * B)], res[int(0.975 * B) - 1]


def boot_mean(v, B=2000):
    rng = random.Random(A.seed)
    n = len(v)
    res = []
    for _ in range(B):
        res.append(sum(v[rng.randrange(n)] for _ in range(n)) / n)
    res.sort()
    return res[int(0.025 * B)], res[int(0.975 * B) - 1]


def strip_opt(d, tool):
    ign = nz.OPTIONAL_IGNORE.get(tool or "", set())
    return {k: v for k, v in d.items() if k not in ign}


def failing_keys(p, g):
    pa, ga = strip_opt(clean(p.get("arguments")), g["tool_called"]), strip_opt(clean(g["arguments"]), g["tool_called"])
    bad = []
    for k in sorted(set(pa) | set(ga)):
        if k not in pa or k not in ga or not nz.value_match(pa[k], ga[k], k):
            bad.append(k)
    return bad


def vote(members, id2user, order=ORDER, rules=True):
    ids = sorted(set.intersection(*[set(m) for m in members.values()]))
    out = [ev.vote_one(i, [(n, members[n][i]) for n in order if i in members[n]], order) for i in ids]
    if rules:
        out = ev.apply_train_conventions(out, id2user, True, True)
    return {r["id"]: r for r in out}


gold = load_gold("dev")
pos = [g for g in gold if g["requires_function"]]
id2user = {g["id"]: g["user"] for g in gold}
print(f"### scorer={A.scorer}  gold={A.gold}  rows={len(gold)} positives={len(pos)}")

# 1. headline through the official evaluate()
v1 = L(V1)
s = eval_lib.evaluate(list(v1.values()), [{k: g[k] for k in ("id", "tool_called", "arguments", "dialect", "requires_function")} for g in gold])
print(f"\n[headline V1] FnAcc {s['fnacc']:.4f}  ArgEM {s['argem']:.4f}  ThinkRate {s['thinkrate']:.4f}  OverallA {s['overall_a']:.4f}  OverallB {s['overall_b']:.4f}")
assert abs(s["argem"] - argem(v1, pos)) < 1e-9, "row_ok disagrees with eval_lib"

# 1b. where the headline moves between the submitted-version evaluator and this one
try:
    _osp = importlib.util.spec_from_file_location("normalize_old", SUBMITTED_SCORER + "/normalize.py")
    _nz_old = importlib.util.module_from_spec(_osp)
    _osp.loader.exec_module(_nz_old)
except Exception:
    _nz_old = None
if _nz_old is not None:
    import copy as _copy
    def _rows(gv):
        df = pd.read_parquet(parquet_path(gv, "dev")).reset_index(drop=True)
        out = {}
        for i, r in df.iterrows():
            a = {}
            if r["requires_function"]:
                for m in r["messages"]:
                    if m.get("role") == "assistant" and m.get("tool_calls") is not None:
                        for tc in m["tool_calls"]:
                            a = {k: v for k, v in dict(tc["function"]["arguments"]).items() if v is not None}
                out[i] = (r["tool_called"], a)
        return out
    g14, gcur = _rows(A.compare_gold), _rows(A.gold)
    def okw(nzmod, gd, i):
        p = v1[i]
        t, a = gd[i]
        return (p.get("tool_called") or "none") == t and nzmod.args_match(clean(p.get("arguments")), clean(a), t)
    o_old = {i: okw(_nz_old, g14, i) for i in g14}
    o_new = {i: okw(nz, gcur, i) for i in gcur}
    o_gold_only = {i: okw(_nz_old, gcur, i) for i in gcur}     # old scorer, this gold
    o_scorer_only = {i: okw(nz, g14, i) for i in g14}          # this scorer, v1.4 gold
    lost = [i for i in g14 if o_old[i] and not o_new[i]]
    won = [i for i in g14 if not o_old[i] and o_new[i]]
    print(f"[transition submitted scorer + {A.compare_gold} -> this] V1 rows right->wrong {len(lost)}, wrong->right {len(won)}; "
          f"gold change alone: {sum(o_old[i] and not o_gold_only[i] for i in g14)} lost / {sum((not o_old[i]) and o_gold_only[i] for i in g14)} won; "
          f"scorer change alone: {sum(o_old[i] and not o_scorer_only[i] for i in g14)} lost / {sum((not o_old[i]) and o_scorer_only[i] for i in g14)} won")
    lk = collections.Counter(k for i in lost for k in failing_keys(v1[i], {"tool_called": gcur[i][0], "arguments": gcur[i][1]}))
    print(f"   fields on the rows lost: {lk.most_common(6)}")

# 2. members
M = {m: L(f) for m, f in MEMBER.items()}
S = {m: L(f) for m, f in SHUF.items()}
print("\n[members]  tag  ArgEM(canon)  FnAcc(canon)  FnAcc(shuf)  order drop  ArgEM(shuf)")
mem_argem = {}
for m in ORDER:
    a, f, fs = argem(M[m], pos), fnacc(M[m], gold), fnacc(S[m], gold)
    mem_argem[m] = a
    print(f"   {m:4s} {a:.4f}  {f:.4f}  {fs:.4f}  {fs - f:+.4f}  {argem(S[m], pos):.4f}")
best = max(ORDER, key=lambda m: mem_argem[m])
worst = min(mem_argem.values())
print(f"   best member {best} {mem_argem[best]:.4f}; weakest {worst:.4f}; spread {max(mem_argem.values()) - worst:.4f} = {round((max(mem_argem.values()) - worst) * len(pos))} rows")

# 3. Table 3
files = VARIANT
V = {"best": vec(M[best], pos)}
for k, f in files.items():
    V[k] = vec(L(f), pos)
print("\n[Table 3]")
for name, (a, b) in {"vote+enum vs best": ("best", "vote+enum"), "+year vs vote": ("vote+enum", "+year"),
                     "+omit vs vote": ("vote+enum", "+omit"), "+both vs vote": ("vote+enum", "+both(V1)")}.items():
    p, b01, b10 = mcnemar(V[a], V[b])
    lo, hi = boot_diff(V[a], V[b])
    print(f"   {name:18s} {sum(V[b]) / len(pos):.4f} ({sum(V[b])})  diff {sum(V[b]) / len(pos) - sum(V[a]) / len(pos):+.4f}  CI [{lo:+.3f},{hi:+.3f}]  p={p:.3f}  gain {b01} loss {b10}")

# 3b. best member with both rules, union oracle headroom, and the enum-validity rule
import copy as _cp
bm = {i: _cp.deepcopy(r) for i, r in M[best].items()}
bm_rules = {r["id"]: r for r in ev.apply_train_conventions(list(bm.values()), id2user, True, True)}
noenum_reg, ev._REG = ev._REG, {}
v_noenum = vote(M, id2user, rules=False)
v_noenum_rules = vote(M, id2user, rules=True)
ev._REG = noenum_reg
v_enum = vote(M, id2user, rules=False)
print(f"\n[Section 3 paragraph] best member {best} with both rules {argem(bm_rules, pos):.4f} vs vote {argem(v1, pos):.4f}; "
      f"union oracle {1 - len([g for g in pos if not any(row_ok(M[m].get(g['id']), g) for m in ORDER)]) / len(pos):.4f}, "
      f"headroom over vote+enum {round((1 - len([g for g in pos if not any(row_ok(M[m].get(g['id']), g) for m in ORDER)]) / len(pos) - argem(v_enum, pos)) * len(pos))} rows; "
      f"vote without enum {argem(v_noenum, pos):.4f} vs with {argem(v_enum, pos):.4f} (rows differing in correctness {sum(row_ok(v_noenum.get(g['id']), g) != row_ok(v_enum.get(g['id']), g) for g in pos)}); "
      f"with rules {argem(v_noenum_rules, pos):.4f} vs {argem(v1, pos):.4f}")
print(f"   voted dictionary identical to some member's on {sum(any(v_enum[i]['tool_called'] == M[m][i]['tool_called'] and clean(v_enum[i].get('arguments')) == clean(M[m][i].get('arguments')) for m in ORDER) for i in v_enum)} of {len(v_enum)} items")

# 4. tiebreak by measured strength
order2 = sorted(ORDER, key=lambda m: (-mem_argem[m], ORDER.index(m)))
print(f"\n[tiebreak by strength] order {order2}: ArgEM {argem(vote(M, id2user, order2), pos):.4f} (fixed order {argem(vote(M, id2user), pos):.4f})")

# 5. shuffled-order vote
vo, vs = vote(M, id2user), vote(S, id2user)
same = all(clean(vo[i].get("arguments")) == clean(v1[i].get("arguments")) and vo[i]["tool_called"] == v1[i]["tool_called"] for i in v1)
print(f"\n[shuffle] reproduced vote == shipped V1 dictionaries: {same}")
fo, ao, fs_, as_ = fnacc(vo, gold), argem(vo, pos), fnacc(vs, gold), argem(vs, pos)
print(f"   FnAcc {fo:.4f} -> {fs_:.4f} ({fs_ - fo:+.4f})   ArgEM {ao:.4f} -> {as_:.4f} ({as_ - ao:+.4f})")
lost = [g for g in pos if row_ok(vo.get(g["id"]), g) and not row_ok(vs.get(g["id"]), g)]
gained = [g for g in pos if not row_ok(vo.get(g["id"]), g) and row_ok(vs.get(g["id"]), g)]
flip = sum(1 for g in lost if (vs[g["id"]].get("tool_called") or "none") != g["tool_called"])
print(f"   rows lost {len(lost)}, gained {len(gained)}; of lost: tool flipped {flip}, tool kept {len(lost) - flip}")
first_only = {g["id"]: {"tool_called": g["cands"][0] if (vo[g["id"]].get("tool_called") or "none") != "none" else "none", "arguments": {}} for g in gold}
print(f"   'always first candidate' with our call/no-call decision: FnAcc {fnacc(first_only, gold):.4f}")

# 6. permutation retraining
print("\n[permtrain]")
for base, tag, f in [("allam", "canon-trained/canon", CONTROL["aC"]), ("allam", "canon-trained/shuf", SHUF["aC"]),
                     ("allam", "perm-trained/canon", PERM["allam_c"]), ("allam", "perm-trained/shuf", PERM["allam_s"]),
                     ("qwen", "canon-trained/canon", CONTROL["qC"]), ("qwen", "canon-trained/shuf", SHUF["qC"]),
                     ("qwen", "perm-trained/canon", PERM["qwen_c"]), ("qwen", "perm-trained/shuf", PERM["qwen_s"])]:
    P = L(f)
    print(f"   {base:6s} {tag:20s} FnAcc {fnacc(P, gold):.4f}  ArgEM {argem(P, pos):.4f}")

# 7. competence against convention (V1)
fail = [g for g in pos if not row_ok(v1.get(g["id"]), g)]
keyset_raw = sum(set(clean(v1[g["id"]].get("arguments"))) == set(clean(g["arguments"])) for g in pos)
keyset_opt = sum(set(strip_opt(clean(v1[g["id"]].get("arguments")), g["tool_called"])) == set(strip_opt(clean(g["arguments"]), g["tool_called"])) for g in pos)
tool_err = sum((v1[g["id"]].get("tool_called") or "none") != g["tool_called"] for g in pos)
date_rows = [g for g in fail if any(k in DATE_KEYS for k in failing_keys(v1[g["id"]], g))]


def relaxed_ok(p, g):
    pa, ga = strip_opt(clean(p.get("arguments")), g["tool_called"]), strip_opt(clean(g["arguments"]), g["tool_called"])
    if (p.get("tool_called") or "none") != g["tool_called"] or set(pa) != set(ga):
        return False
    return all(k in DATE_KEYS or k in CUR_KEYS or nz.value_match(pa[k], ga[k], k) for k in ga)


relaxed = sum(relaxed_ok(v1[g["id"]], g) for g in pos) / len(pos)
print(f"\n[competence] failing rows {len(fail)}; tool errors among positives {tool_err}; key set equal raw {keyset_raw} / after optional-ignore {keyset_opt}; date field wrong on {len(date_rows)} rows; relaxed {relaxed:.4f} vs strict {argem(v1, pos):.4f}; convention-tax rows {round((relaxed - argem(v1, pos)) * len(pos))}")

# 8. residual
allwrong = [g for g in pos if not any(row_ok(M[m].get(g["id"]), g) for m in ORDER)]
print(f"[residual] all-six-wrong {len(allwrong)}; of V1's {len(fail)} failures on such rows {sum(1 for g in fail if g in allwrong)}; oracle {1 - len(allwrong) / len(pos):.4f}")
fk = collections.Counter()
date_inst = 0
date_rows2 = 0
for g in fail:
    bad = failing_keys(v1[g["id"]], g)
    for k in bad:
        fk[k] += 1
    dk = [k for k in bad if k in DATE_KEYS]
    date_inst += len(dk)
    date_rows2 += bool(dk)
print(f"   failing fields (missing on one side or value mismatch): {fk.most_common(14)}")
fk2 = collections.Counter()
miss = collections.Counter()
for g in fail:
    pa, ga = strip_opt(clean(v1[g["id"]].get("arguments")), g["tool_called"]), strip_opt(clean(g["arguments"]), g["tool_called"])
    for k in sorted(set(pa) | set(ga)):
        if k in pa and k in ga:
            if not nz.value_match(pa[k], ga[k], k): fk2[k] += 1
        else:
            miss[("pred-only" if k in pa else "gold-only", k)] += 1
print(f"   value mismatches on keys present both sides: {fk2.most_common(14)}")
print(f"   keys present on one side only: {miss.most_common(10)}")
print(f"   date family {date_inst} instances over {date_rows2} rows")
tt = sum(1 for g in fail if "termination_type" in clean(v1[g["id"]].get("arguments")) and "termination_type" not in clean(g["arguments"])
         and nz.args_match({k: v for k, v in clean(v1[g["id"]].get("arguments")).items() if k != "termination_type"}, clean(g["arguments"]), g["tool_called"]))
print(f"   termination_type emitted where reference omits it AND row otherwise right: {tt} rows")

# 9. agreement
agree_raw = sum(all(M[m][i]["tool_called"] == M[ORDER[0]][i]["tool_called"] and clean(M[m][i].get("arguments")) == clean(M[ORDER[0]][i].get("arguments")) for m in ORDER) for i in range(len(gold)))
agree_canon = 0
for i in range(len(gold)):
    t0 = M[ORDER[0]][i]["tool_called"]
    a0 = clean(M[ORDER[0]][i].get("arguments"))
    agree_canon += all(M[m][i]["tool_called"] == t0 and nz.args_match(clean(M[m][i].get("arguments")), a0, t0) for m in ORDER)
print(f"[agreement] identical on {agree_raw} items; canonically identical on {agree_canon} of {len(gold)}")

# 10. per dialect (V1)
print("[dialect V1]")
for d in ["msa", "gulf", "egyptian", "levantine", "maghrebi"]:
    v = [int(row_ok(v1.get(g["id"]), g)) for g in pos if g["dialect"] == d]
    lo, hi = boot_mean(v)
    top = collections.Counter(k for g in fail if g["dialect"] == d for k in failing_keys(v1[g["id"]], g)).most_common(3)
    print(f"   {d:10s} {sum(v) / len(v):.3f} [{lo:.3f},{hi:.3f}] n={len(v)}  top failing {top}")

# 11. rule statistics in train, and date surface values
train = load_gold("train")
sh = [str(g["arguments"].get(k)) for g in train if g["tool_called"] == "search_hotels" for k in ("check_in", "check_out") if g["arguments"].get(k) not in (None, "")]
bd = [str(g["arguments"].get("date")) for g in train if g["tool_called"] == "book_doctor_appointment" and g["arguments"].get("date") not in (None, "")]
yr = lambda v: (re.search(r"(20\d\d)", v) or [None, None])[1] if re.search(r"(20\d\d)", v) else None
shy = [yr(v) for v in sh if yr(v)]
bdy = [yr(v) for v in bd if yr(v)]
print(f"\n[rules in train {A.gold}] search_hotels dated fields {len(sh)}, of which carry a year {len(shy)}, carry 2023: {shy.count('2023')}; book_doctor dates {len(bd)}, carry a year {len(bdy)}, carry 2023: {bdy.count('2023')}")
print(f"   recipient_iban kept {sum('recipient_iban' in g['arguments'] for g in train)}, insurance_number kept {sum('insurance_number' in g['arguments'] for g in train)}")
dates = collections.Counter(str(g["arguments"]["date"]) for g in train + pos if g["arguments"].get("date") not in (None, ""))
print(f"   distinct surface values of `date` in train+dev gold: {len(dates)}; distinct after scorer canon: {len({nz.canon_value(v, 'date') for v in dates})}; ISO-form share {sum(c for v, c in dates.items() if re.match(r'^\d{4}-\d\d-\d\d$', v)) / sum(dates.values()):.3f}")
allkeys = collections.Counter(str(g["arguments"][k]) for g in train + pos for k in DATE_KEYS if g["arguments"].get(k) not in (None, ""))
print(f"   distinct surface values over the whole date family: {len(allkeys)}; after canon: {len({nz.canon_value(v, 'date') for v in allkeys})}")

# 12. 32B
try:
    print(f"\n[32B] ArgEM {argem(L(BIG), pos):.4f}")
except FileNotFoundError:
    pass

# 13. convention-tax examples surviving under this scorer
print("\n[convention-tax examples: relaxed-right, strict-wrong]")
n = 0
for g in pos:
    p = v1[g["id"]]
    if relaxed_ok(p, g) and not row_ok(p, g):
        bad = failing_keys(p, g)
        print(f"   id {g['id']} {g['tool_called']} " + "; ".join(f"{k}: pred={clean(p.get('arguments')).get(k)!r} gold={clean(g['arguments']).get(k)!r}" for k in bad))
        n += 1
print(f"   total {n}")
