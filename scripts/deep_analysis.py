"""
Deeper analysis of the AISA-ArabicFC training data.
Produces notes/data_deep_dive.md with the stats and patterns
that actually matter for strategy: tool distribution, argument
patterns, dialect coverage per tool, query lengths, and a
close look at the negatives.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw/aisa-arabicfc-sharedtask/data")
OUT_PATH = Path("notes/data_deep_dive.md")


def extract_args(messages) -> dict:
    """Pull the gold tool_calls arguments dict from messages."""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            tcs = m.get("tool_calls")
            if tcs is not None and len(tcs) > 0:
                fn = tcs[0].get("function") or {}
                args = fn.get("arguments") or {}
                return {k: v for k, v in args.items() if v is not None and v != ""}
    return {}


def extract_user_text(messages) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content", "") or ""
    return ""


def is_arabic_str(s: str) -> bool:
    """Does the string contain Arabic characters?"""
    if not isinstance(s, str):
        return False
    return any("؀" <= ch <= "ۿ" for ch in s)


def classify_value(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        if not v:
            return "empty_str"
        if is_arabic_str(v):
            return "arabic_str"
        try:
            float(v)
            return "numeric_str"
        except ValueError:
            pass
        return "english_str"
    if isinstance(v, (list, tuple)):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return "other"


def main() -> None:
    train = pd.read_parquet(DATA_DIR / "train-00000-of-00001.parquet")
    dev = pd.read_parquet(DATA_DIR / "dev-00000-of-00001.parquet")
    OUT_PATH.parent.mkdir(exist_ok=True)

    train["gold_args"] = train["messages"].apply(extract_args)
    train["user_text"] = train["messages"].apply(extract_user_text)
    train["user_len"] = train["user_text"].str.len()

    with OUT_PATH.open("w", encoding="utf-8") as f:
        f.write("# AISA-ArabicFC — deep dive into the data\n\n")
        f.write(f"Train: {len(train):,} · Dev: {len(dev):,}\n\n---\n\n")

        # 1. Tool counts
        f.write("## 1. Tool call distribution (train)\n\n")
        tcounts = train["tool_called"].value_counts()
        f.write("| Tool | Count | % of train |\n|---|---:|---:|\n")
        for tool, n in tcounts.items():
            pct = 100 * n / len(train)
            f.write(f"| `{tool}` | {n:,} | {pct:.1f}% |\n")
        f.write("\n")

        # 2. Tool × dialect cross-tab
        f.write("## 2. Tool × dialect — does every tool see every dialect?\n\n")
        pivot = train.pivot_table(
            index="tool_called", columns="dialect", values="user_text",
            aggfunc="count", fill_value=0
        )
        dialects = ["msa", "gulf", "egyptian", "levantine", "maghrebi"]
        cols = [d for d in dialects if d in pivot.columns]
        f.write("| Tool | " + " | ".join(cols) + " |\n")
        f.write("|---|" + "|".join(["---:"] * len(cols)) + "|\n")
        for tool, row in pivot.iterrows():
            f.write(f"| `{tool}` | " + " | ".join(str(row[c]) for c in cols) + " |\n")
        f.write("\n_Cells with low counts in a dialect = the model will struggle there._\n\n")

        # 3. Argument patterns per tool
        f.write("## 3. Argument-key coverage per tool\n\n")
        f.write("Which argument keys does each tool's gold call actually use? "
                "An argument key listed in the schema but never used in gold calls is dead weight.\n\n")
        for tool in tcounts.index:
            if tool == "none":
                continue
            sub = train[train["tool_called"] == tool]
            arg_counts = Counter()
            for args in sub["gold_args"]:
                arg_counts.update(args.keys())
            n = len(sub)
            if n == 0:
                continue
            f.write(f"### `{tool}` ({n:,} calls)\n\n")
            for key, c in arg_counts.most_common():
                pct = 100 * c / n
                f.write(f"- `{key}` — used {c}/{n} times ({pct:.0f}%)\n")
            f.write("\n")

        # 4. Argument value types
        f.write("## 4. Argument value types overall\n\n")
        type_counts = Counter()
        for args in train["gold_args"]:
            for v in args.values():
                type_counts[classify_value(v)] += 1
        total_args = sum(type_counts.values())
        f.write(f"Total argument values across all gold calls: **{total_args:,}**\n\n")
        f.write("| Type | Count | % |\n|---|---:|---:|\n")
        for t, c in type_counts.most_common():
            f.write(f"| `{t}` | {c:,} | {100*c/total_args:.1f}% |\n")
        f.write("\n_Why this matters: ArgEM is strict. If the model outputs `7` and the gold is `7.0`, it counts as wrong._\n\n")

        # 5. Arabic vs English string-arg breakdown
        f.write("## 5. When does the model translate, when does it keep Arabic?\n\n")
        f.write("For each tool, of its string-typed arg values: how many are Arabic vs English?\n\n")
        f.write("| Tool | Arabic strings | English strings | Mixed/Other |\n|---|---:|---:|---:|\n")
        for tool in tcounts.index:
            if tool == "none":
                continue
            sub = train[train["tool_called"] == tool]
            ar, en, other = 0, 0, 0
            for args in sub["gold_args"]:
                for v in args.values():
                    t = classify_value(v)
                    if t == "arabic_str":
                        ar += 1
                    elif t == "english_str":
                        en += 1
                    elif t in ("numeric_str", "empty_str", "other"):
                        other += 1
            f.write(f"| `{tool}` | {ar} | {en} | {other} |\n")
        f.write("\n_The mix shows where the model has to translate the Arabic query "
                "vs where it should preserve the Arabic._\n\n")

        # 6. Query length stats
        f.write("## 6. Query length distribution\n\n")
        qstats = train["user_len"].describe()
        f.write(f"- mean: {qstats['mean']:.0f} chars\n")
        f.write(f"- median: {qstats['50%']:.0f} chars\n")
        f.write(f"- 90th percentile: {train['user_len'].quantile(0.9):.0f} chars\n")
        f.write(f"- max: {qstats['max']:.0f} chars\n\n")
        f.write("Short, conversational queries (~50-100 chars typical).\n\n")

        # 7. Negatives close-up
        f.write("## 7. Negatives — full close-up\n\n")
        negs = train[~train["requires_function"]]
        f.write(f"Total negatives: **{len(negs)}** out of {len(train):,} ({100*len(negs)/len(train):.1f}%).\n\n")
        f.write("By category:\n\n")
        f.write("| Category | Count | Sample query |\n|---|---:|---|\n")
        for cat in sorted(negs["negative_category"].dropna().unique()):
            sub = negs[negs["negative_category"] == cat]
            example = (sub["user_text"].iloc[0] if len(sub) > 0 else "").strip().replace("\n", " ")
            f.write(f"| `{cat}` | {len(sub)} | {example[:120]} |\n")
        f.write("\n")
        f.write("All 50 negatives in train:\n\n")
        for i, (_, row) in enumerate(negs.iterrows(), 1):
            text = row["user_text"].strip().replace("\n", " ")
            cat = row.get("negative_category") or "—"
            dial = row.get("dialect") or "—"
            f.write(f"{i:>2}. [{cat} / {dial}] {text}\n")
        f.write("\n")

        # 8. Dialect × tool: where ArgEM will be hard
        f.write("## 8. Dialect coverage gaps — predict where ArgEM will fail\n\n")
        f.write("For each tool, the dialect with the fewest training examples is the most likely to fail at test time:\n\n")
        f.write("| Tool | Smallest dialect (count) |\n|---|---|\n")
        for tool in tcounts.index:
            if tool == "none":
                continue
            sub = train[train["tool_called"] == tool]
            d_counts = sub["dialect"].value_counts()
            if len(d_counts) == 0:
                continue
            worst = d_counts.idxmin()
            f.write(f"| `{tool}` | `{worst}` ({d_counts[worst]} samples) |\n")
        f.write("\n")

        # 9. Suspicious / hallucinated arg examples (when query has no number but arg does)
        f.write("## 9. Suspicious gold-data examples (potential hallucinations)\n\n")
        f.write("Cases where the user query has no obvious digits but the gold call has a numeric/long string argument. "
                "These suggest the gold ground truth invented data not in the query — a known data-quality issue.\n\n")
        suspicious = []
        for _, row in train.iterrows():
            text = row["user_text"]
            has_digit_in_text = any(ch.isdigit() for ch in text)
            for k, v in row["gold_args"].items():
                if isinstance(v, str) and len(v) >= 6 and any(ch.isdigit() for ch in v):
                    if not has_digit_in_text:
                        suspicious.append({
                            "tool": row["tool_called"],
                            "dialect": row["dialect"],
                            "user": text.strip()[:120],
                            "arg": f"{k}={v}",
                        })
                        break
            if len(suspicious) >= 10:
                break
        for s in suspicious:
            f.write(f"- **[{s['tool']} / {s['dialect']}]** user: \"{s['user']}\"  →  gold: `{s['arg']}`\n")
        f.write(f"\nFound {len(suspicious)} in a quick scan — likely many more across the full data.\n")

    print(f"Wrote: {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
