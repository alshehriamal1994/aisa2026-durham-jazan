"""
Read a few diverse examples from the AISA-ArabicFC shared-task data
and write them out in a human-readable form so we can SEE the task.

Output: notes/data_examples.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw/aisa-arabicfc-sharedtask/data")
OUT_PATH = Path("notes/data_examples.md")


def fmt_tools(tools) -> str:
    """One tool per line: name + Arabic description + arg names."""
    if tools is None:
        return "_(none)_"
    lines = []
    for t in tools:
        if not isinstance(t, dict) or "function" not in t:
            continue
        fn = t["function"]
        name = fn.get("name", "?")
        desc = fn.get("description", "")
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        args = []
        for k, v in props.items():
            if v is None:
                continue
            d = v.get("description") if isinstance(v, dict) else ""
            args.append(f"`{k}` ({d})" if d else f"`{k}`")
        lines.append(f"- **{name}** — {desc}\n  args: {', '.join(args) if args else '_(none)_'}")
    return "\n".join(lines)


def fmt_messages(msgs) -> str:
    """Render the message list in human-readable Markdown."""
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        think = (m.get("think") or "").strip() if m.get("think") else ""
        tool_calls = m.get("tool_calls")
        out.append(f"**{role}**:")
        if content:
            out.append(f"> {content}")
        if think:
            out.append(f"_think (Arabic reasoning):_ {think}")
        if tool_calls is not None and len(tool_calls) > 0:
            for tc in tool_calls:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "?")
                args = fn.get("arguments") or {}
                args_clean = {k: v for k, v in args.items() if v is not None and v != ""}
                out.append(f"_call:_ `{name}({json.dumps(args_clean, ensure_ascii=False)})`")
        out.append("")
    return "\n".join(out)


def write_example(f, title: str, row: pd.Series) -> None:
    f.write(f"## {title}\n\n")
    f.write(f"- **dialect**: `{row['dialect']}`  ")
    f.write(f"- **requires_function**: `{row['requires_function']}`  ")
    f.write(f"- **gold tool**: `{row['tool_called']}`  ")
    if row.get("negative_category") and not pd.isna(row.get("negative_category")):
        f.write(f"- **negative_category**: `{row['negative_category']}`  ")
    f.write("\n\n")

    f.write("### Candidate tools shown to the model (tools_sampled)\n\n")
    f.write(fmt_tools(row["tools_sampled"]) + "\n\n")

    f.write("### Conversation (gold)\n\n")
    f.write(fmt_messages(row["messages"]) + "\n")

    f.write("---\n\n")


def main() -> None:
    train = pd.read_parquet(DATA_DIR / "train-00000-of-00001.parquet")
    dev = pd.read_parquet(DATA_DIR / "dev-00000-of-00001.parquet")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        f.write("# AISA-ArabicFC — concrete examples\n\n")
        f.write(f"Source: train ({len(train):,}) + dev ({len(dev):,}). "
                "Examples are chosen to span dialects, tools, and the negative categories.\n\n")
        f.write("---\n\n")

        # 1. One positive per dialect
        f.write("# 1. One positive example per dialect\n\n")
        for dialect in ["msa", "gulf", "egyptian", "levantine", "maghrebi"]:
            sub = train[(train["dialect"] == dialect) & (train["requires_function"])]
            if len(sub) == 0:
                f.write(f"## {dialect.upper()} — no rows\n\n---\n\n")
                continue
            row = sub.iloc[0]
            write_example(f, f"{dialect.upper()} → `{row['tool_called']}`", row)

        # 2. Legal-adjacent tool examples
        f.write("# 2. Legal-adjacent tool examples (PhD-relevant)\n\n")
        legal_tools = [
            "calculate_zakat",
            "check_iqama_status",
            "check_visa_status",
            "check_traffic_violations",
            "calculate_end_of_service",
            "calculate_customs",
        ]
        for tool in legal_tools:
            sub = train[train["tool_called"] == tool]
            f.write(f"## Tool: `{tool}` — {len(sub):,} examples in train\n\n")
            if len(sub) > 0:
                row = sub.iloc[0]
                write_example(f, f"Example: {tool} ({row['dialect']})", row)
            else:
                f.write("_(no examples)_\n\n---\n\n")

        # 3. One example per negative category
        f.write("# 3. Negative examples (no tool should be called)\n\n")
        negs = train[~train["requires_function"]]
        for cat in negs["negative_category"].dropna().unique():
            sub = negs[negs["negative_category"] == cat]
            f.write(f"## Negative category: `{cat}` — {len(sub):,} in train\n\n")
            if len(sub) > 0:
                write_example(f, f"Negative: {cat}", sub.iloc[0])

        # 4. The raw `text` field — what the model actually sees
        f.write("# 4. What the model actually sees (`text` field — pre-formatted prompt)\n\n")
        f.write("This is the exact input the baseline expects. It uses Gemma 3 chat turns plus a custom FunctionGemma DSL for tools (NOT JSON).\n\n")
        sample = train.iloc[7]
        f.write(f"- **dialect**: `{sample['dialect']}` · **gold tool**: `{sample['tool_called']}`\n\n")
        f.write("```\n")
        # Truncate if very long
        t = sample["text"]
        f.write(t[:5000] + ("\n... [truncated] ..." if len(t) > 5000 else ""))
        f.write("\n```\n\n")

    print(f"Wrote: {OUT_PATH}")
    print(f"  size: {OUT_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
