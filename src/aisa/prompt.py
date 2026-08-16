"""
Shared prompt/format builder for AISA-ArabicFC.

IMPORTANT: training and inference MUST format inputs identically, so both call
build_messages(). The model is taught to emit:

    <think> reasoning </think>
    {"name": "<tool>", "arguments": {...}}

or, when no tool is needed:

    <think> ... </think>
    {"name": "none", "arguments": {}}

We serialize the target arguments with json.dumps(ensure_ascii=False), which
reproduces the gold's exact rendering (floats as '3.0', Arabic kept as-is), so
the model imitates the gold canonicalization end-to-end and the parsed output
round-trips through the official str()-based ArgEM.
"""
from __future__ import annotations

import json
import re

SYSTEM_TMPL = """أنت مساعد ذكي متخصص في استدعاء الأدوات (function calling) باللغة العربية وبجميع اللهجات.

الأدوات المتاحة (بصيغة JSON):
{tools_json}
{context}
التعليمات:
- اكتب أولاً تفكيرك المختصر بين الوسمين <think> و </think>.
- ثم أعد استدعاء أداة واحدة فقط بصيغة JSON على هذا الشكل بالضبط:
{{"name": "اسم_الأداة", "arguments": {{"المعامل": "القيمة"}}}}
- استخرج قيم المعاملات من رسالة المستخدم. حوّل الأرقام العربية إلى إنجليزية، وطابق صيغة القيم المتوقعة.
- إذا لم تكن هناك حاجة لأي أداة، أعد: {{"name": "none", "arguments": {{}}}}"""


def _tools_block(candidate_tools: list[str], registry: dict) -> str:
    tools = []
    for name in candidate_tools:
        schema = registry.get(name)
        if not schema:
            continue
        tools.append({
            "name": name,
            "description": schema.get("description", ""),
            "parameters": schema.get("parameters", {}),
        })
    return json.dumps(tools, ensure_ascii=False, indent=None)


def build_messages(record: dict, registry: dict) -> list[dict]:
    """Return [system, user] chat messages (no assistant turn)."""
    ctx = (record.get("context") or "").strip()
    ctx_line = f"\nالسياق: {ctx}\n" if ctx else "\n"
    system = SYSTEM_TMPL.format(
        tools_json=_tools_block(record.get("candidate_tools", []), registry),
        context=ctx_line,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": record.get("user", "")},
    ]


def build_target(record: dict, *, synth_think: bool = True) -> str:
    """Assistant target string: <think>..</think>\n{json call}."""
    think = (record.get("gold_think") or "").strip()
    if not think and synth_think:
        think = _synth_think(record)
    call = {"name": record.get("gold_name", "none"), "arguments": record.get("gold_args", {})}
    return f"<think>{think}</think>\n{json.dumps(call, ensure_ascii=False)}"


def _synth_think(record: dict) -> str:
    """Lightweight CoT for rows lacking a gold trace (keeps format + ThinkRate uniform)."""
    name = record.get("gold_name", "none")
    args = record.get("gold_args", {})
    if name == "none":
        return "لا تتطلب رسالة المستخدم استدعاء أي أداة."
    if args:
        kv = "، ".join(f"{k}={v}" for k, v in args.items())
        return f"الأداة المناسبة هي {name}. القيم المستخرجة: {kv}."
    return f"الأداة المناسبة هي {name} دون معاملات."


# ── output parsing (inference side) ──────────────────────────────────────
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_output(text: str) -> dict:
    """Parse a generation into {tool_called, arguments, think} for submission."""
    out = {"tool_called": "none", "arguments": {}, "think": ""}
    if (m := _THINK_RE.search(text)):
        out["think"] = m.group(1).strip()
        text_after = text[m.end():]
    else:
        text_after = text
    # take the LAST json object after think (the call)
    candidates = list(_JSON_RE.finditer(text_after)) or list(_JSON_RE.finditer(text))
    for m in reversed(candidates):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            out["tool_called"] = obj.get("name") or "none"
            a = obj.get("arguments") or {}
            if isinstance(a, dict):
                out["arguments"] = {k: v for k, v in a.items() if v is not None and v != ""}
            break
    return out
