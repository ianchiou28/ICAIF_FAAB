"""工具增强型 LLM 审计员（agentic baseline）。

设计动机：单次全轨迹判官在算术、日期比较与跨 span 检索上失败，而 FAAB 的 14 类故障
里相当一部分本质就是这三件事。若不给判官计算器就宣称「确定性控制优于 LLM 审计员」，
比较是不公平的。本模块因此提供当前最强的 LLM 侧对手：

- system prompt 把 §3 的七条控制契约展开为逐项 checklist，与人工契约同一控制面；
- 六个确定性工具承担算术、单位相乘、日期序、跨 span 数值检索、引文包含与哈希链校验；
- 多轮 function calling，模型自行决定查哪一项、如何解读，最后输出一次结构化裁决。

边界：工具只做原语计算与检索，**不返回任何故障类别判断**；选择运行哪项检查、解释
结果、挑出最早决定性原因并映射到故障类/责任工位，全部由模型完成。工具只读
public_trace，不接触 oracle。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from typing import Any, Callable

from .auditors import (AGENTS, DERIVATIONS, FAULT_TYPES, TAXONOMY, VERDICTS,
                       normalize_result)

MAX_ROUNDS = 8

CONTRACT_CHECKLIST = """
Work through this control checklist. Each item names the fields to compare; use the
tools for every arithmetic, date-ordering, containment, or hash claim instead of
judging it by eye. Do not skip an item because it looks fine in the raw JSON.

C1 Provenance: artifacts[i].parent_hashes resolve to the previous artifact hash, and
   every payload hash recomputes. Tool: verify_hash_chain.
C2 Grounding: artifacts[0].payload.quote is a literal substring of
   source.document_excerpt, and that submitted quote supports raw_value as the EPS
   metric. Tools: find_value_in_document, get_field.
   If raw_value appears in some other span but not in the submitted quote, the citation
   is wrong. If raw_value appears in no span at all, the number is unsupported.
C3 Scale: normalized_value equals raw_value * unit_multiplier, and the declared unit
   agrees with a per-share figure. Tools: multiply, compare_numbers.
C4 Alignment: entity, basis, period, and definition agree between artifacts[0] and
   artifacts[1], and the entity matches source.ticker. Tool: get_field.
C5 Point-in-time: artifacts[1].payload.visible_at is at or before
   artifacts[3].payload.decision_time, and vintage equals latest_eligible_vintage
   (not prior_eligible_vintage). Tool: compare_dates.
C6 Handoff: a qualification raised upstream still has a live reference downstream, and
   artifacts[3].payload keeps its declared source and qualification references
   non-empty. A downstream value that differs from its parent needs an override_reason.
   Tools: get_field, compare_numbers.
C7 Settlement: artifacts[4].payload.ticker equals source.ticker (benchmark_ticker is a
   comparison benchmark, not the target) and horizon_td equals source.control_horizon_td.
   Tools: get_field, compare_numbers.
C8 Role: no artifact asserts a value outside its authority. The evidence schema carries
   a recommendation slot that must stay empty; a null, absent, or empty value is correct
   and only a non-empty recommendation is a violation. Tool: get_field.
"""

TOOL_SYSTEM = """You are an internal auditor of a financial multi-agent workflow.
You inspect the supplied trace, never a market outcome. At most one fault is present.
Report the earliest decisive cause: if a unit error at step 1 propagates into a wrong
surprise grade at step 5, the answer is step 1, not step 5.

The trace has five ordered artifacts: evidence (step 1), expectation (step 2),
consistency (step 3), decision (step 4), settlement (step 5).

Treat source.period, latest_eligible_vintage, decision_time, and control_horizon_td as
verified ledger/control fields unless the trace itself directly contradicts them. A
large earnings surprise is not a fault. The expectation payload comes from a separate
point-in-time I/B/E/S ledger, so its estimate is not expected to appear in the SEC
document excerpt; never reject an estimate merely because the filing does not quote it.
A quote supports a scalar when it explicitly labels that scalar, even if it also
reports a prior-period comparator. Do not invent missing information or use outside
knowledge. A clean, internally consistent trace must pass.
""" + CONTRACT_CHECKLIST + DERIVATIONS + TAXONOMY + """
Call tools until every checklist item is settled, then stop calling tools. You have at
most %d tool rounds. When asked for the final answer, output exactly one JSON object.
""" % MAX_ROUNDS

FINAL_INSTRUCTION = """Tool phase is over. Output exactly one JSON object with these
fields and no other text:
  "verdict": one of %s
  "fault_type": one of %s
  "responsible_agent": one of %s
  "decisive_step": integer 0-5 (0 when there is no fault)
  "materiality": one of ["none","low","medium","high"]
  "evidence_refs": array of strings (artifact hashes or field paths you relied on)
  "rationale": short string naming the checklist item that failed
Use verdict "pass" and fault_type "none" when every control holds.""" % (
    json.dumps(VERDICTS), json.dumps(FAULT_TYPES), json.dumps(AGENTS))


# --- 确定性工具实现：只读 public_trace，只做原语计算 ---------------------------

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUM_RE.findall(text or "")]


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def _spans(excerpt: str) -> list[str]:
    return [s.strip() for s in re.split(r"\n\n+|(?<=[.!?])\s+", excerpt or "") if s.strip()]


def _parse_dt(value: str) -> dt.datetime:
    text = str(value).strip()
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return dt.datetime.fromisoformat(text[:10])


def _aware(value: dt.datetime) -> dt.datetime:
    return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)


def _hash_payload(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _resolve(root: Any, path: str) -> Any:
    node = root
    for part in re.split(r"[.\[\]]+", path.strip()):
        if not part:
            continue
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict):
            if part not in node:
                raise KeyError(part)
            node = node[part]
        else:
            raise KeyError(part)
    return node


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_field",
            "description": (
                "Read one field from the trace by dotted path, e.g. "
                "'artifacts.1.payload.vintage' or 'source.ticker'. Returns the raw "
                "value, or the whole object when the path names one."),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_numbers",
            "description": (
                "Compare two numbers with relative tolerance. Returns whether they are "
                "equal, their order, and the absolute and relative difference."),
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "rel_tol": {"type": "number",
                                "description": "default 1e-9"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Return the exact product a * b. Use for unit-scale checks.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_dates",
            "description": (
                "Compare two ISO timestamps or dates. Returns 'before', 'after', or "
                "'equal' for a relative to b, plus the difference in days."),
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_value_in_document",
            "description": (
                "Search source.document_excerpt for spans whose numbers equal the given "
                "value. Returns the matching spans, whether the submitted evidence quote "
                "is one of them, and whether the submitted quote occurs verbatim in the "
                "document."),
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_hash_chain",
            "description": (
                "Recompute every artifact payload hash and check that declared "
                "parent_hashes resolve to the preceding artifact. Returns per-artifact "
                "status."),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class ToolRuntime:
    """把工具调用落到具体 trace 上；调用计数用于诊断，不进入指标。"""

    def __init__(self, public_case: dict[str, Any]):
        self.trace = public_case["trace"]
        self.calls: list[str] = []

    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(name)
        try:
            return getattr(self, f"_{name}")(**args)
        except AttributeError:
            return {"error": f"unknown tool {name}"}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _get_field(self, path: str) -> dict[str, Any]:
        try:
            value = _resolve(self.trace, path)
        except (KeyError, IndexError, ValueError):
            return {"path": path, "found": False,
                    "error": "no such path; top-level keys are "
                             + json.dumps(sorted(self.trace))}
        return {"path": path, "found": True, "value": value}

    def _compare_numbers(self, a: float, b: float, rel_tol: float = 1e-9) -> dict[str, Any]:
        a, b = float(a), float(b)
        equal = math.isclose(a, b, rel_tol=float(rel_tol), abs_tol=1e-12)
        return {
            "a": a, "b": b, "rel_tol": float(rel_tol), "equal": equal,
            "order": "equal" if equal else ("a_greater" if a > b else "a_less"),
            "abs_diff": abs(a - b),
            "rel_diff": abs(a - b) / abs(b) if b else None,
        }

    def _multiply(self, a: float, b: float) -> dict[str, Any]:
        return {"a": float(a), "b": float(b), "product": float(a) * float(b)}

    def _compare_dates(self, a: str, b: str) -> dict[str, Any]:
        da, db = _aware(_parse_dt(a)), _aware(_parse_dt(b))
        delta = (da - db).total_seconds()
        return {
            "a": a, "b": b,
            "relation": "equal" if delta == 0 else ("after" if delta > 0 else "before"),
            "a_minus_b_days": round(delta / 86400.0, 4),
        }

    def _find_value_in_document(self, value: float) -> dict[str, Any]:
        target = float(value)
        excerpt = self.trace["source"]["document_excerpt"]
        quote = self.trace["artifacts"][0]["payload"].get("quote", "")
        matches = [s for s in _spans(excerpt)
                   if any(_close(target, n) for n in _numbers(s))]
        in_quote = any(_close(target, n) for n in _numbers(quote))
        return {
            "value": target,
            "n_matching_spans": len(matches),
            "matching_spans": [s[:400] for s in matches[:5]],
            "submitted_quote_contains_value": in_quote,
            "submitted_quote_is_verbatim_in_document": quote in excerpt,
            "submitted_quote": quote[:400],
        }

    def _verify_hash_chain(self) -> dict[str, Any]:
        artifacts = self.trace["artifacts"]
        rows = []
        for i, art in enumerate(artifacts):
            recomputed = _hash_payload({k: v for k, v in art.items() if k != "hash"})
            expected_parent = artifacts[i - 1]["hash"] if i else None
            rows.append({
                "index": i,
                "agent": art.get("agent"),
                "hash_recomputes": recomputed == art.get("hash"),
                "declared_parents": art.get("parent_hashes"),
                "parent_resolves": (expected_parent is None
                                    or expected_parent in (art.get("parent_hashes") or [])),
            })
        return {"artifacts": rows,
                "all_ok": all(r["hash_recomputes"] and r["parent_resolves"] for r in rows)}


class ToolAugmentedAuditor:
    """多轮工具增强审计员。

    会话历史的 provider 差异（chat.completions 的 tool 消息 vs responses 的
    function_call_output 条目）由注入的 model 对象负责，本类只管审计逻辑。
    """

    name = "tool_auditor"

    def __init__(self, chat: Callable[..., Any], model: Any,
                 max_rounds: int = MAX_ROUNDS):
        self.chat = chat
        self.model = model
        self.max_rounds = max_rounds

    def audit(self, public_case: dict[str, Any]) -> dict[str, Any]:
        runtime = ToolRuntime(public_case)
        conversation = self.model.seed(
            TOOL_SYSTEM, json.dumps(public_case, ensure_ascii=False, sort_keys=True))
        rounds = 0
        for _ in range(self.max_rounds):
            reply, _usage = self.chat(conversation, tools=TOOL_SPECS, max_tokens=1600,
                                      purpose="tool_round")
            if reply is None:
                break
            rounds += 1
            self.model.add_reply(conversation, reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                try:
                    args = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = runtime.run(call.name, args if isinstance(args, dict) else {})
                self.model.add_tool_result(
                    conversation, call, json.dumps(result, ensure_ascii=False)[:4000])
        self.model.add_user(conversation, FINAL_INSTRUCTION)
        reply, _usage = self.chat(conversation, tools=None, max_tokens=900,
                                  json_object=True, purpose="tool_final")
        payload = getattr(reply, "parsed_json", None) if reply is not None else None
        result = normalize_result(payload if isinstance(payload, dict) else None)
        result["tool_calls_made"] = len(runtime.calls)
        result["tool_rounds"] = rounds
        return result
