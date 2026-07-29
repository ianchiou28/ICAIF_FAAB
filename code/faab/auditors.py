"""FAAB 审计器与统一输出契约。"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Callable

from .benchmark import FAULT_SPECS

FAULT_TYPES = ["none", *FAULT_SPECS]
AGENTS = ["none", "evidence_agent", "expectation_agent", "consistency_agent",
          "decision_agent", "settlement_agent"]
VERDICTS = ["pass", "qualified", "reject", "unresolved"]


def _num_in_quote(value: float, quote: str) -> bool:
    vals = [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*(?:\.\d+)?", quote)]
    return any(math.isclose(float(value), v, rel_tol=1e-6, abs_tol=1e-6) for v in vals)


def finding(fault_type: str, agent: str, step: int, evidence: str,
            verdict: str = "reject", rationale: str = "") -> dict[str, Any]:
    return {"verdict": verdict, "fault_type": fault_type, "responsible_agent": agent,
            "decisive_step": step, "materiality": FAULT_SPECS.get(
                fault_type, {}).get("materiality", "none"),
            "evidence_refs": [evidence] if evidence else [],
            "rationale": rationale}


def pass_result(rationale: str = "") -> dict[str, Any]:
    return finding("none", "none", 0, "", "pass", rationale)


def normalize_result(result: dict[str, Any] | None) -> dict[str, Any]:
    r = result or {}
    verdict = str(r.get("verdict", "unresolved")).lower()
    fault = str(r.get("fault_type", "none")).lower()
    agent = str(r.get("responsible_agent", "none")).lower()
    try:
        step = int(r.get("decisive_step", 0))
    except (TypeError, ValueError):
        step = 0
    if verdict not in VERDICTS:
        verdict = "unresolved"
    if fault not in FAULT_TYPES:
        fault = "none" if verdict == "pass" else "none"
        verdict = "unresolved" if verdict != "pass" else verdict
    if agent not in AGENTS:
        agent = "none"
    refs = r.get("evidence_refs", [])
    if not isinstance(refs, list):
        refs = [str(refs)]
    return {"verdict": verdict, "fault_type": fault, "responsible_agent": agent,
            "decisive_step": step,
            "materiality": str(r.get("materiality", "none")).lower(),
            "evidence_refs": [str(x) for x in refs[:5]],
            "rationale": str(r.get("rationale", ""))[:1000]}


class RuleAuditor:
    """只看公开 trace 的确定性控制；不访问 clean pair 或 oracle。"""

    name = "rule_only"

    def audit(self, public_case: dict[str, Any]) -> dict[str, Any]:
        t = public_case["trace"]
        src = t["source"]
        arts = t["artifacts"]
        e, x, c, d, s = [a["payload"] for a in arts]

        if e.get("recommendation") not in (None, ""):
            return finding("role_violation", "evidence_agent", 1, arts[0]["hash"],
                           rationale="Evidence role emitted a forbidden recommendation.")
        if e["entity"] != src["ticker"]:
            return finding("entity_confusion", "evidence_agent", 1, arts[0]["hash"])
        if e["quote"] not in src["document_excerpt"]:
            return finding("quote_mismatch", "evidence_agent", 1, arts[0]["hash"])
        if not _num_in_quote(float(e["raw_value"]), e["quote"]):
            # 公开轨迹不足以区分“数字编造”和“引文错配”；规则基线固定猜前者。
            return finding("fabricated_value", "evidence_agent", 1, arts[0]["hash"])
        expected_norm = float(e["raw_value"]) * float(e["unit_multiplier"])
        if not math.isclose(expected_norm, float(e["normalized_value"]), rel_tol=1e-9):
            return finding("unit_scaling", "evidence_agent", 1, arts[0]["hash"])
        if float(e["unit_multiplier"]) != 1.0:
            return finding("unit_scaling", "evidence_agent", 1, arts[0]["hash"])
        if x["entity"] != e["entity"]:
            return finding("entity_confusion", "expectation_agent", 2, arts[1]["hash"])
        if x["basis"] != e["basis"]:
            return finding("basis_conflict", "expectation_agent", 2, arts[1]["hash"])
        if x["period"] != e["period"]:
            return finding("period_mismatch", "expectation_agent", 2, arts[1]["hash"])
        if x["definition"] != e["definition"]:
            return finding("definition_conflict", "expectation_agent", 2, arts[1]["hash"])
        if x["visible_at"] > d["decision_time"]:
            return finding("future_leakage", "expectation_agent", 2, arts[1]["hash"])
        if x["vintage"] != x["latest_eligible_vintage"]:
            return finding("stale_expectation", "expectation_agent", 2, arts[1]["hash"])
        if c["qualification"] and c["actual_ref"] not in (arts[0]["hash"],):
            return finding("missing_handoff", "consistency_agent", 3, arts[2]["hash"])
        if c["qualification"] and arts[2]["hash"] not in d["qualification_refs"]:
            return finding("missing_handoff", "decision_agent", 4, arts[3]["hash"])
        if not math.isclose(float(d["value_used"]), float(c["normalized_value"]), rel_tol=1e-9):
            if not d.get("override_reason"):
                return finding("silent_overwrite", "decision_agent", 4, arts[3]["hash"])
        if s["ticker"] != src["ticker"]:
            return finding("wrong_security", "settlement_agent", 5, arts[4]["hash"])
        if int(s["horizon_td"]) != int(src["control_horizon_td"]):
            return finding("wrong_window", "settlement_agent", 5, arts[4]["hash"])
        return pass_result("All deterministic controls passed.")


def llm_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {
        "verdict": {"enum": VERDICTS},
        "fault_type": {"enum": FAULT_TYPES},
        "responsible_agent": {"enum": AGENTS},
        "decisive_step": {"type": "integer"},
        "materiality": {"enum": ["none", "low", "medium", "high"]},
        "evidence_refs": {"type": "array"},
        "rationale": {"type": "string"},
    }}


PROMPT_VERSION = "faab-prompts-1.5"

# v1.5 新增：声明账本自身的推导式。v1.4 的干净轨迹用了一个 trace 里没有出现的
# 离散度下限，200 条干净轨迹里 147 条经不起朴素 (a-mu)/sigma 复算；认真复算的
# 推理型判官因此把干净工作流判成编造。这是可观测性缺陷，不是模型缺陷。
DERIVATIONS = """
Declared ledger derivations. These are part of the control definition; use them and do
not assume a different one:
standardized_surprise = (normalized_value - mean_estimate)
                        / max(dispersion, 0.05*|mean_estimate|, 0.005).
The floor on dispersion is deliberate, so a surprise recomputed from the raw dispersion
alone will disagree with the declared value on many correct traces. Disagreement under
the naive formula is not evidence of a fault.
claim is above_expectations when standardized_surprise > 0 and below_expectations
otherwise.
"""

TAXONOMY = """
Allowed fault types:
fabricated_value: claimed number is not supported anywhere by the relevant evidence.
quote_mismatch: the number may appear elsewhere, but the submitted quote does not support the claim.
unit_scaling: raw value, multiplier, unit, and normalized value are inconsistent.
basis_conflict: GAAP/reported and adjusted/non-GAAP bases are mixed.
period_mismatch: actual and expectation refer to incompatible fiscal periods.
entity_confusion: the wrong company/security is used.
stale_expectation: an older eligible consensus was used instead of the latest pre-event vintage.
future_leakage: an input became visible after the decision time.
definition_conflict: basic/diluted/adjusted metric definitions conflict.
missing_handoff: a required qualification/reference disappears downstream.
role_violation: an agent emits a field outside its authorized role.
silent_overwrite: a downstream value changes without an explicit override reason.
wrong_security: settlement uses the wrong security.
wrong_window: settlement uses a horizon different from the declared control.
none: no fault is present.
"""

JUDGE_SYSTEM = """You are an internal auditor of a financial multi-agent workflow.
Inspect the supplied trace, not the market outcome. Find at most one earliest decisive fault.
The trace contains five ordered artifacts. Attribute propagated errors to their earliest cause.
Treat source.period, latest_eligible_vintage, decision_time, and control_horizon_td as verified
ledger/control fields unless the trace itself contains a direct contradiction. A large earnings
surprise is not a fault. A pre-event consensus vintage is correct when it equals the supplied
latest_eligible_vintage; do not speculate that an unshown later vintage exists. benchmark_ticker
is the comparison benchmark, not the target security. A quote supports a scalar when it explicitly
labels that scalar, even if it also reports a prior-period comparator.
The expectation payload is imported from a separate point-in-time I/B/E/S ledger, so its estimate
is not expected to appear in the SEC document excerpt. Do not reject an estimate merely because
the filing does not quote it.
Do not invent missing information or use outside knowledge. A clean, internally consistent trace
must pass.
Return exactly one JSON object using the required fields.
""" + DERIVATIONS + TAXONOMY

AUDITOR_A_SYSTEM = """You are the first independent specialist auditor.
Focus on evidence, numbers, units, accounting basis, fiscal period, entity, and consensus vintage.
Use only direct contradictions in the trace. Trust an explicitly supplied latest_eligible_vintage,
and do not label an earnings surprise or a prior-period comparator as a fault. The consensus
estimate comes from a separate I/B/E/S ledger and need not appear in the SEC excerpt.
Return a JSON audit finding. If no fault is supported, return pass/none.
""" + DERIVATIONS + TAXONOMY

AUDITOR_B_SYSTEM = """You are the second independent specialist auditor.
Focus on time visibility, handoffs, role permissions, silent overwrites, security identity,
settlement windows, and provenance. Return a JSON audit finding. If no fault is supported,
return pass/none. benchmark_ticker is a comparison benchmark, not the settlement target.
Treat declared control fields as authoritative unless two supplied fields directly conflict.
""" + DERIVATIONS + TAXONOMY

ADJUDICATOR_SYSTEM = """You are the senior audit adjudicator. You receive the original financial
workflow trace plus two independent audit reports. Decide the earliest supported fault. Do not
average opinions, use outside knowledge, speculate about unshown consensus vintages, or infer from
market outcomes. A finding requires a direct trace contradiction; otherwise pass. benchmark_ticker
is not the target security. I/B/E/S estimates need not appear in the SEC excerpt. Return exactly
one JSON object.
""" + DERIVATIONS + TAXONOMY

HYBRID_SYSTEM = """You are the semantic escalation layer in a hybrid audit. Deterministic controls
found no conclusive structural violation or found an ambiguous evidence-grounding problem.
Use the document excerpt and submitted quote to distinguish fabrication from quote mismatch and
to catch semantic conflicts that exact checks cannot resolve. Treat source.period and consensus
control metadata as verified unless directly contradicted by supplied text. Do not use outside
knowledge, speculate about unshown data, or overturn verified numeric/temporal facts without a
direct textual contradiction. A labeled scalar remains supported when the same quote also contains
a prior-period comparator. The consensus estimate is from a separate I/B/E/S ledger and must not
be rejected merely because it is absent from the SEC excerpt or far from actual earnings.

Evidence-grounding decision rule:
1. The submitted artifacts[0].payload.quote—not another sentence in document_excerpt—is the
   evidence agent's citation.
2. If that submitted quote does not support raw_value, but another passage in document_excerpt
   supports it, label quote_mismatch. Do not pass merely because the correct value appears elsewhere.
3. If raw_value is unsupported throughout the relevant document excerpt, label fabricated_value.
4. If the deterministic report is pass and the submitted quote directly supports raw_value and its
   EPS meaning, return pass/none.
Return exactly one JSON object.
""" + DERIVATIONS + TAXONOMY


def case_prompt(public_case: dict[str, Any]) -> str:
    return json.dumps(public_case, ensure_ascii=False, sort_keys=True)


class LLMJudgeAuditor:
    name = "llm_judge"

    def __init__(self, ask: Callable[[str, str, dict], dict]):
        self.ask = ask

    def audit(self, public_case: dict[str, Any]) -> dict[str, Any]:
        return normalize_result(self.ask(JUDGE_SYSTEM, case_prompt(public_case), llm_schema()))


class DebateAuditor:
    name = "debate"

    def __init__(self, ask: Callable[[str, str, dict], dict]):
        self.ask = ask

    def audit(self, public_case: dict[str, Any]) -> dict[str, Any]:
        prompt = case_prompt(public_case)
        a = normalize_result(self.ask(AUDITOR_A_SYSTEM, prompt, llm_schema()))
        b = normalize_result(self.ask(AUDITOR_B_SYSTEM, prompt, llm_schema()))
        user = prompt + "\n\nAUDITOR_A:\n" + json.dumps(a) + "\n\nAUDITOR_B:\n" + json.dumps(b)
        return normalize_result(self.ask(ADJUDICATOR_SYSTEM, user, llm_schema()))


class RulesRetrievalAuditor:
    """规则 + 确定性跨 span 检索，零模型调用。

    这是 hybrid 的确定性内核。单独成类是为了让「残余 LLM 复核是否有增益」成为一条
    可测量的消融，而不是脚注里的断言。
    """

    name = "rules_retrieval"

    def __init__(self) -> None:
        self.rules = RuleAuditor()

    def audit(self, public_case: dict[str, Any]) -> dict[str, Any]:
        r = self.rules.audit(public_case)
        # 精确控制已经给出唯一根因时直接阻断；证据接地类需跨 span 消歧。
        if r["fault_type"] not in ("none", "fabricated_value", "quote_mismatch"):
            return r
        if r["fault_type"] == "none":
            return r

        trace = public_case["trace"]
        evidence = trace["artifacts"][0]["payload"]
        excerpt = trace["source"]["document_excerpt"]
        candidate_spans = [
            span.strip()
            for span in re.split(r"\n\n+|(?<=[.!?])\s+", excerpt)
            if span.strip() and span.strip() != evidence["quote"].strip()
            and _num_in_quote(float(evidence["raw_value"]), span)
        ]
        # submitted quote 已确定不支持 claimed value；同源其他 span 有该值则是错引，
        # 否则是编造。该定义可执行，不再让 LLM 覆盖确定性根因。
        fault = "quote_mismatch" if candidate_spans else "fabricated_value"
        return finding(
            fault, "evidence_agent", 1, trace["artifacts"][0]["hash"],
            rationale=("Claimed value found in another source span."
                       if candidate_spans else "Claimed value absent from all source spans."),
        )


class HybridAuditor(RulesRetrievalAuditor):
    """确定性内核 + 残余 LLM 复核：只有内核判 pass 时才升级到模型。"""

    name = "hybrid"

    def __init__(self, ask: Callable[[str, str, dict], dict]):
        super().__init__()
        self.ask = ask

    def audit(self, public_case: dict[str, Any]) -> dict[str, Any]:
        r = super().audit(public_case)
        if r["fault_type"] != "none":
            return r
        # 内核判 pass 时 r 就是规则报告本身，与 v1.4 冻结运行的 prompt 逐字一致。
        user = case_prompt(public_case) + "\n\nDETERMINISTIC_REPORT:\n" + json.dumps(r)
        return normalize_result(self.ask(HYBRID_SYSTEM, user, llm_schema()))
