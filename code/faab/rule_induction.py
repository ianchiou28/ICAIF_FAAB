"""从带标签的 FAAB development 轨迹归纳可执行审计规则。"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Any

from .auditors import normalize_result
from .benchmark import FAULT_SPECS, public_trace
from .metrics import detected, evaluate
from .rule_dsl import (
    PATHS,
    RULESET_VERSION,
    RuleSetAuditor,
    evaluate_condition,
    feature_snapshot,
    validate_rule_set,
)


def _compact_snapshot(public_case: dict[str, Any]) -> dict[str, Any]:
    values = feature_snapshot(public_case)
    # 原文与 hash 本身不帮助归纳通用规则；保留它们之间的关系，降低 prompt 噪声。
    omitted = {
        "source.document_excerpt",
        "evidence.hash",
        "expectation.hash",
        "consistency.hash",
        "consistency.actual_ref",
        "consistency.expectation_ref",
        "decision.source_ref",
        "decision.qualification_refs",
    }
    fields = {
        key: (str(value)[:180] if isinstance(value, str) else value)
        for key, value in values.items() if key not in omitted
    }
    special = {
        op: evaluate_condition({"op": op}, public_case)
        for op in (
            "quote_not_in_document",
            "number_not_in_quote",
            "number_in_other_span",
            "normalization_mismatch",
        )
    }
    relations = {
        "evidence.entity == source.ticker":
            values["evidence.entity"] == values["source.ticker"],
        "expectation.entity == evidence.entity":
            values["expectation.entity"] == values["evidence.entity"],
        "expectation.basis == evidence.basis":
            values["expectation.basis"] == values["evidence.basis"],
        "expectation.period == evidence.period":
            values["expectation.period"] == values["evidence.period"],
        "expectation.definition == evidence.definition":
            values["expectation.definition"] == values["evidence.definition"],
        "expectation.vintage == expectation.latest_eligible_vintage":
            values["expectation.vintage"] == values["expectation.latest_eligible_vintage"],
        "expectation.visible_at > decision.decision_time":
            evaluate_condition(
                {
                    "op": "gt_time",
                    "left": {"path": "expectation.visible_at"},
                    "right": {"path": "decision.decision_time"},
                },
                public_case,
            ),
        "decision.source_ref == consistency.hash":
            values["decision.source_ref"] == values["consistency.hash"],
        "consistency.hash in decision.qualification_refs":
            isinstance(values["decision.qualification_refs"], list)
            and values["consistency.hash"] in values["decision.qualification_refs"],
        "decision.value_used ~= consistency.normalized_value":
            not evaluate_condition(
                {
                    "op": "numeric_not_close",
                    "left": {"path": "decision.value_used"},
                    "right": {"path": "consistency.normalized_value"},
                },
                public_case,
            ),
        "settlement.ticker == source.ticker":
            values["settlement.ticker"] == values["source.ticker"],
        "settlement.horizon_td == source.control_horizon_td":
            values["settlement.horizon_td"] == values["source.control_horizon_td"],
    }
    return {"fields": fields, "special_operators": special, "relations": relations}


def build_training_bundle(
    cases: list[dict[str, Any]],
    examples_per_fault: int = 3,
    clean_examples: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """只从 development split 采样，标签作为离线规则归纳监督信号。"""
    dev = [case for case in cases if case["split"] == "dev"]
    by_fault: dict[str, list[dict[str, Any]]] = defaultdict(list)
    clean_by_source = {
        case["source_id"]: case
        for case in dev if case["oracle"]["fault_type"] == "none"
    }
    for case in dev:
        by_fault[case["oracle"]["fault_type"]].append(case)
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    clean_pool = list(by_fault["none"])
    rng.shuffle(clean_pool)
    for case in clean_pool[:clean_examples]:
        examples.append({
            "label": {
                "fault_type": "none", "responsible_agent": "none", "decisive_step": 0,
            },
            "trace": _compact_snapshot(public_trace(case)),
        })
    for fault in FAULT_SPECS:
        pool = list(by_fault[fault])
        rng.shuffle(pool)
        for case in pool[:examples_per_fault]:
            clean = clean_by_source[case["source_id"]]
            examples.append({
                "label": {
                    "fault_type": fault,
                    "responsible_agent": case["oracle"]["responsible_agent"],
                    "decisive_step": case["oracle"]["decisive_step"],
                },
                "clean_pair": _compact_snapshot(public_trace(clean)),
                "faulty_trace": _compact_snapshot(public_trace(case)),
            })
    rng.shuffle(examples)
    return {
        "training_split": "faab-v1.4-development-only",
        "seed": seed,
        "n_examples": len(examples),
        "examples": examples,
    }


DSL_GUIDE = f"""
Return exactly one JSON object:
{{
  "version": "{RULESET_VERSION}",
  "rules": [
    {{
      "rule_id": "R01",
      "priority": 10,
      "fault_type": "one of the 14 labels",
      "responsible_agent": "evidence_agent|expectation_agent|consistency_agent|decision_agent|settlement_agent",
      "decisive_step": 1,
      "description": "short generic explanation",
      "condition": {{...}}
    }}
  ]
}}

Produce exactly one generic rule for each fault class. IDs, priorities, and fault classes
must be unique. Conditions may use only these operand forms:
  {{"path": "allowed.path"}}
  {{"const": number|null|boolean}}
String constants are forbidden.

Boolean operators:
  {{"op":"all","args":[condition,...]}}
  {{"op":"any","args":[condition,...]}}
  {{"op":"not","arg":condition}}

Comparison operators:
  {{"op":"eq|ne|numeric_not_close|gt_time","left":operand,"right":operand}}
  {{"op":"empty|not_empty","value":operand}}
  {{"op":"not_list_contains","container":operand,"value":operand}}

Financial grounding operators take no arguments:
  {{"op":"quote_not_in_document"}}
  {{"op":"number_not_in_quote"}}
  {{"op":"number_in_other_span"}}
  {{"op":"normalization_mismatch"}}

Allowed paths:
{json.dumps(sorted(PATHS), ensure_ascii=False)}
"""

INDUCTION_SYSTEM = """You compile labeled financial multi-agent audit traces into a
small executable rule set. Infer cross-field invariants that distinguish clean and faulty
examples. Generalize across tickers, dates, values, and source documents. Never encode a
case identifier, ticker literal, date literal, hash literal, or training-specific value.
Use the earliest decisive cause. Do not emit prose outside the JSON object.
""" + DSL_GUIDE


def induction_prompt(bundle: dict[str, Any]) -> str:
    return (
        "Infer the complete executable audit rule set from these labeled development "
        "examples. The examples are the only supervision signal.\n\n"
        + json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    )


def normalize_rule_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict) and "rule_set" in response:
        response = response["rule_set"]
    return validate_rule_set(response)


def evaluate_rules(
    rule_set: dict[str, Any],
    cases: list[dict[str, Any]],
    split: str,
    name: str = "induced_rules",
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    auditor = RuleSetAuditor(rule_set, name)
    predictions = {
        case["case_id"]: normalize_result(auditor.audit(public_trace(case)))
        for case in cases if case["split"] == split
    }
    return evaluate(cases, predictions, split), predictions


def development_feedback(
    cases: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    max_errors: int = 24,
) -> dict[str, Any]:
    errors = []
    for case in cases:
        if case["split"] != "dev":
            continue
        truth = case["oracle"]
        pred = predictions[case["case_id"]]
        correct = (
            pred["fault_type"] == truth["fault_type"]
            and (
                truth["fault_type"] == "none"
                or (
                    pred["responsible_agent"] == truth["responsible_agent"]
                    and pred["decisive_step"] == truth["decisive_step"]
                )
            )
        )
        if not correct:
            errors.append({
                "truth": {
                    "fault_type": truth["fault_type"],
                    "responsible_agent": truth["responsible_agent"],
                    "decisive_step": truth["decisive_step"],
                },
                "prediction": {
                    "fault_type": pred["fault_type"],
                    "responsible_agent": pred["responsible_agent"],
                    "decisive_step": pred["decisive_step"],
                },
                "trace": _compact_snapshot(public_trace(case)),
            })
        if len(errors) >= max_errors:
            break
    return {"n_errors_shown": len(errors), "errors": errors}


def repair_prompt(
    current_rule_set: dict[str, Any],
    metrics: dict[str, Any],
    feedback: dict[str, Any],
    validation_error: str | None = None,
) -> str:
    payload = {
        "current_rule_set": current_rule_set,
        "development_metrics": {
            key: metrics.get(key)
            for key in (
                "clean_fpr",
                "fault_detection_recall",
                "root_cause_accuracy",
                "root_cause_macro_f1",
                "agent_attribution_accuracy",
                "decisive_step_accuracy",
            )
        },
        "development_errors": feedback,
        "static_validation_error": validation_error,
    }
    return (
        "Revise the rule set using development evidence only. Preserve generic rules that "
        "already work, repair the listed errors, and return the complete 14-rule JSON "
        "object. Do not use identifiers or instance-specific constants.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def score_for_selection(metrics: dict[str, Any]) -> tuple[float, float, float]:
    """开发集选择：先 root macro-F1，再 clean retention，再 agent/step。"""
    return (
        float(metrics["root_cause_macro_f1"]),
        -float(metrics["clean_fpr"]),
        float(metrics["joint_agent_step_accuracy"]),
    )


def prediction_errors(
    cases: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    split: str,
) -> int:
    count = 0
    for case in cases:
        if case["split"] != split:
            continue
        truth = case["oracle"]
        pred = predictions[case["case_id"]]
        if truth["fault_type"] == "none":
            count += int(detected(pred))
        else:
            count += int(
                pred["fault_type"] != truth["fault_type"]
                or pred["responsible_agent"] != truth["responsible_agent"]
                or pred["decisive_step"] != truth["decisive_step"]
            )
    return count
