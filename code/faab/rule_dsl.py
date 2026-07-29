"""FAAB 可执行审计规则 DSL。

LLM 只能在白名单路径和操作符内归纳规则；测试时由本模块确定性执行，
不允许访问 oracle、case_id、source_id 或任何外部状态。
"""
from __future__ import annotations

import datetime as dt
import math
import re
from typing import Any

from .auditors import AGENTS, _num_in_quote, finding, pass_result
from .benchmark import FAULT_SPECS

RULESET_VERSION = "faab-rule-dsl-1.0"

PATHS: dict[str, tuple[Any, ...]] = {
    "source.ticker": ("trace", "source", "ticker"),
    "source.period": ("trace", "source", "period"),
    "source.t_pub": ("trace", "source", "t_pub"),
    "source.document_excerpt": ("trace", "source", "document_excerpt"),
    "source.control_horizon_td": ("trace", "source", "control_horizon_td"),
    "evidence.hash": ("trace", "artifacts", 0, "hash"),
    "evidence.entity": ("trace", "artifacts", 0, "payload", "entity"),
    "evidence.definition": ("trace", "artifacts", 0, "payload", "definition"),
    "evidence.raw_value": ("trace", "artifacts", 0, "payload", "raw_value"),
    "evidence.unit_multiplier": ("trace", "artifacts", 0, "payload", "unit_multiplier"),
    "evidence.normalized_value": ("trace", "artifacts", 0, "payload", "normalized_value"),
    "evidence.basis": ("trace", "artifacts", 0, "payload", "basis"),
    "evidence.period": ("trace", "artifacts", 0, "payload", "period"),
    "evidence.quote": ("trace", "artifacts", 0, "payload", "quote"),
    "evidence.recommendation": ("trace", "artifacts", 0, "payload", "recommendation"),
    "expectation.hash": ("trace", "artifacts", 1, "hash"),
    "expectation.entity": ("trace", "artifacts", 1, "payload", "entity"),
    "expectation.definition": ("trace", "artifacts", 1, "payload", "definition"),
    "expectation.basis": ("trace", "artifacts", 1, "payload", "basis"),
    "expectation.period": ("trace", "artifacts", 1, "payload", "period"),
    "expectation.vintage": ("trace", "artifacts", 1, "payload", "vintage"),
    "expectation.visible_at": ("trace", "artifacts", 1, "payload", "visible_at"),
    "expectation.latest_eligible_vintage": (
        "trace", "artifacts", 1, "payload", "latest_eligible_vintage"),
    "consistency.hash": ("trace", "artifacts", 2, "hash"),
    "consistency.actual_ref": ("trace", "artifacts", 2, "payload", "actual_ref"),
    "consistency.expectation_ref": (
        "trace", "artifacts", 2, "payload", "expectation_ref"),
    "consistency.normalized_value": (
        "trace", "artifacts", 2, "payload", "normalized_value"),
    "consistency.qualification": (
        "trace", "artifacts", 2, "payload", "qualification"),
    "decision.value_used": ("trace", "artifacts", 3, "payload", "value_used"),
    "decision.source_ref": ("trace", "artifacts", 3, "payload", "source_ref"),
    "decision.qualification_refs": (
        "trace", "artifacts", 3, "payload", "qualification_refs"),
    "decision.override_reason": ("trace", "artifacts", 3, "payload", "override_reason"),
    "decision.decision_time": ("trace", "artifacts", 3, "payload", "decision_time"),
    "settlement.ticker": ("trace", "artifacts", 4, "payload", "ticker"),
    "settlement.horizon_td": ("trace", "artifacts", 4, "payload", "horizon_td"),
}

SCALAR_OPS = {"eq", "ne", "numeric_not_close", "gt_time"}
UNARY_OPS = {"empty", "not_empty"}
SPECIAL_OPS = {
    "quote_not_in_document",
    "number_not_in_quote",
    "number_in_other_span",
    "normalization_mismatch",
}
LOGIC_OPS = {"all", "any", "not"}
LIST_OPS = {"not_list_contains"}
ALL_OPS = SCALAR_OPS | UNARY_OPS | SPECIAL_OPS | LOGIC_OPS | LIST_OPS


class RuleValidationError(ValueError):
    """规则集不满足可执行与防泄漏约束。"""


def resolve_path(public_case: dict[str, Any], path: str) -> Any:
    if path not in PATHS:
        raise RuleValidationError(f"path_not_allowed:{path}")
    value: Any = public_case
    for key in PATHS[path]:
        try:
            value = value[key]
        except (KeyError, IndexError, TypeError):
            return None
    return value


def feature_snapshot(public_case: dict[str, Any]) -> dict[str, Any]:
    """给规则归纳器的固定特征视图；不含任何标签或实例标识符。"""
    return {path: resolve_path(public_case, path) for path in PATHS}


def _validate_operand(value: Any, location: str) -> None:
    if not isinstance(value, dict) or set(value) not in ({"path"}, {"const"}):
        raise RuleValidationError(f"{location}:operand_must_be_path_or_const")
    if "path" in value:
        if value["path"] not in PATHS:
            raise RuleValidationError(f"{location}:path_not_allowed:{value['path']}")
        return
    constant = value["const"]
    if constant is None or isinstance(constant, bool):
        return
    if isinstance(constant, (int, float)) and math.isfinite(float(constant)):
        if abs(float(constant)) > 1_000_000_000:
            raise RuleValidationError(f"{location}:numeric_constant_too_large")
        return
    # 禁止 ticker、日期、case/source ID 等字符串常量进入规则。
    raise RuleValidationError(f"{location}:string_constants_forbidden")


def _validate_condition(condition: Any, location: str = "condition",
                        depth: int = 0, budget: list[int] | None = None) -> None:
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > 80 or depth > 8:
        raise RuleValidationError(f"{location}:condition_too_complex")
    if not isinstance(condition, dict):
        raise RuleValidationError(f"{location}:condition_must_be_object")
    op = condition.get("op")
    if op not in ALL_OPS:
        raise RuleValidationError(f"{location}:op_not_allowed:{op}")
    if op in ("all", "any"):
        args = condition.get("args")
        if not isinstance(args, list) or not 1 <= len(args) <= 12:
            raise RuleValidationError(f"{location}:args_invalid")
        for i, child in enumerate(args):
            _validate_condition(child, f"{location}.args[{i}]", depth + 1, budget)
    elif op == "not":
        _validate_condition(condition.get("arg"), f"{location}.arg", depth + 1, budget)
    elif op in SCALAR_OPS:
        _validate_operand(condition.get("left"), f"{location}.left")
        _validate_operand(condition.get("right"), f"{location}.right")
    elif op in UNARY_OPS:
        _validate_operand(condition.get("value"), f"{location}.value")
    elif op in LIST_OPS:
        _validate_operand(condition.get("container"), f"{location}.container")
        _validate_operand(condition.get("value"), f"{location}.value")


def validate_rule_set(rule_set: Any, require_all_faults: bool = True) -> dict[str, Any]:
    if not isinstance(rule_set, dict):
        raise RuleValidationError("ruleset_must_be_object")
    if rule_set.get("version") != RULESET_VERSION:
        raise RuleValidationError(f"version_must_equal:{RULESET_VERSION}")
    rules = rule_set.get("rules")
    if not isinstance(rules, list) or not rules:
        raise RuleValidationError("rules_must_be_nonempty_list")
    seen_ids: set[str] = set()
    seen_faults: set[str] = set()
    seen_priorities: set[int] = set()
    for i, rule in enumerate(rules):
        where = f"rules[{i}]"
        if not isinstance(rule, dict):
            raise RuleValidationError(f"{where}:must_be_object")
        rule_id = rule.get("rule_id")
        if not isinstance(rule_id, str) or not re.fullmatch(r"R[0-9]{2,3}", rule_id):
            raise RuleValidationError(f"{where}:invalid_rule_id")
        if rule_id in seen_ids:
            raise RuleValidationError(f"{where}:duplicate_rule_id:{rule_id}")
        seen_ids.add(rule_id)
        fault = rule.get("fault_type")
        if fault not in FAULT_SPECS:
            raise RuleValidationError(f"{where}:invalid_fault_type:{fault}")
        if fault in seen_faults:
            raise RuleValidationError(f"{where}:duplicate_fault_type:{fault}")
        seen_faults.add(fault)
        if rule.get("responsible_agent") not in AGENTS[1:]:
            raise RuleValidationError(f"{where}:invalid_agent")
        if not isinstance(rule.get("decisive_step"), int) or not 1 <= rule["decisive_step"] <= 5:
            raise RuleValidationError(f"{where}:invalid_step")
        priority = rule.get("priority")
        if not isinstance(priority, int) or not 1 <= priority <= 999:
            raise RuleValidationError(f"{where}:invalid_priority")
        if priority in seen_priorities:
            raise RuleValidationError(f"{where}:duplicate_priority:{priority}")
        seen_priorities.add(priority)
        description = rule.get("description", "")
        if not isinstance(description, str) or len(description) > 240:
            raise RuleValidationError(f"{where}:description_too_long")
        _validate_condition(rule.get("condition"), f"{where}.condition")
    if require_all_faults and seen_faults != set(FAULT_SPECS):
        missing = sorted(set(FAULT_SPECS) - seen_faults)
        extra = sorted(seen_faults - set(FAULT_SPECS))
        raise RuleValidationError(f"fault_coverage_mismatch:missing={missing}:extra={extra}")
    return rule_set


def _operand(public_case: dict[str, Any], spec: dict[str, Any]) -> Any:
    return resolve_path(public_case, spec["path"]) if "path" in spec else spec["const"]


def _date_value(value: Any) -> dt.datetime:
    text = str(value)
    if len(text) == 10:
        text += "T00:00:00+00:00"
    return dt.datetime.fromisoformat(text)


def evaluate_condition(condition: dict[str, Any], public_case: dict[str, Any]) -> bool:
    op = condition["op"]
    if op == "all":
        return all(evaluate_condition(x, public_case) for x in condition["args"])
    if op == "any":
        return any(evaluate_condition(x, public_case) for x in condition["args"])
    if op == "not":
        return not evaluate_condition(condition["arg"], public_case)
    if op == "eq":
        return _operand(public_case, condition["left"]) == _operand(public_case, condition["right"])
    if op == "ne":
        return _operand(public_case, condition["left"]) != _operand(public_case, condition["right"])
    if op == "numeric_not_close":
        left = _operand(public_case, condition["left"])
        right = _operand(public_case, condition["right"])
        try:
            return not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return True
    if op == "gt_time":
        try:
            return _date_value(_operand(public_case, condition["left"])) > _date_value(
                _operand(public_case, condition["right"]))
        except (TypeError, ValueError):
            return True
    if op == "empty":
        return _operand(public_case, condition["value"]) in (None, "", [], {})
    if op == "not_empty":
        return _operand(public_case, condition["value"]) not in (None, "", [], {})
    if op == "not_list_contains":
        container = _operand(public_case, condition["container"])
        value = _operand(public_case, condition["value"])
        return not isinstance(container, list) or value not in container

    quote = str(resolve_path(public_case, "evidence.quote") or "")
    document = str(resolve_path(public_case, "source.document_excerpt") or "")
    raw_value = resolve_path(public_case, "evidence.raw_value")
    if op == "quote_not_in_document":
        return quote not in document
    if op == "number_not_in_quote":
        try:
            return not _num_in_quote(float(raw_value), quote)
        except (TypeError, ValueError):
            return True
    if op == "number_in_other_span":
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return False
        spans = [
            span.strip()
            for span in re.split(r"\n\n+|(?<=[.!?])\s+", document)
            if span.strip() and span.strip() != quote.strip()
        ]
        return any(_num_in_quote(value, span) for span in spans)
    if op == "normalization_mismatch":
        try:
            expected = float(resolve_path(public_case, "evidence.raw_value")) * float(
                resolve_path(public_case, "evidence.unit_multiplier"))
            actual = float(resolve_path(public_case, "evidence.normalized_value"))
            return not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return True
    raise RuleValidationError(f"unreachable_op:{op}")


class RuleSetAuditor:
    """冻结规则集的确定性审计器。"""

    name = "rule_set"

    def __init__(self, rule_set: dict[str, Any], name: str = "rule_set"):
        self.rule_set = validate_rule_set(rule_set)
        self.rules = sorted(rule_set["rules"], key=lambda r: r["priority"])
        self.name = name

    def audit(self, public_case: dict[str, Any]) -> dict[str, Any]:
        for rule in self.rules:
            if evaluate_condition(rule["condition"], public_case):
                return finding(
                    rule["fault_type"],
                    rule["responsible_agent"],
                    rule["decisive_step"],
                    rule["rule_id"],
                    rationale=rule.get("description", ""),
                )
        return pass_result("No frozen executable rule fired.")


def _path(name: str) -> dict[str, str]:
    return {"path": name}


def _const(value: Any) -> dict[str, Any]:
    return {"const": value}


def _cmp(op: str, left: str, right: str | float | int) -> dict[str, Any]:
    right_spec = _path(right) if isinstance(right, str) and right in PATHS else _const(right)
    return {"op": op, "left": _path(left), "right": right_spec}


def human_rule_set() -> dict[str, Any]:
    """人工金融控制基线，以同一 DSL 表示，便于与 LLM 归纳规则公平比较。"""
    rules = [
        {
            "rule_id": "R01", "priority": 10, "fault_type": "role_violation",
            "responsible_agent": "evidence_agent", "decisive_step": 1,
            "description": "Evidence role emitted a forbidden recommendation.",
            "condition": {"op": "not_empty", "value": _path("evidence.recommendation")},
        },
        {
            "rule_id": "R02", "priority": 20, "fault_type": "entity_confusion",
            "responsible_agent": "evidence_agent", "decisive_step": 1,
            "description": "Evidence entity differs from the source ticker.",
            "condition": _cmp("ne", "evidence.entity", "source.ticker"),
        },
        {
            "rule_id": "R03", "priority": 30, "fault_type": "quote_mismatch",
            "responsible_agent": "evidence_agent", "decisive_step": 1,
            "description": "Submitted citation is absent or the value is supported only elsewhere.",
            "condition": {"op": "any", "args": [
                {"op": "quote_not_in_document"},
                {"op": "all", "args": [
                    {"op": "number_not_in_quote"},
                    {"op": "number_in_other_span"},
                ]},
            ]},
        },
        {
            "rule_id": "R04", "priority": 40, "fault_type": "fabricated_value",
            "responsible_agent": "evidence_agent", "decisive_step": 1,
            "description": "Claimed value is absent from the submitted quote and all source spans.",
            "condition": {"op": "all", "args": [
                {"op": "number_not_in_quote"},
                {"op": "not", "arg": {"op": "number_in_other_span"}},
            ]},
        },
        {
            "rule_id": "R05", "priority": 50, "fault_type": "unit_scaling",
            "responsible_agent": "evidence_agent", "decisive_step": 1,
            "description": "Per-share evidence uses a non-unit multiplier or inconsistent normalization.",
            "condition": {"op": "any", "args": [
                _cmp("ne", "evidence.unit_multiplier", 1.0),
                {"op": "normalization_mismatch"},
            ]},
        },
        {
            "rule_id": "R06", "priority": 60, "fault_type": "basis_conflict",
            "responsible_agent": "expectation_agent", "decisive_step": 2,
            "description": "Evidence and expectation use different accounting bases.",
            "condition": _cmp("ne", "evidence.basis", "expectation.basis"),
        },
        {
            "rule_id": "R07", "priority": 70, "fault_type": "period_mismatch",
            "responsible_agent": "expectation_agent", "decisive_step": 2,
            "description": "Evidence and expectation refer to different fiscal periods.",
            "condition": _cmp("ne", "evidence.period", "expectation.period"),
        },
        {
            "rule_id": "R08", "priority": 80, "fault_type": "definition_conflict",
            "responsible_agent": "expectation_agent", "decisive_step": 2,
            "description": "Evidence and expectation use incompatible EPS definitions.",
            "condition": _cmp("ne", "evidence.definition", "expectation.definition"),
        },
        {
            "rule_id": "R09", "priority": 90, "fault_type": "future_leakage",
            "responsible_agent": "expectation_agent", "decisive_step": 2,
            "description": "Expectation became visible after the decision time.",
            "condition": _cmp("gt_time", "expectation.visible_at", "decision.decision_time"),
        },
        {
            "rule_id": "R10", "priority": 100, "fault_type": "stale_expectation",
            "responsible_agent": "expectation_agent", "decisive_step": 2,
            "description": "Selected consensus is not the latest eligible pre-event vintage.",
            "condition": _cmp(
                "ne", "expectation.vintage", "expectation.latest_eligible_vintage"),
        },
        {
            "rule_id": "R11", "priority": 110, "fault_type": "missing_handoff",
            "responsible_agent": "decision_agent", "decisive_step": 4,
            "description": "Decision lost its consistency source or qualification reference.",
            "condition": {"op": "any", "args": [
                _cmp("ne", "decision.source_ref", "consistency.hash"),
                {
                    "op": "not_list_contains",
                    "container": _path("decision.qualification_refs"),
                    "value": _path("consistency.hash"),
                },
            ]},
        },
        {
            "rule_id": "R12", "priority": 120, "fault_type": "silent_overwrite",
            "responsible_agent": "decision_agent", "decisive_step": 4,
            "description": "Decision changed the aligned value without an override reason.",
            "condition": {"op": "all", "args": [
                _cmp("numeric_not_close", "decision.value_used", "consistency.normalized_value"),
                {"op": "empty", "value": _path("decision.override_reason")},
            ]},
        },
        {
            "rule_id": "R13", "priority": 130, "fault_type": "wrong_security",
            "responsible_agent": "settlement_agent", "decisive_step": 5,
            "description": "Settlement target differs from the source security.",
            "condition": _cmp("ne", "settlement.ticker", "source.ticker"),
        },
        {
            "rule_id": "R14", "priority": 140, "fault_type": "wrong_window",
            "responsible_agent": "settlement_agent", "decisive_step": 5,
            "description": "Settlement horizon differs from the declared control horizon.",
            "condition": _cmp(
                "ne", "settlement.horizon_td", "source.control_horizon_td"),
        },
    ]
    return validate_rule_set({"version": RULESET_VERSION, "rules": rules})
