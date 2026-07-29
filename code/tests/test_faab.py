"""FAAB 的单故障、标签隔离与指标回归测试。"""
from __future__ import annotations

import copy
import json

from faab.auditors import HybridAuditor, RuleAuditor, RulesRetrievalAuditor
from faab.benchmark import (
    FAULT_SPECS,
    _extract_eps_span,
    _inject,
    _same_eps_scale,
    public_trace,
)
from faab.challenge import _inject_challenge
from faab.metrics import evaluate
from faab.tool_auditor import TOOL_SPECS, ToolAugmentedAuditor, ToolRuntime
from faab.rule_dsl import (
    RULESET_VERSION,
    RuleSetAuditor,
    RuleValidationError,
    human_rule_set,
    validate_rule_set,
)


def _trace():
    src = {"msg_id": "m1", "ticker": "AAA", "period": "Q1", "t_pub": "2025-01-01T16:00:00-05:00",
           "document_excerpt": "Adjusted diluted EPS was $1.25.\n\nRevenue increased in the quarter.",
           "control_horizon_td": 3, "hash": "src"}
    payloads = [
        {"entity": "AAA", "metric": "eps", "definition": "adjusted_diluted_eps",
         "raw_value": 1.25, "unit": "USD_per_share", "unit_multiplier": 1.0,
         "normalized_value": 1.25, "basis": "non_gaap", "period": "Q1",
         "quote": "Adjusted diluted EPS was $1.25.", "recommendation": None},
        {"entity": "AAA", "metric": "eps", "definition": "adjusted_diluted_eps",
         "mean_estimate": 1.1, "dispersion": 0.1, "basis": "non_gaap", "period": "Q1",
         "fiscal_period_end": "2025-03-31", "vintage": "2024-12-20",
         "visible_at": "2024-12-20T16:00:00-05:00",
         "latest_eligible_vintage": "2024-12-20", "prior_eligible_vintage": "2024-12-01"},
        {"aligned": True, "actual_ref": "h1", "expectation_ref": "h2",
         "normalized_value": 1.25, "standardized_surprise": 1.5,
         "qualification": "non_gaap:adjusted_diluted_eps:Q1"},
        {"claim": "above_expectations", "value_used": 1.25, "source_ref": "h3",
         "qualification_refs": ["h3"], "override_reason": None,
         "decision_time": "2025-01-01T16:00:00-05:00"},
        {"ticker": "AAA", "benchmark_ticker": "BBB", "horizon_td": 3,
         "decision_time": "2025-01-01T16:00:00-05:00",
         "observation_time": "2025-01-06T16:00:00-05:00"},
    ]
    arts = [{"step": i + 1, "agent": a, "schema_version": "faab-1.0",
             "parent_hashes": ["src" if i == 0 else f"h{i}"], "payload": p, "hash": f"h{i+1}"}
            for i, (a, p) in enumerate(zip(
                ["evidence_agent", "expectation_agent", "consistency_agent",
                 "decision_agent", "settlement_agent"], payloads))]
    return {"source": src, "artifacts": arts}


def test_oracle_not_needed_by_rule_auditor():
    result = RuleAuditor().audit({"case_id": "x", "trace": _trace()})
    assert result["verdict"] == "pass"


def test_each_fault_is_detected_by_rules():
    base = _trace()
    for fault in FAULT_SPECS:
        result = RuleAuditor().audit({"case_id": fault, "trace": _inject(base, fault, "BBB")})
        assert result["verdict"] in ("qualified", "reject"), fault
        assert result["responsible_agent"] == FAULT_SPECS[fault]["agent"], fault


def test_metrics_include_clean_fpr_and_attribution():
    clean = {"case_id": "c", "split": "test",
             "oracle": {"fault_type": "none", "responsible_agent": "none",
                        "decisive_step": 0, "materiality": "none"}}
    fault = {"case_id": "f", "split": "test",
             "oracle": {"fault_type": "wrong_window", "responsible_agent": "settlement_agent",
                        "decisive_step": 5, "materiality": "medium"}}
    preds = {
        "c": {"verdict": "pass", "fault_type": "none", "responsible_agent": "none",
              "decisive_step": 0},
        "f": {"verdict": "reject", "fault_type": "wrong_window",
              "responsible_agent": "settlement_agent", "decisive_step": 5},
    }
    m = evaluate([clean, fault], preds)
    assert m["clean_fpr"] == 0
    assert m["fault_detection_recall"] == 1
    assert m["agent_attribution_accuracy"] == 1
    assert m["decisive_step_accuracy"] == 1


def test_clean_span_requires_actual_eps_and_rejects_forecasts():
    text = ("Diluted EPS was $1.25. Adjusted EPS outlook is $2.60 to $2.70. "
            "The board declared a dividend of $0.65 per share.")
    assert _extract_eps_span("Diluted EPS was $1.25", text, 1.25) == (
        "Diluted EPS was $1.25", 1.25)
    assert _extract_eps_span("Adjusted EPS outlook is $2.60 to $2.70", text, 2.60) is None
    assert _extract_eps_span("Dividend was $0.65 per share", text, 0.65) is None
    assert _extract_eps_span("Diluted EPS was $1.25", text, 1.10) is None


def test_ibes_actual_alignment_rejects_scale_mismatch():
    assert _same_eps_scale(1.25, 1.25)
    assert _same_eps_scale(1.25, 1.26)
    assert not _same_eps_scale(10.96, 1.096)


def test_hybrid_cannot_overturn_deterministic_grounding_failure():
    corrupt = _inject(_trace(), "fabricated_value", "BBB")

    def bad_semantic(*_args):
        return {"verdict": "pass", "fault_type": "none", "responsible_agent": "none",
                "decisive_step": 0, "materiality": "none", "evidence_refs": [],
                "rationale": "incorrect override"}

    result = HybridAuditor(bad_semantic).audit({"trace_id": "opaque", "trace": corrupt})
    assert result["verdict"] == "reject"
    assert result["fault_type"] == "fabricated_value"


def test_hybrid_separates_wrong_quote_from_fabricated_value():
    base = _trace()
    wrong_quote = _inject(base, "quote_mismatch", "BBB")

    def should_not_call(*_args):
        raise AssertionError("确定性 grounding 违规不应调用 LLM")

    result = HybridAuditor(should_not_call).audit(
        {"trace_id": "opaque", "trace": wrong_quote})
    assert result["fault_type"] == "quote_mismatch"


def test_residual_llm_is_only_reachable_after_a_deterministic_pass():
    """hybrid = 确定性内核 + 仅在内核放行时的残余复核；故障轨迹上两者必须一致。"""
    base = _trace()
    core = RulesRetrievalAuditor()

    def must_not_be_called(*_args):
        raise AssertionError("内核已定罪时不得升级到 LLM")

    hybrid = HybridAuditor(must_not_be_called)
    for fault in FAULT_SPECS:
        case = {"trace_id": "opaque", "trace": _inject(base, fault, "BBB")}
        assert core.audit(case)["fault_type"] == hybrid.audit(case)["fault_type"]
    assert core.audit({"trace_id": "clean", "trace": base})["verdict"] == "pass"


def test_public_trace_hides_case_label_prefix():
    case = {
        "case_id": "future_leakage_007",
        "source_id": "msg_opaque",
        "trace": _trace(),
        "oracle": {"fault_type": "future_leakage"},
    }
    public = public_trace(case)
    rendered = json.dumps(public)
    assert "case_id" not in public
    assert "oracle" not in rendered
    assert "future_leakage" not in rendered


def test_human_dsl_matches_all_fault_contracts():
    auditor = RuleSetAuditor(human_rule_set(), "human_rules")
    base = _trace()
    clean = auditor.audit({"trace_id": "clean", "trace": base})
    assert clean["fault_type"] == "none"
    for fault, spec in FAULT_SPECS.items():
        pred = auditor.audit({
            "trace_id": "opaque",
            "trace": _inject(base, fault, "BBB"),
        })
        assert pred["fault_type"] == fault
        assert pred["responsible_agent"] == spec["agent"]
        assert pred["decisive_step"] == spec["step"]


def test_rule_dsl_rejects_identifier_literal():
    rules = human_rule_set()
    bad = copy.deepcopy(rules)
    bad["rules"][0]["condition"] = {
        "op": "eq",
        "left": {"path": "source.ticker"},
        "right": {"const": "AAPL"},
    }
    try:
        validate_rule_set(bad)
    except RuleValidationError as exc:
        assert "string_constants_forbidden" in str(exc)
    else:
        raise AssertionError("Ticker literal must be rejected")


def test_rule_dsl_rejects_oracle_path():
    rules = human_rule_set()
    bad = copy.deepcopy(rules)
    bad["version"] = RULESET_VERSION
    bad["rules"][0]["condition"] = {
        "op": "ne",
        "left": {"path": "oracle.fault_type"},
        "right": {"const": None},
    }
    try:
        validate_rule_set(bad)
    except RuleValidationError as exc:
        assert "path_not_allowed" in str(exc)
    else:
        raise AssertionError("Oracle path must be rejected")


def test_metrics_report_agent_step_collinearity():
    """责任工位与决定性步骤在本工作流一一对应；该事实必须被测量并报告。"""
    faults = [
        {"case_id": f, "split": "test",
         "oracle": {"fault_type": f, "responsible_agent": spec["agent"],
                    "decisive_step": spec["step"], "materiality": spec["materiality"]}}
        for f, spec in FAULT_SPECS.items()
    ]
    preds = {
        c["case_id"]: {"verdict": "reject", "fault_type": c["oracle"]["fault_type"],
                       "responsible_agent": c["oracle"]["responsible_agent"],
                       "decisive_step": c["oracle"]["decisive_step"]}
        for c in faults
    }
    m = evaluate(faults, preds)
    assert m["agent_step_bijective"] is True
    assert m["responsibility_localization_accuracy"] == m["joint_agent_step_accuracy"] == 1
    steps = {pair[1] for pair in m["label_agent_step_pairs"]}
    agents = {pair[0] for pair in m["label_agent_step_pairs"]}
    assert len(steps) == len(agents) == len(m["label_agent_step_pairs"])


def test_tool_runtime_answers_arithmetic_and_time_primitives():
    runtime = ToolRuntime({"trace": _trace()})
    assert runtime.run("multiply", {"a": 1.25, "b": 1_000_000})["product"] == 1_250_000
    assert runtime.run("compare_numbers", {"a": 1.25, "b": 1.25})["equal"] is True
    assert runtime.run("compare_numbers", {"a": 1.25, "b": 1.10})["order"] == "a_greater"
    later = runtime.run("compare_dates", {"a": "2025-01-02T16:00:00-05:00",
                                          "b": "2025-01-01T16:00:00-05:00"})
    assert later["relation"] == "after" and later["a_minus_b_days"] == 1.0
    assert runtime.run("get_field", {"path": "artifacts.1.payload.vintage"})["value"] == \
        "2024-12-20"
    assert runtime.run("verify_hash_chain", {})["artifacts"][0]["agent"] == "evidence_agent"


def test_tool_retrieval_separates_wrong_citation_from_unsupported_value():
    """跨 span 检索必须能区分「引文错配」与「数字根本不存在」。"""
    runtime = ToolRuntime({"trace": _inject(_trace(), "quote_mismatch", "BBB")})
    hit = runtime.run("find_value_in_document", {"value": 1.25})
    assert hit["n_matching_spans"] >= 1
    assert hit["submitted_quote_contains_value"] is False

    runtime = ToolRuntime({"trace": _inject(_trace(), "fabricated_value", "BBB")})
    claimed = runtime.trace["artifacts"][0]["payload"]["raw_value"]
    miss = runtime.run("find_value_in_document", {"value": claimed})
    assert miss["n_matching_spans"] == 0


def test_tool_runtime_cannot_reach_oracle_or_case_label():
    """工具只读 public_trace；oracle 与可读 case_id 必须不可达。"""
    case = {"case_id": "future_leakage_007", "trace": _trace(),
            "oracle": {"fault_type": "future_leakage"}}
    runtime = ToolRuntime({"trace": case["trace"]})
    for path in ("oracle", "oracle.fault_type", "case_id"):
        assert runtime.run("get_field", {"path": path})["found"] is False


def test_tool_auditor_stops_early_when_model_stops_calling_tools():
    class _Reply:
        def __init__(self, content, tool_calls):
            self.content = content
            self.tool_calls = tool_calls
            self.parsed_json = None

    class _Model:
        """最小 provider 桩：会话历史只需可追加。"""

        def seed(self, system, user):
            return [system, user]

        def add_user(self, conversation, text):
            conversation.append(text)

        def add_reply(self, conversation, reply):
            conversation.append(reply)

        def add_tool_result(self, conversation, call, output):
            conversation.append(output)

    rounds = []

    def chat(conversation, tools=None, max_tokens=0, json_object=False, purpose=""):
        rounds.append(purpose)
        if purpose == "tool_final":
            reply = _Reply(None, [])
            reply.parsed_json = {"verdict": "reject", "fault_type": "wrong_window",
                                 "responsible_agent": "settlement_agent",
                                 "decisive_step": 5, "materiality": "medium",
                                 "evidence_refs": [], "rationale": "C7"}
            return reply, {}
        return _Reply("done", []), {}

    result = ToolAugmentedAuditor(chat, _Model()).audit({"trace": _trace()})
    assert rounds == ["tool_round", "tool_final"]
    assert result["fault_type"] == "wrong_window"
    assert result["tool_rounds"] == 1


def test_tool_specs_expose_only_primitive_computations():
    """工具不得直接返回故障判断；名字与描述里不能出现故障类别。"""
    names = {t["function"]["name"] for t in TOOL_SPECS}
    assert names == {"get_field", "compare_numbers", "multiply", "compare_dates",
                     "find_value_in_document", "verify_hash_chain"}
    rendered = json.dumps(TOOL_SPECS).lower()
    for fault in FAULT_SPECS:
        assert fault not in rendered


def test_human_dsl_handles_unseen_missing_handoff_variant():
    auditor = RuleSetAuditor(human_rule_set(), "human_rules")
    corrupt = _inject_challenge(_trace(), "missing_handoff", "BBB")
    pred = auditor.audit({"trace_id": "opaque", "trace": corrupt})
    assert pred["fault_type"] == "missing_handoff"
    assert pred["responsible_agent"] == "decision_agent"
