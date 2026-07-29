"""FAAB rule-induction challenge 的聚类区间、配对检验和稳定性分析。"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faab.auditors import normalize_result
from faab.benchmark import load_benchmark, public_trace
from faab.challenge import DEFAULT_CHALLENGE
from faab.metrics import WEIGHT, detected, evaluate
from faab.rule_dsl import RuleSetAuditor
from faab.rule_induction import evaluate_rules

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "faab_rule_induction" / "results"
METHODS = (
    "local_rule_only",
    "human_rules",
    "llm_induced_rules",
    "human_plus_induced",
    "llm_judge",
    "debate",
)
METRICS = (
    "clean_fpr",
    "fault_detection_recall",
    "root_cause_accuracy",
    "root_cause_macro_f1",
    "agent_attribution_accuracy",
    "decisive_step_accuracy",
    "joint_agent_step_accuracy",
    "materiality_weighted_recall",
)


def _load_predictions(method: str) -> dict[str, dict[str, Any]]:
    path = RESULTS / f"{method}_challenge.jsonl"
    return {
        row["case_id"]: row["prediction"]
        for row in (
            json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def _weighted_metrics(
    cases: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    weights: Counter[str],
) -> dict[str, float]:
    clean = [case for case in cases if case["oracle"]["fault_type"] == "none"]
    faults = [case for case in cases if case["oracle"]["fault_type"] != "none"]

    def total(rows: list[dict[str, Any]]) -> int:
        return sum(weights[case["source_id"]] for case in rows)

    def rate(rows: list[dict[str, Any]], hit: Callable[[dict, dict], bool]) -> float:
        denominator = total(rows)
        if not denominator:
            return 0.0
        return sum(
            weights[case["source_id"]] * int(hit(case, predictions[case["case_id"]]))
            for case in rows
        ) / denominator

    f1s = []
    fault_types = sorted({case["oracle"]["fault_type"] for case in faults})
    for fault in fault_types:
        tp = fp = fn = 0
        for case in faults:
            weight = weights[case["source_id"]]
            truth = case["oracle"]["fault_type"]
            pred = predictions[case["case_id"]]["fault_type"]
            tp += weight * int(truth == fault and pred == fault)
            fp += weight * int(truth != fault and pred == fault)
            fn += weight * int(truth == fault and pred != fault)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0)
    denominator_w = sum(
        weights[case["source_id"]] * WEIGHT[case["oracle"]["materiality"]]
        for case in faults
    )
    hit_w = sum(
        weights[case["source_id"]] * WEIGHT[case["oracle"]["materiality"]]
        for case in faults if detected(predictions[case["case_id"]])
    )
    return {
        "clean_fpr": rate(clean, lambda _case, pred: detected(pred)),
        "fault_detection_recall": rate(faults, lambda _case, pred: detected(pred)),
        "root_cause_accuracy": rate(
            faults,
            lambda case, pred: pred["fault_type"] == case["oracle"]["fault_type"],
        ),
        "root_cause_macro_f1": sum(f1s) / len(f1s),
        "agent_attribution_accuracy": rate(
            faults,
            lambda case, pred:
                pred["responsible_agent"] == case["oracle"]["responsible_agent"],
        ),
        "decisive_step_accuracy": rate(
            faults,
            lambda case, pred:
                pred["decisive_step"] == case["oracle"]["decisive_step"],
        ),
        "joint_agent_step_accuracy": rate(
            faults,
            lambda case, pred: (
                pred["responsible_agent"] == case["oracle"]["responsible_agent"]
                and pred["decisive_step"] == case["oracle"]["decisive_step"]
            ),
        ),
        "materiality_weighted_recall": hit_w / denominator_w,
    }


def _percentile(values: list[float], quantile: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def _exact_sign_p(better: int, worse: int) -> float:
    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(better, worse) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _case_correct(case: dict[str, Any], prediction: dict[str, Any],
                  target: str) -> bool:
    if target == "clean_pass":
        return not detected(prediction)
    if target == "detection":
        return detected(prediction)
    if target == "root_cause":
        return prediction["fault_type"] == case["oracle"]["fault_type"]
    if target == "agent_step":
        return (
            prediction["responsible_agent"] == case["oracle"]["responsible_agent"]
            and prediction["decisive_step"] == case["oracle"]["decisive_step"]
        )
    raise KeyError(target)


def _seed_stability(
    challenge_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    base_cases = load_benchmark()
    runs = []
    challenge_predictions: dict[int, dict[str, dict[str, Any]]] = {}
    for path in sorted(RESULTS.glob("induced_seed_*.json")):
        run = json.loads(path.read_text())
        seed = int(run["seed"])
        rule_set = run["selected_rule_set"]
        dev_metrics, _ = evaluate_rules(rule_set, base_cases, "dev", f"seed_{seed}")
        challenge_metrics, predictions = evaluate_rules(
            rule_set, challenge_cases, "challenge", f"seed_{seed}")
        challenge_predictions[seed] = predictions
        runs.append({
            "seed": seed,
            "selected_round": run["selected_round"],
            "development": dev_metrics,
            "challenge": challenge_metrics,
        })
    agreement = {}
    seeds = sorted(challenge_predictions)
    for i, left in enumerate(seeds):
        for right in seeds[i + 1:]:
            rows = [
                case for case in challenge_cases if case["split"] == "challenge"]
            same = sum(
                challenge_predictions[left][case["case_id"]]["fault_type"]
                == challenge_predictions[right][case["case_id"]]["fault_type"]
                for case in rows
            )
            agreement[f"{left}_vs_{right}"] = same / len(rows)
    return {"runs": runs, "pairwise_fault_prediction_agreement": agreement}


def main() -> None:
    cases = [
        case for case in load_benchmark(DEFAULT_CHALLENGE)
        if case["split"] == "challenge"
    ]
    sources = sorted({case["source_id"] for case in cases})
    predictions = {method: _load_predictions(method) for method in METHODS}
    point_weights = Counter({source_id: 1 for source_id in sources})
    point = {
        method: _weighted_metrics(cases, predictions[method], point_weights)
        for method in METHODS
    }

    rng = random.Random(20260727)
    draws = 10_000
    samples = {
        method: {metric: [] for metric in METRICS} for method in METHODS
    }
    differences = {
        method: {metric: [] for metric in METRICS}
        for method in METHODS if method != "llm_induced_rules"
    }
    for _ in range(draws):
        weights = Counter(rng.choices(sources, k=len(sources)))
        draw = {
            method: _weighted_metrics(cases, predictions[method], weights)
            for method in METHODS
        }
        for method in METHODS:
            for metric in METRICS:
                samples[method][metric].append(draw[method][metric])
        for method in differences:
            for metric in METRICS:
                differences[method][metric].append(
                    draw["llm_induced_rules"][metric] - draw[method][metric])
    intervals = {
        method: {
            metric: [_percentile(values, 0.025), _percentile(values, 0.975)]
            for metric, values in metrics.items()
        }
        for method, metrics in samples.items()
    }
    difference_intervals = {
        method: {
            metric: {
                "point": point["llm_induced_rules"][metric] - point[method][metric],
                "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
            }
            for metric, values in metrics.items()
        }
        for method, metrics in differences.items()
    }

    clean = [case for case in cases if case["oracle"]["fault_type"] == "none"]
    faults = [case for case in cases if case["oracle"]["fault_type"] != "none"]
    paired = {}
    for method in differences:
        paired[method] = {}
        for target, rows in (
            ("clean_pass", clean),
            ("detection", faults),
            ("root_cause", faults),
            ("agent_step", faults),
        ):
            better = worse = 0
            for case in rows:
                induced_ok = _case_correct(
                    case, predictions["llm_induced_rules"][case["case_id"]], target)
                baseline_ok = _case_correct(
                    case, predictions[method][case["case_id"]], target)
                better += int(induced_ok and not baseline_ok)
                worse += int(baseline_ok and not induced_ok)
            paired[method][target] = {
                "induced_only_correct": better,
                "baseline_only_correct": worse,
                "two_sided_exact_p": _exact_sign_p(better, worse),
            }

    runtime = {
        method: json.loads(
            (RESULTS / f"{method}_challenge_metrics.json").read_text())
        for method in METHODS
    }
    induction = json.loads((RESULTS / "induced_selected.json").read_text())
    output = {
        "freeze": json.loads(
            (ROOT / "data" / "faab_rule_induction" /
             "experiment_freeze.json").read_text()),
        "bootstrap": {"clusters": len(sources), "draws": draws, "seed": 20260727},
        "point": point,
        "ci95": intervals,
        "induced_minus_baseline": difference_intervals,
        "paired_exact_tests": paired,
        "per_fault": {
            method: runtime[method]["per_fault"] for method in METHODS
        },
        "runtime": {
            method: {
                key: runtime[method].get(key)
                for key in (
                    "api_calls", "api_requests_including_retries", "api_failures",
                    "prompt_tokens", "completion_tokens", "total_tokens",
                    "mean_latency_s",
                )
            }
            for method in METHODS
        },
        "offline_induction_cost": induction["api_usage"],
        "seed_stability": _seed_stability(cases),
    }
    path = RESULTS / "challenge_analysis.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
