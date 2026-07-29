"""汇总冻结 FAAB 评测：source-cluster bootstrap、配对检验与逐故障混淆。

支持任意 (方法, 模型) 臂：预测文件按 `<method>_<split>[__<model>].jsonl` 自动发现，
因此新增模型或新增审计器设计不需要改分析代码。

  python3 eval/faab_analysis.py --split test
  python3 eval/faab_analysis.py --split challenge

责任定位在正文按单一指标报告：本工作流的责任工位与决定性步骤一一对应（见
faab/metrics.py 的 agent_step_bijective），分列两栏会把同一个变量报告两次。
分列值仍写入 JSON 供附录核对。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faab.benchmark import FAULT_SPECS, load_benchmark
from faab.metrics import WEIGHT, detected

ROOT = Path(__file__).resolve().parent.parent
SPLITS = {
    "test": (ROOT / "data/faab/benchmark.jsonl", ROOT / "data/faab/results"),
    "challenge": (ROOT / "data/faab_rule_induction/challenge.jsonl",
                  ROOT / "data/faab_rule_induction/results"),
}
METRICS = (
    "clean_fpr",
    "fault_detection_recall",
    "root_cause_accuracy",
    "root_cause_macro_f1",
    "responsibility_localization_accuracy",
    "agent_attribution_accuracy",
    "decisive_step_accuracy",
    "materiality_weighted_recall",
)
# 正文表格口径：责任定位只报一列。
PAPER_METRICS = (
    "clean_fpr", "fault_detection_recall", "root_cause_accuracy",
    "responsibility_localization_accuracy", "materiality_weighted_recall",
)


def discover_arms(results: Path, split: str) -> dict[str, Path]:
    """<method>_<split>[__<model>].jsonl → 臂名。排除 api 日志与指标文件。"""
    pattern = re.compile(rf"^(?P<method>.+?)_{split}(?:__(?P<model>.+))?\.jsonl$")
    arms: dict[str, Path] = {}
    for path in sorted(results.glob(f"*_{split}*.jsonl")):
        if path.name.endswith("_api.jsonl"):
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        method, model = match.group("method"), match.group("model")
        arms[f"{method}@{model}" if model else method] = path
    return arms


def _load(path: Path) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row["prediction"]
            for row in (json.loads(line) for line in path.read_text().splitlines()
                        if line.strip())}


def _weighted_metrics(cases: list[dict[str, Any]], predictions: dict[str, dict[str, Any]],
                      source_weights: Counter[str]) -> dict[str, float]:
    clean = [c for c in cases if c["oracle"]["fault_type"] == "none"]
    faults = [c for c in cases if c["oracle"]["fault_type"] != "none"]

    def rate(rows: list[dict[str, Any]], hit: Callable[[dict, dict], bool]) -> float:
        denominator = sum(source_weights[c["source_id"]] for c in rows)
        numerator = sum(
            source_weights[c["source_id"]] * int(hit(c, predictions[c["case_id"]]))
            for c in rows)
        return numerator / denominator if denominator else 0.0

    denom_w = sum(source_weights[c["source_id"]] * WEIGHT[c["oracle"]["materiality"]]
                  for c in faults)
    hit_w = sum(source_weights[c["source_id"]] * WEIGHT[c["oracle"]["materiality"]]
                for c in faults if detected(predictions[c["case_id"]]))
    f1s = []
    for fault in FAULT_SPECS:
        tp = fp = fn = 0
        for c in faults:
            weight = source_weights[c["source_id"]]
            truth = c["oracle"]["fault_type"]
            pred = predictions[c["case_id"]]["fault_type"]
            tp += weight * int(truth == fault and pred == fault)
            fp += weight * int(truth != fault and pred == fault)
            fn += weight * int(truth == fault and pred != fault)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall)
                   if precision + recall else 0.0)

    def agent_ok(c, p):
        return p["responsible_agent"] == c["oracle"]["responsible_agent"]

    def step_ok(c, p):
        return p["decisive_step"] == c["oracle"]["decisive_step"]

    return {
        "clean_fpr": rate(clean, lambda _c, p: detected(p)),
        "fault_detection_recall": rate(faults, lambda _c, p: detected(p)),
        "root_cause_accuracy": rate(
            faults, lambda c, p: p["fault_type"] == c["oracle"]["fault_type"]),
        "root_cause_macro_f1": sum(f1s) / len(f1s),
        "responsibility_localization_accuracy": rate(
            faults, lambda c, p: agent_ok(c, p) and step_ok(c, p)),
        "agent_attribution_accuracy": rate(faults, agent_ok),
        "decisive_step_accuracy": rate(faults, step_ok),
        "materiality_weighted_recall": hit_w / denom_w if denom_w else 0.0,
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def _exact_sign_p(better: int, worse: int) -> float:
    """配对二项检验的双侧精确 p 值（忽略 ties）。正文只报区间，此值留档备查。"""
    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(better, worse) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=sorted(SPLITS), default="test")
    parser.add_argument("--reference", default="rules_retrieval",
                        help="配对比较的参照臂")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    benchmark, results = SPLITS[args.split]
    cases = [c for c in load_benchmark(benchmark) if c["split"] == args.split]
    sources = sorted({c["source_id"] for c in cases})
    arms = discover_arms(results, args.split)
    predictions = {arm: _load(path) for arm, path in arms.items()}
    complete = {arm for arm, preds in predictions.items()
                if all(c["case_id"] in preds for c in cases)}
    partial = sorted(set(arms) - complete)
    arm_names = sorted(complete)
    if args.reference not in arm_names:
        raise SystemExit(f"参照臂 {args.reference} 不完整；可用：{arm_names}")

    unit = Counter({source_id: 1 for source_id in sources})
    point = {arm: _weighted_metrics(cases, predictions[arm], unit) for arm in arm_names}

    rng = random.Random(args.seed)
    samples = {arm: {m: [] for m in METRICS} for arm in arm_names}
    differences = {arm: {m: [] for m in METRICS}
                   for arm in arm_names if arm != args.reference}
    for _ in range(args.draws):
        weights = Counter(rng.choices(sources, k=len(sources)))
        draw = {arm: _weighted_metrics(cases, predictions[arm], weights)
                for arm in arm_names}
        for arm in arm_names:
            for metric in METRICS:
                samples[arm][metric].append(draw[arm][metric])
        for arm in differences:
            for metric in METRICS:
                differences[arm][metric].append(
                    draw[args.reference][metric] - draw[arm][metric])

    ci95 = {arm: {metric: [_percentile(v, 0.025), _percentile(v, 0.975)]
                  for metric, v in values.items()}
            for arm, values in samples.items()}
    diff_ci = {
        arm: {metric: {"point": point[args.reference][metric] - point[arm][metric],
                       "ci95": [_percentile(v, 0.025), _percentile(v, 0.975)]}
              for metric, v in values.items()}
        for arm, values in differences.items()
    }

    fault_cases = [c for c in cases if c["oracle"]["fault_type"] != "none"]
    clean_cases = [c for c in cases if c["oracle"]["fault_type"] == "none"]
    checks: dict[str, tuple[list[dict[str, Any]], Callable[[dict, dict], bool]]] = {
        "clean_pass": (clean_cases, lambda _c, p: not detected(p)),
        "detection": (fault_cases, lambda _c, p: detected(p)),
        "root_cause": (fault_cases,
                       lambda c, p: p["fault_type"] == c["oracle"]["fault_type"]),
        "responsibility": (
            fault_cases,
            lambda c, p: (p["responsible_agent"] == c["oracle"]["responsible_agent"]
                          and p["decisive_step"] == c["oracle"]["decisive_step"])),
    }
    paired: dict[str, dict[str, Any]] = {}
    for arm in arm_names:
        if arm == args.reference:
            continue
        paired[arm] = {}
        for name, (rows, check) in checks.items():
            better = worse = 0
            for case in rows:
                ref_ok = check(case, predictions[args.reference][case["case_id"]])
                arm_ok = check(case, predictions[arm][case["case_id"]])
                better += int(ref_ok and not arm_ok)
                worse += int(arm_ok and not ref_ok)
            paired[arm][name] = {"reference_only_correct": better,
                                 "arm_only_correct": worse,
                                 "two_sided_exact_p": _exact_sign_p(better, worse)}

    confusion = {}
    for arm in arm_names:
        errors: Counter[str] = Counter()
        for case in fault_cases:
            truth = case["oracle"]["fault_type"]
            pred = predictions[arm][case["case_id"]]["fault_type"]
            if pred != truth:
                errors[f"{truth}->{pred}"] += 1
        confusion[arm] = dict(errors)

    # 逐臂预测一致性：用来把「残余 LLM 零增益」做成可核对的事实而非断言。
    agreement = {}
    for arm in arm_names:
        if arm == args.reference:
            continue
        same = sum(
            all(predictions[arm][c["case_id"]][f] ==
                predictions[args.reference][c["case_id"]][f]
                for f in ("verdict", "fault_type", "responsible_agent", "decisive_step"))
            for c in cases)
        agreement[arm] = {"identical_predictions": same, "n": len(cases),
                          "rate": same / len(cases)}

    runtime = {}
    for arm in arm_names:
        path = results / (arms[arm].name.replace(".jsonl", "_metrics.json"))
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        runtime[arm] = {k: row.get(k) for k in (
            "model", "provider", "api_calls", "api_requests_including_retries",
            "api_failures", "prompt_tokens", "completion_tokens", "total_tokens",
            "mean_latency_s")}

    output = {
        "split": args.split,
        "reference": args.reference,
        "bootstrap": {"clusters": len(sources), "draws": args.draws, "seed": args.seed},
        "arms": arm_names,
        "incomplete_arms": partial,
        "agent_step_bijective": len({(c["oracle"]["responsible_agent"],
                                      c["oracle"]["decisive_step"]) for c in fault_cases})
        == len({c["oracle"]["responsible_agent"] for c in fault_cases}),
        "point": point,
        "ci95": ci95,
        "reference_minus_arm": diff_ci,
        "paired_exact_tests": paired,
        "prediction_agreement_with_reference": agreement,
        "root_cause_errors": confusion,
        "runtime": runtime,
    }
    path = results / f"analysis_{args.split}.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"[{args.split}] arms={arm_names} incomplete={partial} -> {path}")
    for arm in arm_names:
        row = point[arm]
        print(f"  {arm:<34} " + "  ".join(
            f"{m.split('_')[0]}={100 * row[m]:5.1f}" for m in PAPER_METRICS))


if __name__ == "__main__":
    main()
