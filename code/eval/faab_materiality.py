"""测量每类故障对已声明标准化超预期的实际影响（确定性，无模型调用）。

动机：原 materiality 是按故障模板拍板的高/中二分，与会计审计里 materiality 的量化
含义不符，也没有与任何决策口径挂钩。这里改为可测量：对每条成对 clean/corrupt 轨迹，
按「工作流忠实传播被污染输入」重算 SUE = (actual - mean) / max(stdev, 5%|mean|, 0.005)，
再比较声明档位是否反转。

产出的经验事实：只有改变数值输入的故障才改变声明数字；其余故障保持数字不变而摧毁
该数字的支撑与可采纳性。因此 FAAB 的 materiality 是 admissibility，不是 magnitude。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faab.benchmark import FAULT_SPECS, load_benchmark

ROOT = Path(__file__).resolve().parent.parent


def _sue(actual: float, mean: float, stdev: float) -> float:
    return (actual - mean) / max(stdev, 0.05 * abs(mean), 0.005)


def _value_chain(trace: dict[str, Any]) -> list[float]:
    """同一个实际值沿链条的三次出现；干净轨迹里三者相等。"""
    e, _x, c, d, _s = [a["payload"] for a in trace["artifacts"]]
    return [float(e["normalized_value"]), float(c["normalized_value"]),
            float(d["value_used"])]


def declared(trace: dict[str, Any], clean_trace: dict[str, Any]) -> tuple[float, str]:
    """工作流忠实传播被污染输入后会声明的 SUE 与档位。

    单点注入只会打破链条上的一环，因此「忠实传播」= 取最早偏离干净基准的那个值，
    并让它流到下游。均值与离散度取（可能被污染的）expectation 工件。

    口径边界：这测的是**已物化轨迹**的数值后果。像 period_mismatch 这类故障在现实中
    会换成另一期的共识数，但注入器只改期间键、不改估计值，因此其数值影响记为 0；
    它的实质性在于该比较不可采纳，而不在于数字变了多少。
    """
    e, x, _c, d, _s = [a["payload"] for a in trace["artifacts"]]
    baseline = _value_chain(clean_trace)[0]
    actual = next((v for v in _value_chain(trace) if v != baseline), baseline)
    sue = _sue(actual, float(x["mean_estimate"]), float(x["dispersion"]))
    return sue, "above_expectations" if sue > 0 else "below_expectations"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=str(ROOT / "data/faab/benchmark.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "data/faab/results/materiality.json"))
    args = parser.parse_args()

    cases = load_benchmark(args.benchmark)
    clean_by_source = {c["source_id"]: c for c in cases
                       if c["oracle"]["fault_type"] == "none"}
    rows: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    for case in cases:
        fault = case["oracle"]["fault_type"]
        if fault == "none":
            continue
        pair = clean_by_source.get(case["source_id"])
        if pair is None:
            continue
        clean_sue, clean_claim = declared(pair["trace"], pair["trace"])
        faulty_sue, faulty_claim = declared(case["trace"], pair["trace"])
        rows[fault].append({
            "abs_delta_sue": abs(faulty_sue - clean_sue),
            "claim_reversed": faulty_claim != clean_claim,
            "sign_reversed": (clean_sue > 0) != (faulty_sue > 0),
        })

    summary = {}
    for fault in FAULT_SPECS:
        group = rows.get(fault, [])
        if not group:
            continue
        deltas = [float(r["abs_delta_sue"]) for r in group]
        summary[fault] = {
            "n": len(group),
            "declared_materiality": FAULT_SPECS[fault]["materiality"],
            "changes_declared_number": sum(d > 1e-9 for d in deltas) / len(group),
            "median_abs_delta_sue": statistics.median(deltas),
            "max_abs_delta_sue": max(deltas),
            "claim_reversal_rate": sum(bool(r["claim_reversed"]) for r in group) / len(group),
        }

    numeric = [f for f, v in summary.items() if v["changes_declared_number"] > 0]
    admissibility = [f for f in summary if f not in numeric]
    output = {
        "definition": "SUE = (actual - mean) / max(stdev, 0.05*|mean|, 0.005)",
        "per_fault": summary,
        "magnitude_faults": sorted(numeric),
        "admissibility_faults": sorted(admissibility),
        "note": ("Faults outside magnitude_faults leave the declared number intact; "
                 "their materiality is admissibility of the claim, not its size."),
    }
    Path(args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"{'fault':<22}{'decl':<8}{'changes#':>9}{'med|dSUE|':>11}{'claimflip':>11}")
    for fault, row in summary.items():
        print(f"{fault:<22}{row['declared_materiality']:<8}"
              f"{row['changes_declared_number']:>9.2f}"
              f"{row['median_abs_delta_sue']:>11.3f}"
              f"{row['claim_reversal_rate']:>11.2f}")
    print(f"\nmagnitude faults ({len(numeric)}): {sorted(numeric)}")
    print(f"admissibility faults ({len(admissibility)}): {sorted(admissibility)}")


if __name__ == "__main__":
    main()
