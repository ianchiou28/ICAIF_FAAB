"""Measure the actual effect of each fault class on the declared standardized surprise
(deterministic, no model calls).

Motivation: the original materiality was a high/medium dichotomy fixed per fault template.
That does not match the quantitative meaning of materiality in accounting audit, and it was
not tied to any decision convention. Here it is made measurable: for every paired
clean/corrupt trace, recompute SUE = (actual - mean) / max(stdev, 5%|mean|, 0.005) under
"the workflow faithfully propagates the contaminated input", then check whether the declared
tier flips.

The resulting empirical fact: only faults that change a numeric input change the declared
number; the rest leave the number intact while destroying its support and admissibility.
FAAB materiality is therefore admissibility, not magnitude.
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
    """The three appearances of the same actual value along the chain; equal on a clean trace."""
    e, _x, c, d, _s = [a["payload"] for a in trace["artifacts"]]
    return [float(e["normalized_value"]), float(c["normalized_value"]),
            float(d["value_used"])]


def declared(trace: dict[str, Any], clean_trace: dict[str, Any]) -> tuple[float, str]:
    """The SUE and tier the workflow would declare after faithfully propagating a
    contaminated input.

    A single-point injection breaks only one link in the chain, so "faithful propagation"
    means: take the earliest value that deviates from the clean baseline and let it flow
    downstream. Mean and dispersion come from the (possibly contaminated) expectation
    artifact.

    Scope boundary: this measures the numeric consequence of the **materialized trace**.
    A fault such as period_mismatch would in reality swap in another period's consensus,
    but the injector only changes the period key and not the estimate, so its numeric
    effect is recorded as 0; its materiality lies in the comparison being inadmissible,
    not in how much the number moved.
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
