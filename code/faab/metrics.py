"""FAAB metrics: clean FPR, detection, root cause, agent/step, materiality-weighted recall."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .benchmark import FAULT_SPECS

WEIGHT = {"none": 0, "low": 1, "medium": 2, "high": 3}


def detected(pred: dict[str, Any]) -> bool:
    return pred["verdict"] in ("qualified", "reject") and pred["fault_type"] != "none"


def evaluate(cases: list[dict[str, Any]], predictions: dict[str, dict[str, Any]],
             split: str = "test") -> dict[str, Any]:
    rows = [c for c in cases if c["split"] == split and c["case_id"] in predictions]
    clean = [c for c in rows if c["oracle"]["fault_type"] == "none"]
    faults = [c for c in rows if c["oracle"]["fault_type"] != "none"]
    fp = sum(detected(predictions[c["case_id"]]) for c in clean)
    tp = sum(detected(predictions[c["case_id"]]) for c in faults)
    root = sum(predictions[c["case_id"]]["fault_type"] == c["oracle"]["fault_type"] for c in faults)
    agent = sum(predictions[c["case_id"]]["responsible_agent"] ==
                c["oracle"]["responsible_agent"] for c in faults)
    step = sum(predictions[c["case_id"]]["decisive_step"] ==
               c["oracle"]["decisive_step"] for c in faults)
    joint = sum(
        predictions[c["case_id"]]["responsible_agent"] == c["oracle"]["responsible_agent"]
        and predictions[c["case_id"]]["decisive_step"] == c["oracle"]["decisive_step"]
        for c in faults)
    denom_w = sum(WEIGHT[c["oracle"]["materiality"]] for c in faults)
    hit_w = sum(WEIGHT[c["oracle"]["materiality"]] for c in faults
                if detected(predictions[c["case_id"]]))
    per_fault: dict[str, dict[str, Any]] = {}
    for f in FAULT_SPECS:
        group = [c for c in faults if c["oracle"]["fault_type"] == f]
        if not group:
            continue
        per_fault[f] = {
            "n": len(group),
            "detection_recall": sum(detected(predictions[c["case_id"]]) for c in group) / len(group),
            "root_cause_accuracy": sum(predictions[c["case_id"]]["fault_type"] == f for c in group) / len(group),
            "agent_accuracy": sum(predictions[c["case_id"]]["responsible_agent"] ==
                                  c["oracle"]["responsible_agent"] for c in group) / len(group),
            "step_accuracy": sum(predictions[c["case_id"]]["decisive_step"] ==
                                 c["oracle"]["decisive_step"] for c in group) / len(group),
        }
    # Macro root-cause F1 over 14 fault classes (clean "none" is excluded).
    f1s = []
    for f in FAULT_SPECS:
        tp_f = sum(c["oracle"]["fault_type"] == f and predictions[c["case_id"]]["fault_type"] == f
                   for c in faults)
        fp_f = sum(c["oracle"]["fault_type"] != f and predictions[c["case_id"]]["fault_type"] == f
                   for c in faults)
        fn_f = sum(c["oracle"]["fault_type"] == f and predictions[c["case_id"]]["fault_type"] != f
                   for c in faults)
        p = tp_f / (tp_f + fp_f) if tp_f + fp_f else 0.0
        r = tp_f / (tp_f + fn_f) if tp_f + fn_f else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    verdicts = Counter(predictions[c["case_id"]]["verdict"] for c in rows)
    # Responsible agent and decisive step are in one-to-one correspondence in this
    # workflow, so they are not independent dimensions. Measure that fact and report it
    # alongside the metrics; the paper reports only the merged responsibility localization.
    label_pairs = sorted({(c["oracle"]["responsible_agent"], c["oracle"]["decisive_step"])
                          for c in faults})
    bijective = (len({a for a, _ in label_pairs}) == len(label_pairs)
                 == len({s for _, s in label_pairs}))
    return {
        "split": split, "n": len(rows), "n_clean": len(clean), "n_fault": len(faults),
        "clean_fpr": fp / len(clean) if clean else None,
        "fault_detection_recall": tp / len(faults) if faults else None,
        "root_cause_accuracy": root / len(faults) if faults else None,
        "root_cause_macro_f1": sum(f1s) / len(f1s),
        "agent_attribution_accuracy": agent / len(faults) if faults else None,
        "decisive_step_accuracy": step / len(faults) if faults else None,
        "joint_agent_step_accuracy": joint / len(faults) if faults else None,
        # Paper convention: responsibility localization (agent and step both correct).
        "responsibility_localization_accuracy": joint / len(faults) if faults else None,
        "label_agent_step_pairs": [list(p) for p in label_pairs],
        "agent_step_bijective": bijective,
        "materiality_weighted_recall": hit_w / denom_w if denom_w else None,
        "unresolved_rate": verdicts["unresolved"] / len(rows) if rows else None,
        "verdict_counts": dict(verdicts), "per_fault": per_fault,
    }
