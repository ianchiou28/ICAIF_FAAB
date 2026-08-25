"""Independently validate a frozen FAAB: splits, single fault, source-text quotes,
label isolation and distribution."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .benchmark import DEFAULT_OUT, FAULT_SPECS, load_benchmark, public_trace


def _diff_paths(a: Any, b: Any, prefix: str = "") -> list[str]:
    if type(a) is not type(b):
        return [prefix]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            p = f"{prefix}.{k}" if prefix else k
            if k not in a or k not in b:
                out.append(p)
            else:
                out.extend(_diff_paths(a[k], b[k], p))
        return out
    if isinstance(a, list):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(_diff_paths(x, y, f"{prefix}[{i}]"))
        if len(a) != len(b):
            out.append(prefix + ".length")
        return out
    return [] if a == b else [prefix]


ALLOWED_CHANGED_PREFIXES = {
    "fabricated_value": ["artifacts[0].payload.raw_value", "artifacts[0].payload.normalized_value"],
    "quote_mismatch": ["artifacts[0].payload.quote"],
    "unit_scaling": ["artifacts[0].payload.unit_multiplier", "artifacts[0].payload.normalized_value"],
    "basis_conflict": ["artifacts[1].payload.basis"],
    "period_mismatch": ["artifacts[1].payload.period"],
    "entity_confusion": ["artifacts[0].payload.entity"],
    "stale_expectation": ["artifacts[1].payload.vintage", "artifacts[1].payload.visible_at"],
    "future_leakage": ["artifacts[1].payload.vintage", "artifacts[1].payload.visible_at"],
    "definition_conflict": ["artifacts[1].payload.definition"],
    "missing_handoff": ["artifacts[3].payload.qualification_refs"],
    "role_violation": ["artifacts[0].payload.recommendation"],
    "silent_overwrite": ["artifacts[3].payload.value_used"],
    "wrong_security": ["artifacts[4].payload.ticker"],
    "wrong_window": ["artifacts[4].payload.horizon_td"],
}


def validate(path: str | Path = DEFAULT_OUT,
             db: str | Path = Path(__file__).resolve().parent.parent / "data" / "us500.db"
             ) -> dict[str, Any]:
    cases = load_benchmark(path)
    from world.store import Store  # Lazy import: only building/validating needs the source DB.

    store = Store(str(db))
    filing_text = {r["msg_id"]: r["text"] for r in
                   store.q("SELECT msg_id,text FROM filing_texts")}
    message_time = {r["msg_id"]: r["t_pub"] for r in
                    store.q("SELECT msg_id,t_pub FROM messages")}
    clean = {c["source_id"]: c for c in cases if c["oracle"]["fault_type"] == "none"}
    errors: list[str] = []
    split_of: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    for case in cases:
        split_of[case["source_id"]].add(case["split"])
        fault = case["oracle"]["fault_type"]
        counts[(case["split"], fault)] += 1
        # The object handed to auditors must not contain the oracle.
        if "oracle" in json.dumps(public_trace(case), ensure_ascii=False):
            errors.append(f"{case['case_id']}: oracle leaked into public trace")
        src = case["trace"]["source"]
        if fault == "none":
            quote = case["trace"]["artifacts"][0]["payload"]["quote"]
            if quote not in src["document_excerpt"]:
                errors.append(f"{case['case_id']}: clean quote absent from source excerpt")
            raw = " ".join(filing_text.get(src["msg_id"], "").split()).lower()
            if " ".join(quote.split()).lower() not in raw:
                errors.append(f"{case['case_id']}: clean quote absent from original filing")
            if message_time.get(src["msg_id"]) != src["t_pub"]:
                errors.append(f"{case['case_id']}: source timestamp differs from ledger")
            continue
        if case["source_id"] not in clean:
            errors.append(f"{case['case_id']}: missing clean pair")
            continue
        base = clean[case["source_id"]]["trace"]
        # Hashes and explicit references change under legitimate propagation; they do not
        # count as a second semantic fault.
        changed = [p for p in _diff_paths(base, case["trace"])
                   if not p.endswith(".hash") and ".parent_hashes" not in p
                   and not p.endswith(".actual_ref") and not p.endswith(".expectation_ref")
                   and not p.endswith(".source_ref")]
        qprefix = "artifacts[3].payload.qualification_refs"
        if fault == "missing_handoff":
            changed = [qprefix if p.startswith(qprefix) else p for p in changed]
            changed = list(dict.fromkeys(changed))
        else:
            changed = [p for p in changed if not p.startswith(qprefix)]
        allowed = ALLOWED_CHANGED_PREFIXES[fault]
        unexpected = [p for p in changed if p not in allowed]
        missing = [p for p in allowed if p not in changed]
        if unexpected or missing:
            errors.append(f"{case['case_id']}: changed={changed} expected={allowed}")
    for source_id, splits in split_of.items():
        if len(splits) != 1:
            errors.append(f"{source_id}: crosses splits {sorted(splits)}")
    summary = {
        "valid": not errors, "n_cases": len(cases), "n_sources": len(clean),
        "counts": {f"{s}:{f}": n for (s, f), n in sorted(counts.items())},
        "errors": errors[:50],
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=str(DEFAULT_OUT))
    ap.add_argument("--db", default=str(Path(__file__).resolve().parent.parent /
                                        "data" / "us500.db"))
    args = ap.parse_args()
    summary = validate(args.benchmark, args.db)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(0 if summary["valid"] else 1)


if __name__ == "__main__":
    main()
