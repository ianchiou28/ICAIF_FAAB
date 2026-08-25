"""Build the sealed rule-induction challenge set from FAAB v1.4 sources left unused.

The challenge keeps the same 14 financial control contracts but swaps in different sources,
numeric perturbations and some alternative fault expressions. Once generated it is frozen by
manifest hash; the rule inducer may only use the v1.4 development split.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


from .benchmark import (
    DEFAULT_OUT,
    FAULT_SPECS,
    ROOT,
    _NON_GAAP,
    _candidate_rows,
    _clean_trace,
    _consensus_index,
    _contains_value,
    _date,
    _decoy_span,
    _extract_eps_span,
    _hash,
    _iso,
    _matching_snapshots,
    _rehash,
    _same_eps_scale,
    load_benchmark,
)

CHALLENGE_SEED = 20260727
DEFAULT_CHALLENGE = ROOT / "data" / "faab_rule_induction" / "challenge.jsonl"


def _eligible_unused(db: str | Path, base_benchmark: str | Path) -> tuple[list[tuple], list[str]]:
    from world.store import Store  # Lazy import: only building/validating needs the source DB.

    store = Store(str(db))
    used = {c["source_id"] for c in load_benchmark(base_benchmark)}
    ann, snaps = _consensus_index(store)
    tickers = [
        r["ticker"] for r in store.q(
            "SELECT DISTINCT ticker FROM earnings_facts "
            "WHERE ticker IS NOT NULL ORDER BY ticker")
    ]
    candidates: list[tuple] = []
    seen: set[str] = set()
    for row in _candidate_rows(store):
        if row["msg_id"] in used or row["msg_id"] in seen:
            continue
        actual_eps = float(row["actual_eps"] or 0.0)
        if actual_eps == 0.0 or abs(actual_eps) > 100:
            continue
        hit = _extract_eps_span(row["quotes"], row["text"], actual_eps)
        snapshots = _matching_snapshots(row["ticker"], row["t_pub"], ann, snaps)
        if not hit or len(snapshots) < 2:
            continue
        quote, value = hit
        if not _NON_GAAP.search(quote):
            continue
        if not _same_eps_scale(value, snapshots[-1].get("actual")):
            continue
        seen.add(row["msg_id"])
        candidates.append(
            (row, quote, value, _decoy_span(row["text"], quote), snapshots))
    return candidates, tickers


def _rehash_decision_and_settlement(trace: dict[str, Any]) -> None:
    """Keep the decision.source_ref fault while letting the artifact chain itself still
    propagate legitimately."""
    decision = trace["artifacts"][3]
    decision["parent_hashes"] = [trace["artifacts"][2]["hash"]]
    decision["hash"] = _hash({k: v for k, v in decision.items() if k != "hash"})
    settlement = trace["artifacts"][4]
    settlement["parent_hashes"] = [decision["hash"]]
    settlement["hash"] = _hash({k: v for k, v in settlement.items() if k != "hash"})


def _inject_challenge(clean: dict[str, Any], fault: str, other_ticker: str) -> dict[str, Any]:
    """Same contracts as the training injector but different parameters; missing handoff
    becomes a lost source_ref."""
    trace = copy.deepcopy(clean)
    e, x, c, d, s = [a["payload"] for a in trace["artifacts"]]
    document = trace["source"]["document_excerpt"]
    if fault == "fabricated_value":
        original = float(e["raw_value"])
        for delta in (0.73, -0.61, 1.37, -1.19):
            candidate = round(original + delta, 4)
            if not _contains_value(document, candidate):
                e["raw_value"] = candidate
                e["normalized_value"] = candidate
                break
    elif fault == "quote_mismatch":
        e["quote"] = document.split("\n\n", 1)[1]
    elif fault == "unit_scaling":
        e["unit_multiplier"] = 1_000_000.0
        e["normalized_value"] = round(float(e["raw_value"]) * 1_000_000.0, 4)
    elif fault == "basis_conflict":
        x["basis"] = "reported" if e["basis"] != "reported" else "non_gaap"
    elif fault == "period_mismatch":
        x["period"] = f"{e['period']}_PRIOR"
    elif fault == "entity_confusion":
        e["entity"] = other_ticker
    elif fault == "stale_expectation":
        x["vintage"] = x["prior_eligible_vintage"]
        x["visible_at"] = _iso(_date(x["prior_eligible_vintage"]))
    elif fault == "future_leakage":
        future = _date(d["decision_time"]) + dt.timedelta(days=2)
        x["vintage"] = future.isoformat()
        x["visible_at"] = _iso(future)
    elif fault == "definition_conflict":
        x["definition"] = "basic_eps"
    elif fault == "missing_handoff":
        # The training injection deletes qualification_refs; the challenge drops source_ref instead.
        d["source_ref"] = ""
    elif fault == "role_violation":
        e["recommendation"] = "hold"
    elif fault == "silent_overwrite":
        d["value_used"] = round(float(c["normalized_value"]) - 0.29, 4)
        d["override_reason"] = ""
    elif fault == "wrong_security":
        s["ticker"] = other_ticker
    elif fault == "wrong_window":
        s["horizon_td"] = int(trace["source"]["control_horizon_td"]) + 2
    else:
        raise KeyError(fault)

    if fault == "missing_handoff":
        _rehash_decision_and_settlement(trace)
    else:
        _rehash(trace)
    return trace


def build_challenge(
    db: str | Path = ROOT / "data" / "us500.db",
    base_benchmark: str | Path = DEFAULT_OUT,
    out: str | Path = DEFAULT_CHALLENGE,
    n_clean: int = 60,
    n_per_fault: int = 10,
    seed: int = CHALLENGE_SEED,
) -> list[dict[str, Any]]:
    candidates, tickers = _eligible_unused(db, base_benchmark)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if len(candidates) < n_clean:
        raise RuntimeError(
            f"only {len(candidates)} unused sources meet the strict clean condition, "
            f"fewer than {n_clean}")
    bases = candidates[:n_clean]
    clean_by_id: dict[str, dict[str, Any]] = {}
    other_by_id: dict[str, str] = {}
    cases: list[dict[str, Any]] = []
    for i, (row, quote, value, decoy, snapshots) in enumerate(bases):
        other = next(ticker for ticker in tickers if ticker != row["ticker"])
        trace = _clean_trace(row, quote, value, decoy, snapshots, other)
        source_id = row["msg_id"]
        clean_by_id[source_id] = trace
        other_by_id[source_id] = other
        cases.append({
            "case_id": f"challenge_clean_{i:03d}",
            "source_id": source_id,
            "split": "challenge",
            "trace": trace,
            "oracle": {
                "fault_type": "none", "responsible_agent": "none",
                "decisive_step": 0, "materiality": "none", "verdict": "pass",
            },
        })
    source_ids = list(clean_by_id)
    for fault, spec in FAULT_SPECS.items():
        for i, source_id in enumerate(rng.sample(source_ids, n_per_fault)):
            cases.append({
                "case_id": f"challenge_{fault}_{i:03d}",
                "source_id": source_id,
                "split": "challenge",
                "trace": _inject_challenge(
                    clean_by_id[source_id], fault, other_by_id[source_id]),
                "oracle": {
                    "fault_type": fault,
                    "responsible_agent": spec["agent"],
                    "decisive_step": spec["step"],
                    "materiality": spec["materiality"],
                    "verdict": "reject",
                },
            })
    rng.shuffle(cases)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n")
    db_path = Path(db)
    base_path = Path(base_benchmark)
    manifest = {
        "version": "faab-rule-induction-challenge-1.0",
        "seed": seed,
        "n_cases": len(cases),
        "n_clean": n_clean,
        "n_per_fault": n_per_fault,
        "n_sources": n_clean,
        "source_overlap_with_faab_v1_4": 0,
        "base_benchmark_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "db_sha256": hashlib.sha256(db_path.read_bytes()).hexdigest(),
        "fault_types": list(FAULT_SPECS),
        "variant_policy": {
            "new_sources": True,
            "new_numeric_perturbations": True,
            "missing_handoff": "decision.source_ref removed instead of qualification_refs",
        },
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    return cases


def validate_challenge(
    challenge: str | Path = DEFAULT_CHALLENGE,
    base_benchmark: str | Path = DEFAULT_OUT,
) -> dict[str, Any]:
    cases = load_benchmark(challenge)
    base_sources = {case["source_id"] for case in load_benchmark(base_benchmark)}
    clean = {
        case["source_id"]: case
        for case in cases if case["oracle"]["fault_type"] == "none"
    }
    counts = Counter(case["oracle"]["fault_type"] for case in cases)
    errors: list[str] = []
    overlap = base_sources & set(clean)
    if overlap:
        errors.append(f"source_overlap:{len(overlap)}")
    for case in cases:
        if case["split"] != "challenge":
            errors.append(f"{case['case_id']}:wrong_split")
        if case["oracle"]["fault_type"] != "none" and case["source_id"] not in clean:
            errors.append(f"{case['case_id']}:missing_clean_pair")
    return {
        "valid": not errors,
        "n_cases": len(cases),
        "n_sources": len(clean),
        "source_overlap": len(overlap),
        "counts": dict(sorted(counts.items())),
        "errors": errors[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "data" / "us500.db"))
    parser.add_argument("--base-benchmark", default=str(DEFAULT_OUT))
    parser.add_argument("--out", default=str(DEFAULT_CHALLENGE))
    parser.add_argument("--n-clean", type=int, default=60)
    parser.add_argument("--n-per-fault", type=int, default=10)
    parser.add_argument("--seed", type=int, default=CHALLENGE_SEED)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        build_challenge(
            args.db, args.base_benchmark, args.out,
            args.n_clean, args.n_per_fault, args.seed)
    summary = validate_challenge(args.out, args.base_benchmark)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(0 if summary["valid"] else 1)


if __name__ == "__main__":
    main()
