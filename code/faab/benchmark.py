"""从真实 SEC/IBES 账本构造 FAAB clean/corrupted 成对轨迹。

原则：
1. 原始 filing、引证、时间戳、ticker 与共识快照均来自 data/us500.db；
2. 每个 corrupted case 只注入一个预先声明的最小故障；
3. oracle 标签与公开 trace 分离，审计器永远看不到 oracle；
4. 同一 source 的 clean/fault 变体固定进入同一 split。
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "faab" / "benchmark.jsonl"
SEED = 20260726

FAULT_SPECS: dict[str, dict[str, Any]] = {
    "fabricated_value": {"agent": "evidence_agent", "step": 1, "materiality": "high"},
    "quote_mismatch": {"agent": "evidence_agent", "step": 1, "materiality": "medium"},
    "unit_scaling": {"agent": "evidence_agent", "step": 1, "materiality": "high"},
    "basis_conflict": {"agent": "expectation_agent", "step": 2, "materiality": "high"},
    "period_mismatch": {"agent": "expectation_agent", "step": 2, "materiality": "high"},
    "entity_confusion": {"agent": "evidence_agent", "step": 1, "materiality": "high"},
    "stale_expectation": {"agent": "expectation_agent", "step": 2, "materiality": "medium"},
    "future_leakage": {"agent": "expectation_agent", "step": 2, "materiality": "high"},
    "definition_conflict": {"agent": "expectation_agent", "step": 2, "materiality": "high"},
    "missing_handoff": {"agent": "decision_agent", "step": 4, "materiality": "medium"},
    "role_violation": {"agent": "evidence_agent", "step": 1, "materiality": "medium"},
    "silent_overwrite": {"agent": "decision_agent", "step": 4, "materiality": "high"},
    "wrong_security": {"agent": "settlement_agent", "step": 5, "materiality": "high"},
    "wrong_window": {"agent": "settlement_agent", "step": 5, "materiality": "medium"},
}

_NUM = re.compile(r"(?<![\w.])(?:\$|US\$)?\s*(-?\d[\d,]*(?:\.\d+)?)\s*(%|percent|million|billion|thousand)?",
                  re.I)
_EPS = re.compile(r"\b(?:EPS|earnings per (?:common |diluted |basic )?share)\b", re.I)
_FORECAST = re.compile(
    r"\b(?:guidance|outlook|forecast|expects?|expected|anticipates?|projects?|"
    r"target|range|raising|raises|reaffirm|maintain)\b",
    re.I,
)
_NON_GAAP = re.compile(r"\b(?:adjusted|non-gaap|core|comparable)\b", re.I)


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s[:10])


def _iso(d: dt.date) -> str:
    return d.isoformat() + "T16:00:00-05:00"


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _value_token(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _contains_value(span: str, value: float) -> bool:
    """数值必须直接出现在引文中；不把百分比、年份或近似数字当作命中。"""
    for m in _NUM.finditer(span):
        if m.group(2) and m.group(2).lower() in {
            "%", "percent", "million", "billion", "thousand",
        }:
            continue
        candidate = float(m.group(1).replace(",", ""))
        if abs(candidate - float(value)) <= 1e-6:
            return True
    return False


def _same_eps_scale(filing_actual: float, ibes_actual: float | None) -> bool:
    """用 I/B/E/S 同源 actual 排除拆股回溯调整和会计口径错配。"""
    if ibes_actual is None:
        return False
    actual = float(ibes_actual)
    tolerance = max(0.01, 0.01 * max(abs(filing_actual), abs(actual)))
    return abs(float(filing_actual) - actual) <= tolerance


def _extract_eps_span(quotes: str, text: str, actual_eps: float) -> tuple[str, float] | None:
    """只接受明确支持数据库 actual_eps 的已实现 EPS 引文。

    FAAB 的 clean trace 不能把股息、年度指引或区间端点伪装成季度实际 EPS。
    这里宁可丢弃模糊样本，也不依赖 LLM 替 benchmark 猜语义。
    """
    parts = [_norm_text(p) for p in re.split(r"[;\n]+", quotes) if _norm_text(p)]
    text_n = _norm_text(text).lower()
    for p in parts:
        if not _EPS.search(p):
            continue
        lower = p.lower()
        if _FORECAST.search(p) or "dividend" in lower or "basic earnings" in lower:
            continue
        if not _contains_value(p, actual_eps):
            continue
        if _norm_text(p).lower() in text_n:
            return p, float(actual_eps)
    return None


def _decoy_span(text: str, evidence: str) -> str:
    """从同一真实 filing 取另一段作为错误引证，避免跨公司拼接。"""
    text_n = _norm_text(text)
    chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", text_n) if 25 <= len(c.strip()) <= 260]
    ev = evidence.lower()
    for c in chunks:
        cl = c.lower()
        if cl == ev or evidence.lower() in cl or _EPS.search(c):
            continue
        if any(k in cl for k in ("dividend", "per share", "guidance", "outlook")):
            continue
        if any(k in cl for k in ("revenue", "sales", "cash", "quarter")):
            return c
    return next((c for c in chunks if c.lower() != ev), "The filing also contains other corporate disclosures.")


def _consensus_index(store: "Store") -> tuple[dict[str, list[tuple[str, str]]],
                                            dict[tuple[str, str], list[dict[str, Any]]]]:
    ann: dict[str, list[tuple[str, str]]] = defaultdict(list)
    snaps: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in store.q("SELECT * FROM consensus WHERE num_est>0 AND mean_est IS NOT NULL"):
        r = dict(row)
        snaps[(r["ticker"], r["fpedats"])].append(r)
        pair = (r["anndats"], r["fpedats"])
        if r["anndats"] and pair not in ann[r["ticker"]]:
            ann[r["ticker"]].append(pair)
    for rows in snaps.values():
        rows.sort(key=lambda x: x["statpers"])
    return ann, snaps


def _matching_snapshots(ticker: str, t_pub: str, ann: dict, snaps: dict) -> list[dict[str, Any]]:
    d = _date(t_pub)
    candidates = [(abs((_date(a) - d).days), f) for a, f in ann.get(ticker, ())
                  if abs((_date(a) - d).days) <= 3]
    if not candidates:
        return []
    _, fpe = min(candidates)
    return [r for r in snaps.get((ticker, fpe), []) if _date(r["statpers"]) < d]


def _make_artifact(step: int, agent: str, payload: dict[str, Any],
                   parent_hashes: list[str]) -> dict[str, Any]:
    a = {"step": step, "agent": agent, "schema_version": "faab-1.0",
         "parent_hashes": parent_hashes, "payload": payload}
    a["hash"] = _hash(a)
    return a


def _rehash(trace: dict[str, Any]) -> None:
    parent = trace["source"]["hash"]
    for i, artifact in enumerate(trace["artifacts"]):
        # 合法传播：上游工件版本变化后，下游显式引用同步；空 handoff 保持为空，
        # 以免把 missing_handoff 注入器“修好”。
        if i == 2:
            artifact["payload"]["actual_ref"] = trace["artifacts"][0]["hash"]
            artifact["payload"]["expectation_ref"] = trace["artifacts"][1]["hash"]
        elif i == 3:
            artifact["payload"]["source_ref"] = trace["artifacts"][2]["hash"]
            if artifact["payload"].get("qualification_refs"):
                artifact["payload"]["qualification_refs"] = [trace["artifacts"][2]["hash"]]
        artifact["parent_hashes"] = [parent]
        tmp = {k: v for k, v in artifact.items() if k != "hash"}
        artifact["hash"] = _hash(tmp)
        parent = artifact["hash"]


def _clean_trace(row: dict[str, Any], evidence: str, raw_value: float, decoy: str,
                 snapshots: list[dict[str, Any]], other_ticker: str) -> dict[str, Any]:
    t_pub = row["t_pub"]
    latest, stale = snapshots[-1], snapshots[-2]
    basis = "non_gaap" if _NON_GAAP.search(evidence) else "reported"
    definition = "adjusted_diluted_eps" if basis == "non_gaap" else "diluted_eps"
    # 季度自然语言标签来自抽取器，可能把 fiscal Q2 写成 calendar Q3；
    # I/B/E/S FPEDATS 是匹配共识时使用的确定性财季连接键，公开 trace 统一采用它。
    period = latest["fpedats"]
    source_payload = {
        "msg_id": row["msg_id"], "ticker": row["ticker"], "period": period,
        "t_pub": t_pub, "document_excerpt": evidence + "\n\n" + decoy,
        "control_horizon_td": 3,
    }
    source = {**source_payload, "hash": _hash(source_payload)}
    evidence_payload = {
        "entity": row["ticker"], "metric": "eps", "definition": definition,
        "raw_value": raw_value, "unit": "USD_per_share", "unit_multiplier": 1.0,
        "normalized_value": raw_value, "basis": basis, "period": period,
        "quote": evidence, "recommendation": None,
    }
    a1 = _make_artifact(1, "evidence_agent", evidence_payload, [source["hash"]])
    mean = float(latest["mean_est"])
    stdev = float(latest["stdev"] or max(abs(mean) * 0.05, 0.005))
    expectation_payload = {
        "entity": row["ticker"], "metric": "eps", "definition": definition,
        "mean_estimate": mean, "dispersion": stdev, "basis": basis, "period": period,
        "fiscal_period_end": latest["fpedats"], "vintage": latest["statpers"],
        "visible_at": _iso(_date(latest["statpers"])),
        "latest_eligible_vintage": latest["statpers"],
        "prior_eligible_vintage": stale["statpers"],
    }
    a2 = _make_artifact(2, "expectation_agent", expectation_payload, [a1["hash"]])
    sue = (raw_value - mean) / max(stdev, 0.05 * abs(mean), 0.005)
    consistency_payload = {
        "aligned": True, "actual_ref": a1["hash"], "expectation_ref": a2["hash"],
        "normalized_value": raw_value, "standardized_surprise": round(sue, 6),
        "qualification": f"{basis}:{definition}:{period}",
    }
    a3 = _make_artifact(3, "consistency_agent", consistency_payload, [a2["hash"]])
    decision_payload = {
        "claim": "above_expectations" if sue > 0 else "below_expectations",
        "value_used": raw_value, "source_ref": a3["hash"],
        "qualification_refs": [a3["hash"]], "override_reason": None,
        "decision_time": t_pub,
    }
    a4 = _make_artifact(4, "decision_agent", decision_payload, [a3["hash"]])
    settlement_payload = {
        "ticker": row["ticker"], "benchmark_ticker": other_ticker,
        "horizon_td": 3, "decision_time": t_pub,
        "observation_time": _iso(_date(t_pub) + dt.timedelta(days=5)),
    }
    a5 = _make_artifact(5, "settlement_agent", settlement_payload, [a4["hash"]])
    return {"source": source, "artifacts": [a1, a2, a3, a4, a5]}


def _inject(clean: dict[str, Any], fault: str, other_ticker: str) -> dict[str, Any]:
    trace = copy.deepcopy(clean)
    e, x, c, d, s = [a["payload"] for a in trace["artifacts"]]
    if fault == "fabricated_value":
        e["raw_value"] = round(float(e["raw_value"]) + 0.37, 4)
        e["normalized_value"] = e["raw_value"]
    elif fault == "quote_mismatch":
        e["quote"] = trace["source"]["document_excerpt"].split("\n\n", 1)[1]
    elif fault == "unit_scaling":
        e["unit_multiplier"] = 1000.0
        e["normalized_value"] = round(float(e["raw_value"]) * 1000.0, 4)
    elif fault == "basis_conflict":
        x["basis"] = "reported" if e["basis"] == "non_gaap" else "non_gaap"
    elif fault == "period_mismatch":
        x["period"] = "FY" if not str(e["period"]).startswith("FY") else "Q1"
    elif fault == "entity_confusion":
        e["entity"] = other_ticker
    elif fault == "stale_expectation":
        x["vintage"] = x["prior_eligible_vintage"]
        x["visible_at"] = _iso(_date(x["prior_eligible_vintage"]))
    elif fault == "future_leakage":
        future = _date(d["decision_time"]) + dt.timedelta(days=1)
        x["vintage"] = future.isoformat()
        x["visible_at"] = _iso(future)
    elif fault == "definition_conflict":
        x["definition"] = "basic_eps" if e["definition"] != "basic_eps" else "diluted_eps"
    elif fault == "missing_handoff":
        d["qualification_refs"] = []
    elif fault == "role_violation":
        e["recommendation"] = "buy"
    elif fault == "silent_overwrite":
        d["value_used"] = round(float(c["normalized_value"]) + 0.51, 4)
        d["override_reason"] = None
    elif fault == "wrong_security":
        s["ticker"] = other_ticker
    elif fault == "wrong_window":
        s["horizon_td"] = 10
    else:
        raise KeyError(fault)
    _rehash(trace)
    return trace


def _candidate_rows(store: "Store") -> list[dict[str, Any]]:
    rows = store.q(
        "SELECT e.*,m.t_pub,f.report_date,t.text FROM earnings_facts e "
        "JOIN messages m USING(msg_id) JOIN filings f USING(msg_id) "
        "JOIN filing_texts t USING(msg_id) "
        "WHERE e.quotes IS NOT NULL AND TRIM(e.quotes)<>'' "
        "AND e.ticker IS NOT NULL AND e.period IS NOT NULL")
    return [dict(r) for r in rows]


def build_benchmark(db: str | Path = ROOT / "data" / "us500.db",
                    out: str | Path = DEFAULT_OUT,
                    n_clean: int = 300,
                    n_per_fault: int = 50,
                    seed: int = SEED) -> list[dict[str, Any]]:
    from world.store import Store  # 惰性导入：仅构建/校验需要源库

    store = Store(str(db))
    ann, snaps = _consensus_index(store)
    rng = random.Random(seed)
    tickers = [r["ticker"] for r in store.q(
        "SELECT DISTINCT ticker FROM earnings_facts WHERE ticker IS NOT NULL ORDER BY ticker")]
    candidates: list[tuple[dict, str, float, str, list[dict]]] = []
    for row in _candidate_rows(store):
        actual_eps = float(row["actual_eps"] or 0.0)
        if actual_eps == 0.0 or abs(actual_eps) > 100:
            continue
        hit = _extract_eps_span(row["quotes"], row["text"], actual_eps)
        ss = _matching_snapshots(row["ticker"], row["t_pub"], ann, snaps)
        if not hit or len(ss) < 2:
            continue
        quote, value = hit
        # I/B/E/S consensus 与 actual 是同一 non-GAAP 口径；只保留 filing 引文也明确
        # 为 non-GAAP 且二者同尺度的事件，避免拆股回溯或 GAAP/non-GAAP 假冲突。
        if not _NON_GAAP.search(quote) or not _same_eps_scale(value, ss[-1].get("actual")):
            continue
        candidates.append((row, quote, value, _decoy_span(row["text"], quote), ss))
    rng.shuffle(candidates)
    if len(candidates) < n_clean:
        raise RuntimeError(f"可用基础轨迹仅 {len(candidates)}，少于 n_clean={n_clean}")

    bases = candidates[:n_clean]
    clean_by_id: dict[str, dict] = {}
    other_by_id: dict[str, str] = {}
    cases: list[dict[str, Any]] = []
    # 前 20% source 进入 dev，剩余进入 test；所有变体继承该 split。
    n_dev = max(1, n_clean // 5)
    for i, (row, quote, value, decoy, ss) in enumerate(bases):
        other = next(t for t in tickers if t != row["ticker"])
        trace = _clean_trace(row, quote, value, decoy, ss, other)
        source_id = row["msg_id"]
        split = "dev" if i < n_dev else "test"
        clean_by_id[source_id] = trace
        other_by_id[source_id] = other
        cases.append({
            "case_id": f"clean_{i:04d}", "source_id": source_id, "split": split,
            "trace": trace,
            "oracle": {"fault_type": "none", "responsible_agent": "none",
                       "decisive_step": 0, "materiality": "none", "verdict": "pass"},
        })

    dev_ids = [c["source_id"] for c in cases if c["split"] == "dev"]
    test_ids = [c["source_id"] for c in cases if c["split"] == "test"]
    n_fault_dev = max(1, n_per_fault // 5)
    for fault, spec in FAULT_SPECS.items():
        chosen = rng.sample(dev_ids, n_fault_dev) + rng.sample(test_ids, n_per_fault - n_fault_dev)
        for j, source_id in enumerate(chosen):
            split = "dev" if j < n_fault_dev else "test"
            trace = _inject(clean_by_id[source_id], fault, other_by_id[source_id])
            cases.append({
                "case_id": f"{fault}_{j:03d}", "source_id": source_id, "split": split,
                "trace": trace,
                "oracle": {"fault_type": fault, "responsible_agent": spec["agent"],
                           "decisive_step": spec["step"], "materiality": spec["materiality"],
                           "verdict": "reject"},
            })
    rng.shuffle(cases)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n")
    db_path = Path(db)
    try:
        db_ref = str(db_path.resolve().relative_to(ROOT))
    except ValueError:
        db_ref = db_path.name
    manifest = {
        "version": "faab-1.4", "seed": seed, "db": db_ref, "n_cases": len(cases),
        "db_sha256": hashlib.sha256(db_path.read_bytes()).hexdigest(),
        "n_clean": n_clean, "n_per_fault": n_per_fault,
        "clean_policy": (
            "exact filing actual_eps; explicit non-GAAP EPS; no forecast/dividend/basic ambiguity; "
            "filing actual aligned to IBES same-source actual within max($0.01, 1%)"
        ),
        "fault_types": list(FAULT_SPECS), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return cases


def load_benchmark(path: str | Path = DEFAULT_OUT) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def public_trace(case: dict[str, Any]) -> dict[str, Any]:
    """剥掉 oracle 与可读 case_id；后者包含故障类前缀，绝不能进入模型输入。"""
    return {"trace_id": _hash(case["case_id"]), "source_id": case["source_id"],
            "trace": case["trace"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "us500.db"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n-clean", type=int, default=300)
    ap.add_argument("--n-per-fault", type=int, default=50)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    cases = build_benchmark(args.db, args.out, args.n_clean, args.n_per_fault, args.seed)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for c in cases:
        counts[(c["split"], c["oracle"]["fault_type"])] += 1
    print(f"FAAB: {len(cases)} cases -> {args.out}")
    print(json.dumps({f"{s}:{f}": n for (s, f), n in sorted(counts.items())}, indent=2))


if __name__ == "__main__":
    main()
