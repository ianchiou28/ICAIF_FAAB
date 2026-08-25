"""Run FAAB rule-only / LLM judge / debate / tool auditor / hybrid, saving per-case outputs
and metrics.

Usage:
  python3 -m faab.benchmark
  python3 eval/faab_experiment.py --methods rule_only
  python3 eval/faab_experiment.py --methods llm_judge debate hybrid --workers 12
  python3 eval/faab_experiment.py --methods tool_auditor --model gpt-5.6-sol --workers 8

Keys for the LLM modes are listed in VENDORS in faab/models.py. Result files for the default
model deepseek-chat carry no suffix (preserving the frozen v1.4 naming); other models are
written to <method>_<split>__<model>.jsonl. Runs are resumable, and completed cases are not
rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faab import models
from faab.auditors import (PROMPT_VERSION, DebateAuditor, HybridAuditor, LLMJudgeAuditor,
                          RuleAuditor, RulesRetrievalAuditor, normalize_result)
from faab.benchmark import DEFAULT_OUT, load_benchmark, public_trace
from faab.metrics import evaluate
from faab.tool_auditor import ToolAugmentedAuditor

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "faab" / "results"

LLM_METHODS = ("llm_judge", "debate", "hybrid", "tool_auditor")
DETERMINISTIC_METHODS = ("rule_only", "rules_retrieval")
# Runs made after a prompt revision are written to version-suffixed files and never mixed with
# the frozen v1.4 results. Deterministic arms do not depend on the prompt, so they get no suffix.
PROMPT_TAG = "" if PROMPT_VERSION.endswith("1.4") else (
    "__p" + PROMPT_VERSION.rsplit("-", 1)[-1].replace(".", ""))


class CachedCaller:
    def __init__(self, log_path: Path, model: str = models.DEFAULT_MODEL):
        self.model = models.ChatModel(model)
        self.model_spec = model
        self.lock = threading.Lock()
        self.log_path = log_path
        self.calls = 0
        self.failures = 0
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._request_seq = 0
        self._restore_log_totals()

    def _restore_log_totals(self) -> None:
        """Restore API statistics from before an interruption; the log holds no keys, so it can be
        archived alongside the per-case predictions."""
        if not self.log_path.exists():
            return
        for line in self.log_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            self.requests += int(row.get("attempts", 0))
            if row.get("status") == "ok":
                self.calls += 1
            else:
                self.failures += 1
            usage = row.get("usage") or {}
            self.prompt_tokens += int(usage.get("prompt_tokens", 0))
            self.completion_tokens += int(usage.get("completion_tokens", 0))
            self.total_tokens += int(usage.get("total_tokens", 0))
            self._request_seq = max(self._request_seq, int(row.get("request_id", 0)))

    def _append_log(self, row: dict[str, Any]) -> None:
        raw = json.dumps(row, ensure_ascii=False) + "\n"
        with self.lock:
            with self.log_path.open("a") as f:
                f.write(raw)

    def _ask_once(self, system: str, user: str, schema: dict) -> tuple[dict, dict[str, int]]:
        """One structured adjudication.

        The system assembly mirrors the v1.4 frozen run; the trailing field instruction below
        was originally written in Chinese and has since been translated, so prompts are no
        longer byte-identical to that run.
        """
        fields = {k: (v.get("enum") or v.get("type"))
                  for k, v in schema.get("properties", {}).items()}
        sys2 = system + "\n\nOutput only one JSON object, which must contain these fields " \
            "(values must be valid):\n" + \
            json.dumps(fields, ensure_ascii=False)
        reply, usage = self.model.complete(
            self.model.seed(sys2, user), max_tokens=600, json_object=True)
        return models.parse_json(reply.content), usage

    def _record(self, request_id: int, status: str, payload: dict[str, Any],
                attempts: int, usage: dict[str, int] | None = None) -> None:
        with self.lock:
            if status == "ok":
                self.calls += 1
            else:
                self.failures += 1
            self.requests += attempts
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                setattr(self, key, getattr(self, key) + (usage or {}).get(key, 0))
        self._append_log({
            "request_id": request_id, "status": status,
            "provider": self.model.vendor, "model": self.model.model_name,
            "attempts": attempts, "usage": usage or {}, **payload,
        })

    def chat(self, conversation: list[Any], tools: list[dict[str, Any]] | None = None,
             max_tokens: int = 1024, json_object: bool = False,
             purpose: str = "chat") -> tuple[Any, dict[str, int]]:
        """Multi-turn channel: returns (Reply, usage), or (None, {}) once retries are exhausted."""
        with self.lock:
            self._request_seq += 1
            request_id = self._request_seq
        # Log the full conversation on the first round and only the incremental tail afterwards:
        # tool results are a deterministic function of the trace, so it can be rebuilt in full.
        logged = conversation if len(conversation) <= 2 else conversation[2:]
        err = None
        attempts = 0
        for wait in (0, 1, 3, 8):
            if wait:
                time.sleep(wait)
            attempts += 1
            try:
                reply, usage = self.model.complete(
                    conversation, max_tokens=max_tokens, tools=tools, json_object=json_object)
                if json_object:
                    reply.parsed_json = models.parse_json(reply.content)
                self._record(request_id, "ok", {
                    "purpose": purpose, "n_items": len(conversation),
                    "conversation_tail": logged,
                    "tools": [t["function"]["name"] for t in (tools or [])],
                    "response": reply.content,
                    "tool_calls": [{"name": c.name, "arguments": c.arguments}
                                   for c in reply.tool_calls],
                }, attempts, usage)
                return reply, usage
            except Exception as exc:  # Retry transient API errors; give up as an explicit unresolved.
                err = exc
        self._record(request_id, "failed", {
            "purpose": purpose, "n_items": len(conversation), "conversation_tail": logged,
            "error_type": type(err).__name__, "error": str(err)[:400],
        }, attempts)
        return None, {}

    def __call__(self, system: str, user: str, schema: dict) -> dict:
        with self.lock:
            self._request_seq += 1
            request_id = self._request_seq
        err = None
        attempts = 0
        for wait in (0, 1, 3, 8):
            if wait:
                time.sleep(wait)
            attempts += 1
            try:
                out, usage = self._ask_once(system, user, schema)
                self._record(request_id, "ok", {
                    "system": system, "user": user, "schema": schema, "response": out,
                }, attempts, usage)
                return out
            except Exception as exc:  # Retry transient API errors; give up as an explicit unresolved.
                err = exc
        self._record(request_id, "failed", {
            "system": system, "user": user, "schema": schema,
            "error_type": type(err).__name__,
        }, attempts)
        return {"verdict": "unresolved", "fault_type": "none",
                "responsible_agent": "none", "decisive_step": 0,
                "materiality": "none", "evidence_refs": [],
                "rationale": f"api_failure:{type(err).__name__}"}


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["case_id"]] = r
    return out


def _append(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    raw = json.dumps(row, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a") as f:
            f.write(raw)


def run_method(method: str, cases: list[dict[str, Any]], split: str, workers: int,
               model: str = models.DEFAULT_MODEL, tag_suffix: str = "",
               out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ((models.slug(model) + PROMPT_TAG) if method in LLM_METHODS else "") + tag_suffix
    pred_path = out_dir / f"{method}_{split}{tag}.jsonl"
    existing = _load_existing(pred_path)
    target = [c for c in cases if c["split"] == split]
    if method in DETERMINISTIC_METHODS:
        auditor = {"rule_only": RuleAuditor,
                   "rules_retrieval": RulesRetrievalAuditor}[method]()
        caller = None
    elif method == "tool_auditor":
        caller = CachedCaller(out_dir / f"{method}_{split}{tag}_api.jsonl", model)
        auditor = ToolAugmentedAuditor(caller.chat, caller.model)
    else:
        caller = CachedCaller(out_dir / f"{method}_{split}{tag}_api.jsonl", model)
        auditor = {"llm_judge": LLMJudgeAuditor, "debate": DebateAuditor,
                   "hybrid": HybridAuditor}[method](caller)
    write_lock = threading.Lock()

    def one(case: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        raw = auditor.audit(public_trace(case))
        row = {"case_id": case["case_id"], "source_id": case["source_id"],
               "method": method, "prediction": normalize_result(raw),
               "latency_s": round(time.time() - started, 4)}
        if "tool_calls_made" in raw:  # Tool-auditor diagnostic; does not enter the metrics.
            row["tool_calls_made"] = raw["tool_calls_made"]
        return row

    todo = [c for c in target if c["case_id"] not in existing]
    if workers == 1:
        for i, case in enumerate(todo, 1):
            row = one(case)
            existing[row["case_id"]] = row
            _append(pred_path, row, write_lock)
            if i % 50 == 0:
                print(f"[{method}] {i}/{len(todo)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                existing[row["case_id"]] = row
                _append(pred_path, row, write_lock)
                if i % 50 == 0:
                    print(f"[{method}] {i}/{len(todo)}", flush=True)

    preds = {cid: row["prediction"] for cid, row in existing.items()}
    metrics = evaluate(cases, preds, split)
    metrics["method"] = method
    metrics["provider"] = caller.model.vendor if caller else "deterministic"
    metrics["model"] = caller.model.model_name if caller else None
    metrics["api_calls"] = caller.calls if caller else 0
    metrics["api_requests_including_retries"] = caller.requests if caller else 0
    metrics["api_failures"] = caller.failures if caller else 0
    metrics["prompt_tokens"] = caller.prompt_tokens if caller else 0
    metrics["completion_tokens"] = caller.completion_tokens if caller else 0
    metrics["total_tokens"] = caller.total_tokens if caller else 0
    metrics["mean_latency_s"] = (sum(row["latency_s"] for row in existing.values()) /
                                 len(existing) if existing else 0)
    (out_dir / f"{method}_{split}{tag}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=str(DEFAULT_OUT))
    ap.add_argument("--methods", nargs="+",
                    choices=[*DETERMINISTIC_METHODS, *LLM_METHODS],
                    default=["rule_only"])
    ap.add_argument("--split", choices=["dev", "test", "challenge"], default="test")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model", default=models.DEFAULT_MODEL,
                    help="model used by the LLM baselines, e.g. deepseek-reasoner or gpt-5.6-sol")
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N cases (pipeline check; 0 means all)")
    args = ap.parse_args()
    cases = load_benchmark(args.benchmark)
    if args.limit:
        keep = {c["case_id"] for c in
                [c for c in cases if c["split"] == args.split][:args.limit]}
        cases = [c for c in cases if c["split"] != args.split or c["case_id"] in keep]
    # Smoke runs write to their own files and never overwrite frozen results.
    tag_suffix = f"__smoke{args.limit}" if args.limit else ""
    for method in args.methods:
        metrics = run_method(method, cases, args.split,
                             1 if method in DETERMINISTIC_METHODS else args.workers, args.model,
                             tag_suffix, Path(args.out_dir))
        print(json.dumps({k: v for k, v in metrics.items() if k != "per_fault"},
                         indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
