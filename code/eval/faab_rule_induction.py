"""FAAB 离线规则归纳与 held-out challenge 评测。

流程分离：
1. induce 只读取 FAAB v1.4 development split，调用 DeepSeek 生成并修复 DSL；
2. freeze 固定规则、prompt、代码与 challenge 哈希；
3. evaluate 才读取 challenge labels，并运行确定性/在线基线。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import llm
from eval.faab_experiment import CachedCaller
from faab.auditors import (
    DebateAuditor,
    LLMJudgeAuditor,
    RuleAuditor,
    normalize_result,
)
from faab.benchmark import DEFAULT_OUT, load_benchmark, public_trace
from faab.challenge import DEFAULT_CHALLENGE, validate_challenge
from faab.metrics import evaluate
from faab.rule_dsl import (
    RuleSetAuditor,
    RuleValidationError,
    human_rule_set,
    validate_rule_set,
)
from faab.rule_induction import (
    INDUCTION_SYSTEM,
    build_training_bundle,
    development_feedback,
    evaluate_rules,
    induction_prompt,
    normalize_rule_response,
    repair_prompt,
    score_for_selection,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "faab_rule_induction" / "results"
FREEZE_PATH = ROOT / "data" / "faab_rule_induction" / "experiment_freeze.json"
DEFAULT_SEEDS = (101, 202, 303)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock | None = None) -> None:
    raw = json.dumps(row, ensure_ascii=False) + "\n"
    if lock is None:
        with path.open("a") as handle:
            handle.write(raw)
        return
    with lock:
        with path.open("a") as handle:
            handle.write(raw)


class InductionCaller:
    """大输出 JSON 调用器；日志保存 prompt/response/usage，但不保存密钥。"""

    def __init__(self, log_path: Path):
        self.model = llm.make_llm()
        if not isinstance(self.model, llm.DeepSeekLLM):
            raise RuntimeError("规则归纳实验需要 DEEPSEEK_API_KEY")
        self.log_path = log_path
        self.request_id = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._restore()

    def _restore(self) -> None:
        if not self.log_path.exists():
            return
        for line in self.log_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            self.request_id = max(self.request_id, int(row["request_id"]))
            self.successful_calls += int(row["status"] == "ok")
            self.failed_calls += int(row["status"] == "failed")
            usage = row.get("usage") or {}
            self.prompt_tokens += int(usage.get("prompt_tokens", 0))
            self.completion_tokens += int(usage.get("completion_tokens", 0))
            self.total_tokens += int(usage.get("total_tokens", 0))

    def __call__(self, system: str, user: str, purpose: str) -> dict[str, Any]:
        self.request_id += 1
        request_id = self.request_id
        error: Exception | None = None
        for attempt, wait in enumerate((0, 1, 3, 8), 1):
            if wait:
                time.sleep(wait)
            try:
                response = self.model.client.chat.completions.create(
                    model=self.model.model_name,
                    max_tokens=7000,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                )
                output = json.loads(response.choices[0].message.content)
                usage_obj = response.usage
                usage = {
                    "prompt_tokens": int(
                        getattr(usage_obj, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        getattr(usage_obj, "completion_tokens", 0) or 0),
                    "total_tokens": int(
                        getattr(usage_obj, "total_tokens", 0) or 0),
                }
                self.successful_calls += 1
                self.prompt_tokens += usage["prompt_tokens"]
                self.completion_tokens += usage["completion_tokens"]
                self.total_tokens += usage["total_tokens"]
                _append_jsonl(self.log_path, {
                    "request_id": request_id,
                    "status": "ok",
                    "purpose": purpose,
                    "provider": "deepseek",
                    "model": self.model.model_name,
                    "temperature": 0,
                    "attempts": attempt,
                    "system": system,
                    "user": user,
                    "response": output,
                    "usage": usage,
                })
                return output
            except Exception as exc:  # 网络和 JSON 暂态错误显式重试。
                error = exc
        self.failed_calls += 1
        _append_jsonl(self.log_path, {
            "request_id": request_id,
            "status": "failed",
            "purpose": purpose,
            "provider": "deepseek",
            "model": self.model.model_name,
            "temperature": 0,
            "attempts": 4,
            "system": system,
            "user": user,
            "error_type": type(error).__name__ if error else "UnknownError",
            "usage": {},
        })
        raise RuntimeError(f"DeepSeek rule induction failed: {type(error).__name__}")

    def usage(self) -> dict[str, Any]:
        return {
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def _invalid_repair_prompt(response: Any, error: str) -> str:
    return (
        "The previous JSON failed the static DSL validator. Return a complete corrected "
        "14-rule object. Do not remove fault classes and do not use string constants.\n\n"
        + json.dumps(
            {"validator_error": error, "previous_response": response},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def induce_one(
    caller: InductionCaller,
    cases: list[dict[str, Any]],
    seed: int,
    max_rounds: int,
) -> dict[str, Any]:
    bundle = build_training_bundle(cases, seed=seed)
    response: Any = caller(
        INDUCTION_SYSTEM,
        induction_prompt(bundle),
        f"seed_{seed}_initial",
    )
    rounds: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score: tuple[float, float, float] | None = None
    for round_index in range(max_rounds):
        validation_error = None
        try:
            rule_set = normalize_rule_response(response)
        except RuleValidationError as exc:
            validation_error = str(exc)
            rounds.append({
                "round": round_index,
                "valid": False,
                "validation_error": validation_error,
                "response": response,
            })
            response = caller(
                INDUCTION_SYSTEM,
                _invalid_repair_prompt(response, validation_error),
                f"seed_{seed}_static_repair_{round_index}",
            )
            continue

        metrics, predictions = evaluate_rules(rule_set, cases, "dev", f"induced_{seed}")
        feedback = development_feedback(cases, predictions)
        score = score_for_selection(metrics)
        current = {
            "round": round_index,
            "valid": True,
            "rule_set": rule_set,
            "metrics": metrics,
            "feedback": feedback,
        }
        rounds.append(current)
        if best_score is None or score > best_score:
            best_score = score
            best = current
        if (
            metrics["clean_fpr"] == 0
            and metrics["root_cause_accuracy"] == 1
            and metrics["joint_agent_step_accuracy"] == 1
        ):
            break
        response = caller(
            INDUCTION_SYSTEM,
            repair_prompt(rule_set, metrics, feedback, validation_error),
            f"seed_{seed}_development_repair_{round_index}",
        )
    if best is None:
        raise RuntimeError(f"seed {seed}: no statically valid rule set")
    return {
        "seed": seed,
        "training_bundle": {
            "training_split": bundle["training_split"],
            "n_examples": bundle["n_examples"],
            "seed": bundle["seed"],
        },
        "rounds": rounds,
        "selected_round": best["round"],
        "selected_rule_set": best["rule_set"],
        "development_metrics": best["metrics"],
    }


def induce(seeds: tuple[int, ...], max_rounds: int) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_benchmark(DEFAULT_OUT)
    caller = InductionCaller(OUT_DIR / "rule_induction_api.jsonl")
    runs = []
    for seed in seeds:
        path = OUT_DIR / f"induced_seed_{seed}.json"
        if path.exists():
            run = json.loads(path.read_text())
            validate_rule_set(run["selected_rule_set"])
        else:
            run = induce_one(caller, cases, seed, max_rounds)
            path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
        runs.append(run)
        print(json.dumps({
            "seed": seed,
            "selected_round": run["selected_round"],
            "development_metrics": {
                key: run["development_metrics"][key]
                for key in (
                    "clean_fpr", "fault_detection_recall", "root_cause_accuracy",
                    "root_cause_macro_f1", "joint_agent_step_accuracy",
                )
            },
        }, ensure_ascii=False), flush=True)
    selected = max(
        runs,
        key=lambda run: score_for_selection(run["development_metrics"]),
    )
    output = {
        "model": caller.model.model_name,
        "temperature": 0,
        "seeds": list(seeds),
        "selection_policy": (
            "development root macro-F1, then clean retention, then joint agent-step"),
        "selected_seed": selected["seed"],
        "selected_rule_set": selected["selected_rule_set"],
        "development_metrics": selected["development_metrics"],
        "api_usage": caller.usage(),
        "runs": [
            {
                "seed": run["seed"],
                "selected_round": run["selected_round"],
                "development_metrics": run["development_metrics"],
            }
            for run in runs
        ],
    }
    path = OUT_DIR / "induced_selected.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    return output


def freeze(induction: dict[str, Any], challenge: str | Path) -> dict[str, Any]:
    challenge_path = Path(challenge)
    rule_path = OUT_DIR / "induced_selected.json"
    freeze_record = {
        "freeze_date": "2026-07-26",
        "protocol": "faab-offline-rule-induction-1.0",
        "base_benchmark_sha256": _sha(DEFAULT_OUT),
        "challenge_sha256": _sha(challenge_path),
        "challenge_manifest_sha256": _sha(challenge_path.with_suffix(".manifest.json")),
        "rule_dsl_sha256": _sha(ROOT / "faab" / "rule_dsl.py"),
        "rule_induction_sha256": _sha(ROOT / "faab" / "rule_induction.py"),
        "runner_sha256": _sha(Path(__file__)),
        "selected_rules_sha256": _sha(rule_path),
        "model": induction["model"],
        "temperature": induction["temperature"],
        "seeds": induction["seeds"],
        "selected_seed": induction["selected_seed"],
        "policy": (
            "Rule induction reads FAAB v1.4 development only. Challenge labels are "
            "opened only after this record and selected rule set are written."),
    }
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(freeze_record, indent=2, ensure_ascii=False))
    return freeze_record


class AugmentedAuditor:
    """人工控制拥有 veto；仅在人工作出 pass 时允许归纳规则补充 finding。"""

    name = "human_plus_induced"

    def __init__(self, human_rules: dict[str, Any], induced_rules: dict[str, Any]):
        self.human = RuleSetAuditor(human_rules, "human_rules")
        self.induced = RuleSetAuditor(induced_rules, "induced_rules")

    def audit(self, public_case: dict[str, Any]) -> dict[str, Any]:
        primary = self.human.audit(public_case)
        if primary["fault_type"] != "none":
            return primary
        return self.induced.audit(public_case)


def _write_deterministic_method(
    name: str,
    auditor: Any,
    cases: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    predictions: dict[str, dict[str, Any]] = {}
    rows = []
    for case in cases:
        if case["split"] != split:
            continue
        started = time.perf_counter()
        prediction = normalize_result(auditor.audit(public_trace(case)))
        latency = time.perf_counter() - started
        predictions[case["case_id"]] = prediction
        rows.append({
            "case_id": case["case_id"],
            "source_id": case["source_id"],
            "method": name,
            "prediction": prediction,
            "latency_s": latency,
        })
    path = OUT_DIR / f"{name}_{split}.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    metrics = evaluate(cases, predictions, split)
    metrics.update({
        "method": name,
        "provider": "deterministic",
        "model": None,
        "api_calls": 0,
        "api_failures": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "mean_latency_s": sum(row["latency_s"] for row in rows) / len(rows),
    })
    (OUT_DIR / f"{name}_{split}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        row["case_id"]: row
        for row in (
            json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def _run_online_method(
    method: str,
    cases: list[dict[str, Any]],
    split: str,
    workers: int,
) -> dict[str, Any]:
    path = OUT_DIR / f"{method}_{split}.jsonl"
    existing = _load_existing(path)
    caller = CachedCaller(OUT_DIR / f"{method}_{split}_api.jsonl")
    auditor = {
        "llm_judge": LLMJudgeAuditor,
        "debate": DebateAuditor,
    }[method](caller)
    target = [
        case for case in cases
        if case["split"] == split and case["case_id"] not in existing
    ]
    lock = threading.Lock()

    def one(case: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        prediction = normalize_result(auditor.audit(public_trace(case)))
        return {
            "case_id": case["case_id"],
            "source_id": case["source_id"],
            "method": method,
            "prediction": prediction,
            "latency_s": time.perf_counter() - started,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one, case): case for case in target}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            existing[row["case_id"]] = row
            _append_jsonl(path, row, lock)
            if index % 25 == 0:
                print(f"[{method}] {index}/{len(target)}", flush=True)
    predictions = {
        case_id: row["prediction"] for case_id, row in existing.items()
    }
    metrics = evaluate(cases, predictions, split)
    metrics.update({
        "method": method,
        "provider": llm.provider(),
        "model": getattr(caller.model, "model_name", None),
        "api_calls": caller.calls,
        "api_requests_including_retries": caller.requests,
        "api_failures": caller.failures,
        "prompt_tokens": caller.prompt_tokens,
        "completion_tokens": caller.completion_tokens,
        "total_tokens": caller.total_tokens,
        "mean_latency_s": (
            sum(row["latency_s"] for row in existing.values()) / len(existing)),
    })
    (OUT_DIR / f"{method}_{split}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def evaluate_challenge(
    induction: dict[str, Any],
    challenge: str | Path,
    online_methods: tuple[str, ...],
    workers: int,
) -> dict[str, Any]:
    summary = validate_challenge(challenge)
    if not summary["valid"]:
        raise RuntimeError(f"challenge invalid: {summary['errors']}")
    cases = load_benchmark(challenge)
    induced_rules = induction["selected_rule_set"]
    methods: dict[str, Any] = {
        "local_rule_only": RuleAuditor(),
        "human_rules": RuleSetAuditor(human_rule_set(), "human_rules"),
        "llm_induced_rules": RuleSetAuditor(induced_rules, "llm_induced_rules"),
        "human_plus_induced": AugmentedAuditor(human_rule_set(), induced_rules),
    }
    metrics = {
        name: _write_deterministic_method(name, auditor, cases, "challenge")
        for name, auditor in methods.items()
    }
    for method in online_methods:
        metrics[method] = _run_online_method(method, cases, "challenge", workers)
    output = {
        "freeze": json.loads(FREEZE_PATH.read_text()),
        "challenge_validation": summary,
        "metrics": metrics,
    }
    (OUT_DIR / "challenge_summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["induce", "evaluate", "all"], default="all")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--challenge", default=str(DEFAULT_CHALLENGE))
    parser.add_argument(
        "--online-methods", nargs="*", choices=["llm_judge", "debate"],
        default=["llm_judge", "debate"])
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if args.phase in ("induce", "all"):
        induction = induce(seeds, args.max_rounds)
        freeze(induction, args.challenge)
    else:
        induction = json.loads((OUT_DIR / "induced_selected.json").read_text())
        if not FREEZE_PATH.exists():
            raise RuntimeError("freeze record missing; run --phase induce first")
    if args.phase in ("evaluate", "all"):
        output = evaluate_challenge(
            induction, args.challenge, tuple(args.online_methods), args.workers)
        concise = {
            method: {
                key: values[key]
                for key in (
                    "clean_fpr", "fault_detection_recall", "root_cause_accuracy",
                    "root_cause_macro_f1", "joint_agent_step_accuracy",
                    "api_calls", "total_tokens",
                )
            }
            for method, values in output["metrics"].items()
        }
        print(json.dumps(concise, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
