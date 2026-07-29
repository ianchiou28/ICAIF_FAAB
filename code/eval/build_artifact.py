"""打包匿名评审产物到 artifact/。

边界（与论文 §可复现性一致）：
- 代码、manifest、冻结哈希、逐例预测、指标与分析全部随附；
- **不随附**冻结的 benchmark/challenge 本体：其 expectation 工件含有许可的 I/B/E/S
  一致预期数值，不可再分发。随附构建脚本、manifest 与 SHA-256，持有许可的读者可
  逐位重建并核对哈希；
- **不随附**原始 API 日志：共 307 MB，且其中嵌入的 prompt 同样含许可数值。改为随附
  逐次调用的用量摘要（模型、用途、重试、token、状态），足以复核成本与调用次数声明。

用法：python3 eval/build_artifact.py [--out artifact]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CODE = [
    ("faab", "code/faab"),
    ("eval/faab_analysis.py", "code/eval/faab_analysis.py"),
    ("eval/faab_experiment.py", "code/eval/faab_experiment.py"),
    ("eval/faab_materiality.py", "code/eval/faab_materiality.py"),
    ("eval/faab_paper_figures.py", "code/eval/faab_paper_figures.py"),
    ("eval/faab_rule_induction.py", "code/eval/faab_rule_induction.py"),
    ("eval/faab_rule_induction_analysis.py", "code/eval/faab_rule_induction_analysis.py"),
    ("eval/faab_tables.py", "code/eval/faab_tables.py"),
    ("eval/build_artifact.py", "code/eval/build_artifact.py"),
    ("tests/test_faab.py", "code/tests/test_faab.py"),
]

FROZEN = [
    "data/faab/benchmark.manifest.json",
    "data/faab/experiment_freeze.json",
    "data/faab_rule_induction/challenge.manifest.json",
    "data/faab_rule_induction/experiment_freeze.json",
    "data/faab_rule_induction/results/induced_selected.json",
    "data/faab_rule_induction/results/induced_seed_101.json",
    "data/faab_rule_induction/results/induced_seed_202.json",
    "data/faab_rule_induction/results/induced_seed_303.json",
]

HASHED = ["data/faab/benchmark.jsonl", "data/faab_rule_induction/challenge.jsonl"]

# 打包时排除的目录：作废运行与被配额中断的部分运行。
EXCLUDE_DIRS = ("invalid_v1_3_label_leak", "prepilot", "archive_gpt56sol_quota")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded(path: Path) -> bool:
    return any(token in str(path) for token in EXCLUDE_DIRS)


def copy_code(out: Path) -> int:
    count = 0
    for src, dst in CODE:
        source, target = ROOT / src, out / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            count += len(list(target.rglob("*.py")))
        else:
            shutil.copy2(source, target)
            count += 1
    return count


_DECIMAL = re.compile(r"(?<![\w.])\d+\.\d{2,}")


def _redact_rationale(row: dict) -> dict:
    """把模型 rationale 里的多位小数替换为占位符。

    模型经常在自由文本里引用轨迹中的数值（约 31% 的预测），其中可能包含许可的
    I/B/E/S 共识值。裁决、故障类、责任方与步骤不受影响，因此所有论文指标仍可复算。
    """
    rationale = row.get("prediction", {}).get("rationale")
    if rationale:
        row["prediction"]["rationale"] = _DECIMAL.sub("<num>", rationale)
    return row


def copy_results(out: Path, redact: bool = False) -> tuple[int, int]:
    predictions = out / "results/predictions"
    metrics = out / "results/metrics"
    analysis = out / "results/analysis"
    for directory in (predictions, metrics, analysis):
        directory.mkdir(parents=True, exist_ok=True)
    n_pred = n_json = 0
    for base in ("data/faab/results", "data/faab_rule_induction/results"):
        for path in sorted((ROOT / base).glob("*")):
            if _excluded(path) or not path.is_file():
                continue
            if path.name.endswith("_api.jsonl") or path.suffix == ".quota_bak":
                continue
            if path.suffix == ".jsonl":
                if redact:
                    lines = [json.dumps(_redact_rationale(json.loads(line)),
                                        ensure_ascii=False)
                             for line in path.read_text().splitlines() if line.strip()]
                    (predictions / path.name).write_text("\n".join(lines) + "\n")
                else:
                    shutil.copy2(path, predictions / path.name)
                n_pred += 1
            elif path.suffix == ".json":
                target = analysis if path.name.startswith("analysis") or \
                    path.name in ("challenge_analysis.json", "challenge_summary.json",
                                  "materiality.json") else metrics
                shutil.copy2(path, target / path.name)
                n_json += 1
    return n_pred, n_json


def summarize_api_logs(out: Path) -> int:
    """逐次调用的用量摘要；不含 prompt 与响应文本，因此不携带许可数值。"""
    rows = []
    for base in ("data/faab/results", "data/faab_rule_induction/results"):
        for path in sorted((ROOT / base).rglob("*_api.jsonl")):
            if _excluded(path):
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                usage = record.get("usage") or {}
                rows.append({
                    "run": path.name.replace("_api.jsonl", ""),
                    "request_id": record.get("request_id"),
                    "status": record.get("status"),
                    "purpose": record.get("purpose", "verdict"),
                    "provider": record.get("provider"),
                    "model": record.get("model"),
                    "attempts": record.get("attempts"),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                })
    target = out / "results/api_usage.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_hashes(out: Path) -> dict[str, str]:
    hashes = {}
    for rel in HASHED:
        path = ROOT / rel
        if path.exists():
            hashes[rel] = _sha256(path)
    for rel in FROZEN:
        path = ROOT / rel
        if path.exists():
            hashes[rel] = _sha256(path)
    (out / "frozen/HASHES.json").write_text(json.dumps(hashes, indent=2))
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "artifact"))
    parser.add_argument("--redact-rationales", action="store_true",
                        help="把预测 rationale 里的多位小数替换为 <num>，"
                             "避免带出许可的共识数值；指标不受影响")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "frozen").mkdir(parents=True, exist_ok=True)

    n_code = copy_code(out)
    for rel in FROZEN:
        source = ROOT / rel
        if source.exists():
            shutil.copy2(source, out / "frozen" / source.name)
    hashes = write_hashes(out)
    n_pred, n_json = copy_results(out, args.redact_rationales)
    n_calls = summarize_api_logs(out)

    print(f"code files       {n_code}")
    print(f"frozen records   {len(list((out / 'frozen').glob('*')))}")
    print(f"prediction files {n_pred}")
    print(f"metric/analysis  {n_json}")
    print(f"api calls logged {n_calls}")
    print(f"rationales       {'redacted' if args.redact_rationales else 'verbatim'}")
    print(f"hashed artifacts {len(hashes)}")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"total size       {total / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
