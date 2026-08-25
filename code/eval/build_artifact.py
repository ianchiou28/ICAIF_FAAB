"""Package the anonymous review artifact into artifact/.

Boundaries (consistent with the Reproducibility section of the paper):
- code, manifests, frozen hashes, per-case predictions, metrics and analysis are all included;
- the frozen benchmark/challenge themselves are **not included**: their expectation artifacts
  carry licensed I/B/E/S consensus values and cannot be redistributed. The build scripts,
  manifests and SHA-256 digests are included, so a reader holding the license can rebuild them
  bit for bit and check the hashes;
- the raw API logs are **not included**: 307 MB in total, and the prompts embedded in them
  likewise carry licensed values. Included instead is a per-call usage summary (model, purpose,
  retries, tokens, status), which is enough to check the cost and call-count claims.

Usage: python3 eval/build_artifact.py [--out artifact]
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

# Directories excluded from the package: voided runs and partial runs cut short by quota.
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
    """Replace multi-digit decimals in model rationales with a placeholder.

    Models frequently quote trace values in free text (about 31% of predictions), and those may
    include licensed I/B/E/S consensus values. Verdict, fault class, responsible agent and step
    are unaffected, so every paper metric remains recomputable.
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
    """Per-call usage summary; it holds no prompt or response text and therefore carries no
    licensed values."""
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
                        help="replace multi-digit decimals in prediction rationales with <num> "
                             "so licensed consensus values are not carried out; metrics are "
                             "unaffected")
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
