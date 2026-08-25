"""Generate paper table rows from the analysis JSON so the numbers in the text match the
data exactly (never transcribed by hand).

  python3 eval/faab_analysis.py --split test
  python3 eval/faab_analysis.py --split challenge
  python3 eval/faab_tables.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = {
    "test": ROOT / "data/faab/results/analysis_test.json",
    "challenge": ROOT / "data/faab_rule_induction/results/analysis_challenge.json",
}
RESULTS = {
    "test": ROOT / "data/faab/results",
    "challenge": ROOT / "data/faab_rule_induction/results",
}

# Arm name -> (display name, model column, prompt version). This order is the table row order.
# The "__p15" suffix marks reruns after the declaration-ledger derivation was added; LLM arms
# with no suffix are v1.4.
DISPLAY: list[tuple[str, str, str, str]] = [
    ("rule_only", "Rules", "--", "--"),
    ("local_rule_only", "Rules", "--", "--"),
    ("rules_retrieval", "Rules+retrieval", "--", "--"),
    ("human_rules", "Human contracts", "--", "--"),
    ("llm_induced_rules", "LLM-induced rules", "--", "--"),
    ("human_plus_induced", "Human+induced", "--", "--"),
    ("hybrid", "Rules+retrieval+LLM", "deepseek-chat", "v1.4"),
    ("hybrid@p15", "Rules+retrieval+LLM", "deepseek-chat", "v1.5"),
    ("llm_judge", "LLM judge", "deepseek-chat", "v1.4"),
    ("llm_judge@p15", "LLM judge", "deepseek-chat", "v1.5"),
    ("llm_judge@deepseek-reasoner", "LLM judge", "deepseek-reasoner", "v1.4"),
    ("llm_judge@deepseek-reasoner__p15", "LLM judge", "deepseek-reasoner", "v1.5"),
    ("llm_judge@gpt-5.6-sol", "LLM judge", "gpt-5.6-sol", "v1.4"),
    ("debate", "Debate", "deepseek-chat", "v1.4"),
    ("debate@p15", "Debate", "deepseek-chat", "v1.5"),
    ("tool_auditor", "Tool-augmented", "deepseek-chat", "v1.4"),
    ("tool_auditor@p15", "Tool-augmented", "deepseek-chat", "v1.5"),
    ("tool_auditor@deepseek-reasoner__p15", "Tool-augmented", "deepseek-reasoner", "v1.5"),
    ("tool_auditor@gpt-5.6-sol", "Tool-augmented", "gpt-5.6-sol", "v1.4"),
    ("llm_judge@local-gpt-5.6-terra__p15", "LLM judge", "gpt-5.6-terra", "v1.5"),
    ("tool_auditor@local-gpt-5.6-terra__p15", "Tool-augmented", "gpt-5.6-terra", "v1.5"),
]
COLUMNS = ("clean_fpr", "fault_detection_recall", "root_cause_accuracy",
           "responsibility_localization_accuracy")


def _fmt(value: float) -> str:
    return f"{100 * value:.1f}"


def _thousands(n: int | None) -> str:
    return "0" if not n else f"{n:,}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=sorted(ANALYSIS), default="test")
    parser.add_argument("--prompts", choices=["v1.4", "v1.5", "all"], default="v1.5",
                        help="filter main-table rows by prompt version")
    args = parser.parse_args()

    analysis = json.loads(ANALYSIS[args.split].read_text())
    point, ci95, runtime = analysis["point"], analysis["ci95"], analysis["runtime"]

    print(f"% --- {args.split}: main table rows "
          f"(bootstrap {analysis['bootstrap']['draws']} draws over "
          f"{analysis['bootstrap']['clusters']} source clusters) ---")
    for arm, label, model, prompt in DISPLAY:
        if arm not in point:
            continue
        if args.prompts != "all" and prompt not in ("--", args.prompts):
            continue
        row = point[arm]
        cells = " & ".join(_fmt(row[c]) for c in COLUMNS)
        calls = runtime.get(arm, {}).get("api_calls") or 0
        tokens = runtime.get(arm, {}).get("total_tokens") or 0
        print(f"{label} & {model} & {cells} & {_thousands(calls)} & "
              f"{_thousands(tokens)} \\\\")

    print(f"\n% --- {args.split}: 95% bootstrap intervals ---")
    for arm, label, model, _prompt in DISPLAY:
        if arm not in ci95:
            continue
        parts = []
        for column in ("fault_detection_recall", "root_cause_accuracy"):
            lo, hi = ci95[arm][column]
            parts.append(f"{column.split('_')[0]} [{_fmt(lo)}, {_fmt(hi)}]")
        print(f"%   {label} ({model}): " + "; ".join(parts))

    reference = analysis["reference"]
    print(f"\n% --- {args.split}: paired differences, {reference} minus arm ---")
    for arm, values in analysis["reference_minus_arm"].items():
        detect = values["fault_detection_recall"]
        root = values["root_cause_accuracy"]
        print(f"%   {arm}: detect {_fmt(detect['point'])} "
              f"[{_fmt(detect['ci95'][0])}, {_fmt(detect['ci95'][1])}]; "
              f"root {_fmt(root['point'])} "
              f"[{_fmt(root['ci95'][0])}, {_fmt(root['ci95'][1])}]")

    print(f"\n% --- {args.split}: prediction agreement with {reference} ---")
    for arm, values in analysis["prediction_agreement_with_reference"].items():
        print(f"%   {arm}: {values['identical_predictions']}/{values['n']} identical "
              f"({100 * values['rate']:.1f}%)")

    print(f"\n% --- {args.split}: root-cause error patterns (top 4 per arm) ---")
    for arm, errors in analysis["root_cause_errors"].items():
        top = sorted(errors.items(), key=lambda kv: -kv[1])[:4]
        if top:
            print(f"%   {arm}: " + ", ".join(f"{k}={v}" for k, v in top))

    tool_rows = {}
    for arm, _label, _model, _prompt in DISPLAY:
        if not arm.startswith("tool_auditor"):
            continue
        suffix = "" if "@" not in arm else "__" + arm.split("@", 1)[1]
        path = RESULTS[args.split] / f"tool_auditor_{args.split}{suffix}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        made = [r.get("tool_calls_made", 0) for r in rows]
        rounds = [r.get("tool_rounds", 0) for r in rows]
        tool_rows[arm] = (sum(made) / len(made), sum(rounds) / len(rounds), len(rows))
    if tool_rows:
        print(f"\n% --- {args.split}: tool usage (mean calls, mean rounds, n) ---")
        for arm, (calls, rounds, n) in tool_rows.items():
            print(f"%   {arm}: {calls:.1f} tool calls, {rounds:.1f} rounds, n={n}")


if __name__ == "__main__":
    main()
