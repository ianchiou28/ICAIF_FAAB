# FAAB — Financial Agent Audit Benchmark

Anonymous artifact for the ICAIF '26 submission *FAAB: A Benchmark for Fault Detection
and Responsibility Attribution in Financial Multi-Agent Workflows*.

Everything needed to check the paper's numbers is here. What is missing is missing for a
stated reason, given in **Data boundary** below.

## Layout

```
code/faab/            benchmark construction, injectors, auditors, control DSL, metrics
  benchmark.py        clean-trace materialization + 14 single-fault injectors
  challenge.py        unused-source challenge, incl. the held-out handoff realization
  auditors.py         Rules, Rules+retrieval, Rules+ret.+LLM, judge, debate
  tool_auditor.py     tool-augmented agentic auditor: checklist + 6 deterministic tools
  rule_dsl.py         control DSL: validation, execution, human contract rule set
  rule_induction.py   offline trace-to-control compilation
  models.py           provider abstraction (deepseek / openai / OpenAI-compatible)
  metrics.py          clean FPR, detection, root cause, responsibility localization
  validate.py         independent benchmark validator
code/eval/            runners and analysis
  faab_experiment.py  runs any auditor x model x split, resumable, logs every call
  faab_analysis.py    source-cluster bootstrap, paired differences, confusion
  faab_materiality.py measured decision impact per fault class
  faab_tables.py      emits the paper's table rows from the analysis JSON
  faab_paper_figures.py
  faab_rule_induction.py / faab_rule_induction_analysis.py
  build_artifact.py   the script that produced this folder
code/tests/           19 FAAB regression tests (42 in the full repository)
frozen/               manifests, freeze records, induced rule sets, HASHES.json
results/predictions/  one record per case, per (auditor, model, split)
results/metrics/      per-arm metric JSON
results/analysis/     bootstrap intervals, paired differences, confusion, materiality
results/api_usage.csv 35,749 model calls: model, purpose, retries, tokens, status
```

## Reproducing the paper

**Every number in the results tables, with no API access and no licensed data.** The
per-case predictions in `results/predictions/` are the raw outputs; the analysis scripts
recompute the metrics, bootstrap intervals, and table rows from them:

```bash
python3 code/eval/faab_analysis.py --split test
python3 code/eval/faab_analysis.py --split challenge
python3 code/eval/faab_tables.py --split test --prompts v1.5
```

**The regression tests**, which cover label isolation, the injectors, the metric
definitions, DSL restrictions, the held-out handoff, the role--step bijection, the tool
primitives, tool isolation from the oracle, and the veto ordering that stops a residual
model from overturning a deterministic finding:

```bash
python3 -m pytest code/tests/test_faab.py -q
```

**Re-running an auditor** needs an API key for the relevant vendor and the frozen
benchmark (see below):

```bash
python3 code/eval/faab_experiment.py --methods tool_auditor --model gpt-5.6-terra --split test
```

## Data boundary

The frozen `benchmark.jsonl` (1,000 traces) and `challenge.jsonl` (200 traces) are
**not** included. Their expectation artifacts carry licensed I/B/E/S consensus values
(`mean_estimate`, `dispersion`, `vintage`), which we may not redistribute. Included in
their place are the manifests, the construction code, and SHA-256 digests in
`frozen/HASHES.json`, so a reader with a licensed consensus source can rebuild both files
and verify they match bit for bit:

```bash
python3 -m code.faab.benchmark --db <your-source.db>
python3 -m code.faab.challenge  --db <your-source.db>
shasum -a 256 benchmark.jsonl challenge.jsonl   # compare against frozen/HASHES.json
```

Frozen digests, also quoted in the paper:

- benchmark `b54fad29993e4081…ec5864a6`
- challenge `c8cd0744f15b9ba9…d459566c`

Raw API logs are also excluded. They total 307 MB, and the prompts embed the same
licensed values. `results/api_usage.csv` carries one row per call with model, purpose,
retry count, token counts, and status, which is what the cost and call-count claims in
the paper rest on. Model outputs themselves are preserved in full in
`results/predictions/`.

## Notes on the runs

- Result files are named `<method>_<split>[__<model>][__p15].jsonl`. The `__p15` suffix
  marks runs under the corrected prompt that declares the standardized-surprise
  derivation; files without it are the earlier runs the paper reports alongside them
  (paper §"What the benchmark discloses is part of the result"). Deterministic arms carry
  no suffix because their output does not depend on any prompt.
- `deepseek-chat` arms run at temperature 0. The reasoning models reject an explicit
  temperature and use provider defaults, so those arms are not bitwise reproducible; this
  is stated in the paper's limitations.
- Three excluded run directories are not part of any reported result: a quarantined run
  with a label leak in its prompts, early pre-pilots, and a partial second-vendor run cut
  short by an API quota. The paper describes the first and third; the artifact omits them
  to avoid confusion.

Source code comments are in Chinese; identifiers, prompts, and all released data are in
English.
