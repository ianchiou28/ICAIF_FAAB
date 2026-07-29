"""Render publication figures from the frozen FAAB challenge analysis."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib import patheffects


# 正文实测用 CMR/CMBX（acmart 缺 libertine 时回退到 Computer Modern）。
# Latin Modern 是 CM 的 OpenType 后继，形态同源且有完整粗体/斜体；
# 不走 text.usetex，因为它还需要 dvipng，多一个易碎的外部依赖。
LM_DIR = Path.home() / "Library/TinyTeX/texmf-dist/fonts/opentype/public/lm"


def register_body_font() -> str:
    """Register Latin Modern if present; fall back to a bundled serif."""
    if LM_DIR.is_dir():
        for face in sorted(LM_DIR.glob("lmroman10-*.otf")):
            try:
                font_manager.fontManager.addfont(str(face))
            except Exception:  # pragma: no cover - font parsing is best effort
                pass
        names = {font.name for font in font_manager.fontManager.ttflist}
        for candidate in ("Latin Modern Roman", "LMRoman10"):
            if candidate in names:
                return candidate
    return "DejaVu Serif"


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data/faab_rule_induction/results/challenge_analysis.json"
OUTPUT = ROOT / "paper/figs/faab_diagnostics.pdf"

METHOD_LABELS = {
    "local_rule_only": "Local",
    "rules_retrieval": "Rules+ret.",
    "human_rules": "Human",
    "llm_induced_rules": "Induced",
    "human_plus_induced": "Human+ind.",
    "llm_judge": "Judge",
    "debate": "Debate",
    "tool_auditor": "Tool",
    "llm_judge@deepseek-reasoner": "Judge (R)",
    "llm_judge@gpt-5.6-sol": "Judge (G)",
    "tool_auditor@deepseek-reasoner": "Tool (R)",
    "tool_auditor@gpt-5.6-sol": "Tool (G)",
    "llm_judge@local-gpt-5.6-terra__p15": "Judge (T)",
    "tool_auditor@local-gpt-5.6-terra__p15": "Tool (T)",
    "llm_judge@p15": "Judge",
    "debate@p15": "Debate",
    "tool_auditor@p15": "Tool",
    "llm_judge@deepseek-reasoner__p15": "Judge (R)",
    "tool_auditor@deepseek-reasoner__p15": "Tool (R)",
}

FAULT_LABELS = {
    "fabricated_value": "Fabricated value",
    "quote_mismatch": "Quote mismatch",
    "unit_scaling": "Unit scaling",
    "basis_conflict": "Basis conflict",
    "period_mismatch": "Period mismatch",
    "entity_confusion": "Entity confusion",
    "stale_expectation": "Stale expectation",
    "future_leakage": "Future leakage",
    "definition_conflict": "Definition conflict",
    "missing_handoff": "Missing handoff",
    "role_violation": "Role violation",
    "silent_overwrite": "Silent overwrite",
    "wrong_security": "Wrong security",
    "wrong_window": "Wrong window",
}


METRICS_PATTERN = re.compile(
    r"^(?P<method>.+?)_challenge(?:__(?P<model>.+))?_metrics\.json$")


def load_analysis() -> dict:
    """冻结分析 + 后加入的臂（新审计器设计、新模型）逐文件并入。

    并入而不是重跑归纳分析，是为了不动已冻结的规则诱导与选择记录。
    """
    with ANALYSIS.open(encoding="utf-8") as stream:
        data = json.load(stream)
    for path in sorted(ANALYSIS.parent.glob("*_challenge*_metrics.json")):
        match = METRICS_PATTERN.match(path.name)
        if not match:
            continue
        method, model = match.group("method"), match.group("model")
        arm = f"{method}@{model}" if model else method
        if arm in data["point"]:
            continue
        row = json.loads(path.read_text())
        if row.get("n") != data.get("n_cases", row.get("n")):
            pass  # 不完整的臂仍然并入，由调用方决定是否使用
        data["point"][arm] = {
            "clean_fpr": row["clean_fpr"],
            "fault_detection_recall": row["fault_detection_recall"],
            "root_cause_accuracy": row["root_cause_accuracy"],
            "root_cause_macro_f1": row["root_cause_macro_f1"],
            "joint_agent_step_accuracy": row["joint_agent_step_accuracy"],
            "responsibility_localization_accuracy": row.get(
                "responsibility_localization_accuracy", row["joint_agent_step_accuracy"]),
            "materiality_weighted_recall": row["materiality_weighted_recall"],
            "api_calls": row.get("api_calls", 0),
            "total_tokens": row.get("total_tokens", 0),
        }
        data["per_fault"][arm] = row["per_fault"]
    return data


def draw_frontier(ax: plt.Axes, point: dict) -> None:
    """审计前沿：干净误报 vs 根因准确率。臂按可用性动态选取。

    强模型把根因准确率推得很高，却同时把干净误报推到 20% 量级，所以横轴不能再按
    v1.4 的 12% 上限截断——那样会把最重要的对比裁掉。
    """
    order = [
        ("human_rules", "Human", "#009E73", "s"),
        ("rules_retrieval", "Rules+ret.", "#0072B2", "o"),
        ("llm_induced_rules", "Induced", "#E69F00", "^"),
        ("llm_judge@p15", "Judge", "#CC79A7", "P"),
        ("llm_judge@deepseek-reasoner__p15", "Judge (R)", "#8E5AA8", "v"),
        ("llm_judge@local-gpt-5.6-terra__p15", "Judge (G)", "#6A3D9A", "<"),
        ("debate@p15", "Debate", "#D55E00", "X"),
        ("tool_auditor@p15", "Tool", "#117733", "D"),
        ("tool_auditor@local-gpt-5.6-terra__p15", "Tool (G)", "#44AA99", ">"),
    ]
    present = [item for item in order if item[0] in point]
    xs, ys = [], []
    for arm, label, color, marker in present:
        x = 100 * point[arm]["clean_fpr"]
        y = 100 * point[arm]["root_cause_accuracy"]
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=34, marker=marker, color=color, edgecolor="white",
                   linewidth=0.5, zorder=3, label=label)

    x_hi = max(12.0, max(xs) * 1.18) if xs else 12.0
    y_lo = min(37.0, (min(ys) - 6) if ys else 37.0)
    x_lo, y_hi = -0.6, 107.0
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Clean false-positive rate (%)")
    ax.set_ylabel("Root-cause accuracy (%)")
    ax.grid(True, color="#D8D8D8", linewidth=0.45)
    ax.set_axisbelow(True)

    # 目标工作区：FPR <= 2% 且 root accuracy >= 90%
    ax.add_patch(Rectangle((x_lo, 90.0), 2.0 - x_lo, y_hi - 90.0, facecolor="#009E73",
                           alpha=0.07, edgecolor="none", zorder=0))
    dashes = (0, (2.6, 1.9))
    ax.plot([x_lo, 2.0], [90.0, 90.0], color="#009E73", linewidth=0.55,
            linestyle=dashes, zorder=1)
    ax.plot([2.0, 2.0], [90.0, y_hi], color="#009E73", linewidth=0.55,
            linestyle=dashes, zorder=1)
    legend = ax.legend(loc="lower right", bbox_to_anchor=(0.995, 0.02), frameon=True,
                       framealpha=0.95, facecolor="white", edgecolor="#CFCFCF",
                       fancybox=False, fontsize=4.9, handlelength=0.8,
                       handletextpad=0.3, labelspacing=0.34, borderpad=0.32, ncol=1)
    legend.get_frame().set_linewidth(0.4)
    legend.set_zorder(5)
    ax.set_title("(a) Challenge audit frontier", loc="left", fontweight="bold")


def draw_fault_heatmap(ax: plt.Axes, per_fault: dict) -> None:
    # 人工契约作为参照列，归纳规则是主角，右侧两列是最强的在线对手。
    # 最强模型的判官与工具版并列：留出 handoff 那一行是 0/10 对 10/10。
    methods = [m for m in ("human_rules", "llm_induced_rules",
                           "llm_judge@local-gpt-5.6-terra__p15",
                           "tool_auditor@local-gpt-5.6-terra__p15",
                           "llm_judge@p15", "tool_auditor@p15")
               if m in per_fault][:4]
    faults = list(FAULT_LABELS)
    values = np.array(
        [
            [100 * per_fault[method][fault]["root_cause_accuracy"] for method in methods]
            for fault in faults
        ]
    )
    cmap = LinearSegmentedColormap.from_list(
        "faab_accuracy", ["#B2182B", "#F4A582", "#F7F7F7", "#92C5DE", "#2166AC"]
    )
    image = ax.imshow(values, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(methods)), [METHOD_LABELS[item] for item in methods])
    ax.set_yticks(range(len(faults)), [FAULT_LABELS[item] for item in faults])
    ax.tick_params(axis="x", labelrotation=28, pad=1)
    ax.tick_params(axis="y", labelsize=5.4, pad=1)
    ax.set_xticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(faults), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if value < 100:
                ax.text(
                    column,
                    row,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5.0,
                    color="white" if value <= 30 or value >= 80 else "#222222",
                )
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.set_title("Root accuracy by fault (%, $n{=}10$/cell)", loc="left",
                 fontweight="bold")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.055, pad=0.025)
    colorbar.set_ticks([0, 50, 100])
    colorbar.ax.tick_params(labelsize=5.3, length=2)
    colorbar.outline.set_linewidth(0.45)


def draw_stability(ax: plt.Axes, seed_stability: dict) -> None:
    runs = seed_stability["runs"]
    seeds = [str(run["seed"]) for run in runs]
    dev = [100 * run["development"]["root_cause_accuracy"] for run in runs]
    challenge = [100 * run["challenge"]["root_cause_accuracy"] for run in runs]
    selected = seeds[0]
    pairwise = seed_stability["pairwise_fault_prediction_agreement"]
    agreement = [
        100.0,
        100 * pairwise[f"{selected}_vs_{seeds[1]}"],
        100 * pairwise[f"{selected}_vs_{seeds[2]}"],
    ]
    x = np.arange(len(seeds))
    # 用点图而非柱状图：柱长编码数值大小，非零起点会夸大差异；
    # 点图不靠长度编码，因此可以合法地放大到数据所在区间。
    series = [
        ("Dev", dev, "#0072B2", "o", -0.22),
        ("Challenge", challenge, "#E69F00", "s", 0.0),
        ("Agreement", agreement, "#8B8B8B", "^", 0.22),
    ]
    for label, values, color, marker, dodge in series:
        positions = x + dodge
        ax.scatter(positions, values, s=26, marker=marker, color=color,
                   edgecolor="white", linewidth=0.4, zorder=3, label=label)
        for xpos, value in zip(positions, values):
            ax.text(xpos, value + 1.0, f"{value:.1f}", ha="center", va="bottom",
                    fontsize=5.2, color="#222222", zorder=4,
                    path_effects=[patheffects.withStroke(linewidth=1.7,
                                                         foreground="white")])

    ax.set_xticks(x, [f"{seed}{'*' if index == 0 else ''}" for index, seed in enumerate(seeds)])
    ax.set_xlim(-0.5, len(seeds) - 0.5)
    ax.set_xlabel("Induction seed (* selected on dev)")
    ax.set_ylabel("Root-cause accuracy / agreement (%)")
    ax.set_ylim(68, 106)
    ax.set_yticks([70, 75, 80, 85, 90, 95, 100])
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.45)
    ax.set_axisbelow(True)
    # 图例落在 68--76 的空白带，且用 mode="expand" 强制与 x 轴等宽
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 0.005, 1.0, 0.06),
        mode="expand",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#CFCFCF",
        fancybox=False,
        fontsize=5.0,
        ncol=3,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=0.5,
        borderpad=0.3,
    )
    ax.get_legend().get_frame().set_linewidth(0.4)
    ax.set_title("(c) Example-order stability", loc="left", fontweight="bold")


def main() -> None:
    data = load_analysis()
    body_font = register_body_font()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [body_font, "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": 6.3,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.1,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.linewidth": 0.55,
            # 数据最大化、墨水最小化：去掉顶/右边框，刻度线内收
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "legend.fontsize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(3.34, 2.62), constrained_layout=False)
    # (a) 与 (c) 取正方形：原来 wspace=0.72 让每条间隙宽达 1.06in，
    # 远超 (b) 纵轴标签所需，把富余的宽度还给两侧面板。
    grid = figure.add_gridspec(1, 1, left=0.30, right=0.90, bottom=0.20, top=0.92)
    draw_fault_heatmap(figure.add_subplot(grid[0, 0]), data["per_fault"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
