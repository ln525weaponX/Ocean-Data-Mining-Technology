"""绘制 V2 自适应策略的逐次验收散点图，不读取或修改 Word 报告。

依赖：
    pip install numpy pandas matplotlib
运行（CSV 与脚本放在同一目录时）：
    python plot_acceptance.py
也可直接使用原始 800 次主运行 CSV：
    python plot_acceptance.py ../checkpoint_study_v2/results/test_runs.csv

说明：
1. risk_exceed_fraction 的单位是比例；画图时乘 100，变成百分数。
2. 纵轴用 symlog：零附近线性，其余部分对数。零值原样保留。
3. 每条运行记录绘制一个标记，坐标不抖动；相同坐标允许重叠。
4. n 标签统计零暴露率处的重合记录；全部 100 条数据另行导出。
5. 验收和余量均按原始数值计算，不按显示精度或图上位置计算。
"""

from pathlib import Path
import argparse
import hashlib
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import numpy as np
import pandas as pd


# 英文图中文字与原报告中的实验图一致；注释及字段说明采用中文。
SCENARIOS = {
    "normal": ("Normal input", "#17668A", "o", 95),
    "outage2h": ("2 h outage", "#D58B3D", "s", 76),
    "outage6h": ("6 h outage", "#8D709C", "D", 57),
    "io_slow": ("Slow storage", "#AF5257", "^", 90),
    "combined": ("Outage + slow storage", "#238F84", "v", 84),
}
WAIT_BUDGET_S = 60.0
EXPOSURE_BUDGET_PERCENT = 1.0


def load_runs(path):
    """兼容原始 800 行明细与仅包含 100 次自适应结果的明细。"""
    df = pd.read_csv(path)
    required = {
        "scenario", "seed", "policy", "publish_wait_p99",
        "risk_exceed_fraction", "exposure_ticks", "exposure_exceed_ticks",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"CSV 缺少字段：{missing}")
    df = df.loc[df["policy"].eq("adaptive")].copy()
    if "split" in df:
        df = df.loc[df["split"].eq("test")].copy()
    if set(df["scenario"]) != set(SCENARIOS):
        raise ValueError("场景必须恰为正常、两小时补传、六小时补传、降速、叠加五种。")
    if len(df) != 100 or not df.groupby("scenario").size().eq(20).all():
        raise ValueError("应有五场景各 20 次，共 100 次自适应留出运行。")
    if df.duplicated(["scenario", "seed"]).any():
        raise ValueError("存在重复的场景与种子组合。")
    for scenario, part in df.groupby("scenario"):
        if set(part["seed"]) != set(range(100, 120)):
            raise ValueError(f"{scenario} 的种子应为 100 至 119。")
    numeric = list(required.difference({"scenario", "policy"}))
    for key in numeric:
        df[key] = pd.to_numeric(df[key], errors="raise")
    if not np.isfinite(df[numeric].to_numpy()).all():
        raise ValueError("存在非有限数值。")
    if (
        (df["publish_wait_p99"] < 0).any()
        or (df["risk_exceed_fraction"] < 0).any()
        or (df["risk_exceed_fraction"] > 1).any()
        or (df["exposure_ticks"] <= 0).any()
        or (df["exposure_exceed_ticks"] < 0).any()
        or (df["exposure_exceed_ticks"] > df["exposure_ticks"]).any()
    ):
        raise ValueError("等待时间、暴露率或统计秒数不合法。")
    # 与统计域的分项计数交叉核对，避免误读比例或旧口径字段。
    recalculated = df["exposure_exceed_ticks"] / df["exposure_ticks"]
    if not np.allclose(df["risk_exceed_fraction"], recalculated, atol=1e-12, rtol=1e-10):
        raise ValueError("暴露率与超限秒数 / 统计秒数不一致。")
    df["exposure_percent"] = df["risk_exceed_fraction"] * 100.0
    df["wait_margin_s"] = WAIT_BUDGET_S - df["publish_wait_p99"]
    df["exposure_margin_pp"] = EXPOSURE_BUDGET_PERCENT - df["exposure_percent"]
    df["passes"] = df["wait_margin_s"].ge(0) & df["exposure_margin_pp"].ge(0)
    df["budget_utilization"] = np.maximum(
        df["publish_wait_p99"] / WAIT_BUDGET_S,
        df["exposure_percent"] / EXPOSURE_BUDGET_PERCENT,
    )
    return df.sort_values(["scenario", "seed"]).reset_index(drop=True)


def make_figure(df, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#98A7B3",
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(11, 7.6))
    fig.subplots_adjust(left=.11, right=.96, bottom=.23, top=.82)
    fig.text(.11, .95, "Run-by-run acceptance of the V2 adaptive policy",
             fontsize=17, weight="bold", color="#1E3346")
    fig.text(.11, .902, "Five held-out scenarios | seeds 100-119 | 100 runs | exact coordinates",
             fontsize=11, color="#52677B")

    p99_max = float(df["publish_wait_p99"].max())
    rho_max = float(df["exposure_percent"].max())
    n_pass = int(df["passes"].sum())
    fig.text(.11, .862,
             f"PASS {n_pass}/100     Max P99 {p99_max:g} s"
             f"     Max exposure {rho_max:.5f}%"
             f"     Zero exposure {int(df['exposure_percent'].eq(0).sum())}/100",
             fontsize=11, weight="bold", color="#17668A")

    # 主视图聚焦于数据所在的 50 秒以后；若替换数据后出现更小值，会自动扩展。
    xmin = min(50.0, np.floor(df["publish_wait_p99"].min() - 2))
    xmax = max(62.5, float(df["publish_wait_p99"].max()) + 2.0)
    ymax = max(1.8, rho_max * 1.4)
    ax.set_xlim(xmin, xmax)
    ax.set_yscale("symlog", linthresh=.005, linscale=.65, base=10)
    ax.set_ylim(-.0012, ymax)

    # 真正的合格集合是 P99 <= 60 且暴露率 <= 1%。
    # 图只画出这个集合落入当前坐标视窗的部分，不隐藏超界记录。
    ax.add_patch(Rectangle((xmin, 0), WAIT_BUDGET_S - xmin,
                           EXPOSURE_BUDGET_PERCENT, facecolor="#EAF4F0",
                           edgecolor="#6F9B88", linewidth=.8, alpha=.65, zorder=0))
    ax.axvspan(WAIT_BUDGET_S, xmax, color="#F8EDEC", zorder=0)
    ax.axhspan(EXPOSURE_BUDGET_PERCENT, ymax, color="#F8EDEC", zorder=0)
    ax.axvline(WAIT_BUDGET_S, color="#A94E4E", lw=1.3, ls="--", zorder=2)
    ax.axhline(EXPOSURE_BUDGET_PERCENT, color="#A94E4E", lw=1.3, ls="--", zorder=2)
    ax.text(WAIT_BUDGET_S + .12, .38, "P99 limit: 60 s", rotation=90,
            va="center", fontsize=10, color="#9A4545")
    ax.text(xmin + .2, 1.16, "Exposure limit: 1%", fontsize=10, color="#9A4545")
    ax.text(xmin + .2, .43, "ACCEPTED REGION", fontsize=10, color="#537E6B")

    plotted = 0
    for scenario, (label, color, marker, size) in SCENARIOS.items():
        part = df.loc[df["scenario"].eq(scenario)]
        ax.scatter(part["publish_wait_p99"], part["exposure_percent"],
                   s=size, marker=marker, facecolors="none", edgecolors=color,
                   linewidths=1.35, alpha=.82, label=f"{label} (n={len(part)})",
                   zorder=4)
        plotted += len(part)
    assert plotted == 100

    # 73 个零值会重合：标出每个真实横坐标的记录数，不把零换成正数。
    zero_counts = df.loc[df["exposure_percent"].eq(0)].groupby("publish_wait_p99").size()
    for x, count in zero_counts.items():
        ax.annotate(f"n={count}", xy=(x, 0), xytext=(0, 12),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color="#566576", zorder=7)

    worst_wait = df.loc[df["publish_wait_p99"].idxmax()]
    worst_risk = df.loc[df["exposure_percent"].idxmax()]
    # 示例数据各有一个最坏点；如有并列，closest_runs.csv 中保留全部并列记录。
    ax.annotate(
        "Largest P99: "
        f"{worst_wait['publish_wait_p99']:g} s\n"
        f"{worst_wait['scenario']}, seed {int(worst_wait['seed'])}\n"
        f"{worst_wait['wait_margin_s']:g} s below the deadline",
        xy=(worst_wait["publish_wait_p99"], worst_wait["exposure_percent"]),
        xytext=(57.5, .0037), fontsize=9.5, ha="left", va="top",
        color="#813F46", arrowprops={"arrowstyle": "->", "color": "#AF5257", "lw": 1},
        bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#E4D5D8", "alpha": .96},
        zorder=8,
    )
    ax.annotate(
        "Largest exposure: "
        f"{worst_risk['exposure_percent']:.5f}%\n"
        f"{worst_risk['scenario']}, seed {int(worst_risk['seed'])}",
        xy=(worst_risk["publish_wait_p99"], worst_risk["exposure_percent"]),
        xytext=(50.6, .15), fontsize=9.5, ha="left", va="center",
        color="#813F46", arrowprops={"arrowstyle": "->", "color": "#AF5257", "lw": 1},
        bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#E4D5D8", "alpha": .96},
        zorder=8,
    )
    ticks = [0, .005, .01, .03, .1, 1]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(["0", "0.005", "0.01", "0.03", "0.1", "1"]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(np.arange(np.ceil(xmin), np.floor(xmax) + 1))
    ax.grid(axis="both", color="#CDD9DF", lw=.6, alpha=.6, zorder=1)
    ax.set_xlabel("Run-level P99 publish wait (replay seconds)", labelpad=10)
    ax.set_ylabel("Run-level exposure rate (%)\nSymlog: linear near zero, logarithmic above 0.005%", labelpad=11)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(.53, .082), ncol=3, frameon=False, fontsize=10,
               handletextpad=.5, columnspacing=1.4)
    fig.text(.11, .05,
             "All markers use recorded coordinates. Coincident runs overlap; n labels count zero-exposure overlaps.",
             fontsize=9, color="#53677B")
    fig.text(.11, .022,
             "Acceptance is calculated from the unrounded CSV values. No zero substitution and no coordinate jitter.",
             fontsize=9, color="#53677B")

    for suffix in (".png", ".svg"):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=suffix[1:], dpi=320)
        (output / f"acceptance_scatter{suffix}").write_bytes(buffer.getvalue())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path,
                        default=Path(__file__).with_name("adaptive_runs_100.csv"))
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_runs(args.csv)
    make_figure(df, args.outdir)
    # 附带原值及验收余量，便于定位离边界最近的实际运行。
    cols = ["scenario", "seed", "publish_wait_p99", "risk_exceed_fraction",
            "exposure_percent", "exposure_exceed_ticks", "exposure_ticks",
            "wait_margin_s", "exposure_margin_pp", "budget_utilization", "passes"]
    df.sort_values(["budget_utilization", "exposure_percent"], ascending=False)[cols].to_csv(
        args.outdir / "closest_runs.csv", index=False, encoding="utf-8-sig")
    summary = {
        "input_file": args.csv.name,
        "input_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "runs": len(df),
        "passed": int(df["passes"].sum()),
        "p99_min_s": float(df["publish_wait_p99"].min()),
        "p99_max_s": float(df["publish_wait_p99"].max()),
        "exposure_max_percent": float(df["exposure_percent"].max()),
        "zero_exposure_runs": int(df["exposure_percent"].eq(0).sum()),
        "max_p99_runs": df.loc[df["publish_wait_p99"].eq(df["publish_wait_p99"].max()),
                               ["scenario", "seed"]].to_dict("records"),
        "max_exposure_runs": df.loc[df["exposure_percent"].eq(df["exposure_percent"].max()),
                                    ["scenario", "seed"]].to_dict("records"),
    }
    (args.outdir / "acceptance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
