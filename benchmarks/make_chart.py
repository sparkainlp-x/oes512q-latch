"""Per-series F1 / AP chart for the NAB benchmark (block 32, warm-up-calibrated thresholds)."""

from __future__ import annotations

from pathlib import Path

LABELS = {
    "oes_latch": ("OES latch (self-calibrated τ)", "o", "#d62728"),
    "rolling_zscore": ("Rolling z-score", "s", "#7f7f7f"),
    "cusum": ("CUSUM", "^", "#1f77b4"),
    "isolation_forest": ("IsolationForest (seed 42)", "D", "#2ca02c"),
}


def make_chart(report: dict, path: Path, block: str = "32", mode: str = "calibrated") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = report["series"]
    keys = [
        k
        for k in sorted(series)
        if block in series[k]["blocks"]
        and series[k]["blocks"][block]["counts"]["positive_blocks_test"] > 0
    ]
    n = len(keys)
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.28 * n + 2.2), sharey=True)
    ypos = list(range(n))[::-1]
    for ax, metric, title in ((axes[0], "f1", "F1"), (axes[1], "ap", "Average precision")):
        for det, (label, marker, color) in LABELS.items():
            vals = [series[k]["blocks"][block]["modes"][mode][det][metric] for k in keys]
            ax.scatter(vals, ypos, marker=marker, color=color, label=label, s=28, alpha=0.85)
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(f"{title} per series (block {block}, warm-up q0.99 thresholds)", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        for i in range(1, n):
            if keys[n - i].split("/")[0] != keys[n - i - 1].split("/")[0]:
                ax.axhline(ypos[n - i] + 0.5, color="black", lw=0.5, alpha=0.4)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([k.replace(".csv", "") for k in keys], fontsize=7)
    axes[1].legend(loc="lower right", fontsize=8)
    commit = report.get("code", {}).get("commit", "")[:7]
    fig.suptitle(
        f"NAB real-data series @ {report['dataset']['nab_commit'][:7]}, code {commit} — REPORTED "
        "(single run; IsolationForest seed 42)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
