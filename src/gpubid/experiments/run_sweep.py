"""Run the headline experiments (deterministic — no API keys required) and save figures.

Usage:
    python -m gpubid.experiments.run_sweep
    python -m gpubid.experiments.run_sweep --seeds 30  # more seeds for tighter CIs

Produces these figures into `data/figures/`:
  1. welfare_vs_rounds.png      — recovery vs N rounds
  2. welfare_vs_cap.png         — recovery and Gini as the concentration cap tightens
  3. agentic_vs_posted.png      — boxplot of recovery across seeds in both regimes
  4. offpeak_utilization.png    — off-peak slot utilization, agentic vs posted

The two LLM-only experiments (homogeneous vs heterogeneous, free-form vs structured)
require API keys and are run separately by `bake_presets.py` and a follow-on script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from gpubid.benchmark.posted_price import solve_posted_price
from gpubid.benchmark.vcg import solve_vcg
from gpubid.eval.metrics import compute_metrics
from gpubid.market import generate_market
from gpubid.viz.trading_floor import collect_snapshots


def _agentic_metrics(market, *, max_rounds: int, cap: Optional[float]):
    snaps = collect_snapshots(market, max_rounds=max_rounds, concentration_cap_pct=cap)
    return compute_metrics(market, list(snaps[-1].all_deals))


# ---------------------------------------------------------------------------
# Experiment 1 — welfare recovery vs round count
# ---------------------------------------------------------------------------


def experiment_welfare_vs_rounds(seeds: list[int], rounds_grid: list[int]) -> dict:
    rows = {n: [] for n in rounds_grid}
    for seed in seeds:
        market = generate_market(8, 4, "tight", seed=seed)
        vcg = solve_vcg(market)
        if vcg.welfare <= 0:
            continue
        for n in rounds_grid:
            am = _agentic_metrics(market, max_rounds=n, cap=None)
            rows[n].append(am.welfare / vcg.welfare * 100)
    return rows


# ---------------------------------------------------------------------------
# Experiment 2 — welfare and Gini as cap tightens
# ---------------------------------------------------------------------------


def experiment_welfare_vs_cap(seeds: list[int], caps: list[Optional[float]]) -> dict:
    welfare_rows: dict[Optional[float], list[float]] = {c: [] for c in caps}
    gini_rows: dict[Optional[float], list[float]] = {c: [] for c in caps}
    for seed in seeds:
        market = generate_market(8, 4, "slack", seed=seed)
        vcg = solve_vcg(market)
        if vcg.welfare <= 0:
            continue
        for c in caps:
            am = _agentic_metrics(market, max_rounds=5, cap=c)
            welfare_rows[c].append(am.welfare / vcg.welfare * 100)
            gini_rows[c].append(am.gini_buyer_welfare)
    return {"welfare": welfare_rows, "gini": gini_rows}


# ---------------------------------------------------------------------------
# Experiment 3 — agentic vs posted-price across seeds
# ---------------------------------------------------------------------------


def experiment_agentic_vs_posted(seeds: list[int]) -> dict:
    out = {"tight": {"agentic": [], "posted": []}, "slack": {"agentic": [], "posted": []}}
    for regime in ("tight", "slack"):
        for seed in seeds:
            market = generate_market(8, 4, regime, seed=seed)
            vcg = solve_vcg(market)
            if vcg.welfare <= 0:
                continue
            am = _agentic_metrics(market, max_rounds=5, cap=None)
            posted = solve_posted_price(market)
            pm = compute_metrics(market, posted.deals)
            out[regime]["agentic"].append(am.welfare / vcg.welfare * 100)
            out[regime]["posted"].append(pm.welfare / vcg.welfare * 100)
    return out


# ---------------------------------------------------------------------------
# Experiment 4 — off-peak utilization
# ---------------------------------------------------------------------------


def experiment_offpeak_utilization(seeds: list[int]) -> dict:
    out = {"agentic": [], "posted": []}
    for seed in seeds:
        market = generate_market(8, 4, "slack", seed=seed)
        # Skip markets without off-peak slots.
        if not any(sl.is_offpeak for s in market.sellers for sl in s.capacity_slots):
            continue
        am = _agentic_metrics(market, max_rounds=5, cap=None)
        posted = solve_posted_price(market)
        pm = compute_metrics(market, posted.deals)
        out["agentic"].append(am.offpeak_utilization * 100)
        out["posted"].append(pm.offpeak_utilization * 100)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_welfare_vs_rounds(rows: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rounds = sorted(rows.keys())
    means = [np.mean(rows[r]) if rows[r] else 0 for r in rounds]
    stds = [np.std(rows[r]) if rows[r] else 0 for r in rounds]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(rounds, means, yerr=stds, marker="o", capsize=4, color="#2563eb", linewidth=2)
    ax.axhline(85, ls="--", color="#94a3b8", lw=1, label="85% target")
    ax.set_xlabel("Max rounds")
    ax.set_ylabel("% of VCG welfare recovered")
    ax.set_ylim(0, 105)
    ax.set_title("Recovery vs round count (tight 8×4)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_welfare_vs_cap(data: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    caps = list(data["welfare"].keys())
    labels = [f"{c*100:.0f}%" if c else "off" for c in caps]
    welfare_means = [np.mean(data["welfare"][c]) if data["welfare"][c] else 0 for c in caps]
    gini_means = [np.mean(data["gini"][c]) if data["gini"][c] else 0 for c in caps]

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(labels, welfare_means, marker="o", color="#2563eb", linewidth=2, label="welfare recovery (%)")
    ax1.set_ylabel("% of VCG welfare recovered", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.set_ylim(0, 105)
    ax1.set_xlabel("Concentration cap")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(labels, gini_means, marker="s", color="#dc2626", linewidth=2, label="Gini (buyer welfare)")
    ax2.set_ylabel("Gini coefficient", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax2.set_ylim(0, 1)

    fig.suptitle("Fairness vs welfare as concentration cap tightens (slack 8×4)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_agentic_vs_posted(data: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
    for ax, regime in zip(axes, ("tight", "slack")):
        ax.boxplot(
            [data[regime]["agentic"], data[regime]["posted"]],
            tick_labels=["Agentic", "Posted-price"],
            patch_artist=True,
            boxprops=dict(facecolor="#dbeafe"),
            medianprops=dict(color="#1d4ed8"),
        )
        ax.set_title(f"{regime} supply")
        ax.set_ylabel("% of VCG welfare recovered" if regime == "tight" else "")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Agentic vs posted-price across seeds")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_offpeak(data: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    means = [np.mean(data["agentic"]) if data["agentic"] else 0,
             np.mean(data["posted"]) if data["posted"] else 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Agentic", "Posted-price"], means, color=["#2563eb", "#94a3b8"])
    ax.set_ylabel("Off-peak slot utilization (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Off-peak filling — slack supply, mean across seeds")
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_all(n_seeds: int = 20, output_dir: Path = Path("data/figures")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(n_seeds))

    print(f"Running 4 deterministic experiments across {n_seeds} seeds…", flush=True)

    print("  1/4 welfare vs rounds…", flush=True)
    rounds_data = experiment_welfare_vs_rounds(seeds, [1, 2, 3, 5, 7, 10])
    _plot_welfare_vs_rounds(rounds_data, output_dir / "welfare_vs_rounds.png")

    print("  2/4 welfare and Gini vs cap…", flush=True)
    cap_data = experiment_welfare_vs_cap(seeds, [None, 0.4, 0.3, 0.2, 0.1])
    _plot_welfare_vs_cap(cap_data, output_dir / "welfare_vs_cap.png")

    print("  3/4 agentic vs posted-price…", flush=True)
    avp_data = experiment_agentic_vs_posted(seeds)
    _plot_agentic_vs_posted(avp_data, output_dir / "agentic_vs_posted.png")

    print("  4/4 off-peak utilization…", flush=True)
    op_data = experiment_offpeak_utilization(seeds)
    _plot_offpeak(op_data, output_dir / "offpeak_utilization.png")

    print(f"Done. Figures saved to {output_dir}/")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20, help="Number of random seeds per experiment.")
    parser.add_argument("--output", type=str, default="data/figures", help="Output directory.")
    args = parser.parse_args()
    run_all(n_seeds=args.seeds, output_dir=Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run_all",
    "experiment_welfare_vs_rounds",
    "experiment_welfare_vs_cap",
    "experiment_agentic_vs_posted",
    "experiment_offpeak_utilization",
]
