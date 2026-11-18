"""稳健性检验：参数敏感性 + 交易成本敏感性。

在统一的样本区间（与 12-1 主结果相同）上比较：
  - 形成期 lookback ∈ {3, 6, 9, 12} 个月
  - 是否跳过最近一个月 skip ∈ {0, 1}
  - 分组数 buckets ∈ {3, 5, 10}
  - 单边交易成本 ∈ {0, 5, 10, 20, 50} bp（12-1、10 组）

用法:  python robustness.py      （需先运行过 run.py 以缓存数据，或会自动下载）
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from momentum.analysis import perf_stats
from momentum.data import get_french_factors, get_prices, get_sp500_tickers
from momentum.factor import build_wml

OUT = Path(__file__).resolve().parent / "results"
START = "2000-01-01"


def summarize(res: dict, umd: pd.Series, window: pd.DatetimeIndex) -> dict:
    wml = res["wml"].reindex(window)
    s = perf_stats(wml)
    ex09 = wml[wml.index.year != 2009]
    return {
        "ann_return": s["ann_return"],
        "sharpe": s["sharpe"],
        "t_stat": s["t_stat"],
        "max_drawdown": s["max_drawdown"],
        "ann_return_ex2009": ex09.mean() * 12,
        "sharpe_ex2009": ex09.mean() / ex09.std() * np.sqrt(12),
        "corr_umd": wml.corr(umd.reindex(window)),
        "turnover": res["turnover"].reindex(window).mean(),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    daily = get_prices(get_sp500_tickers(), START)
    ff = get_french_factors()
    umd = ff["UMD"]

    # 统一样本区间：以基准 12-1 为准，剔除不完整的最后一个月
    base = build_wml(daily)["wml"]
    last_full = daily.index[-1] + pd.offsets.MonthEnd(0)
    if daily.index[-1] < last_full:
        base = base[base.index < last_full]
    window = base.index.intersection(ff.index)
    print(f"Window: {window[0]:%Y-%m} -> {window[-1]:%Y-%m} ({len(window)} months)")

    # ---------- 1. 参数网格 ----------
    rows = []
    for lb, sk, nb in itertools.product([3, 6, 9, 12], [0, 1], [3, 5, 10]):
        res = build_wml(daily, lookback=lb, skip=sk, n_buckets=nb)
        rows.append({"lookback": lb, "skip": sk, "buckets": nb, **summarize(res, umd, window)})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "robustness_grid.csv", index=False, float_format="%.4f")

    # ---------- 2. 交易成本 ----------
    cost_rows = []
    for c in [0, 5, 10, 20, 50]:
        r = build_wml(daily, cost_bps=c)["wml_net"].reindex(window)
        st = perf_stats(r)
        cost_rows.append({"cost_bps": c, "ann_return": st["ann_return"], "sharpe": st["sharpe"]})
    costs = pd.DataFrame(cost_rows)
    # 盈亏平衡成本：毛收益 / (年化换手 × 2)
    res12 = build_wml(daily)
    gross = res12["wml"].reindex(window).mean() * 12
    ann_to = res12["turnover"].reindex(window).mean() * 12
    breakeven = gross / (ann_to * 2) * 1e4
    costs.to_csv(OUT / "robustness_costs.csv", index=False, float_format="%.4f")

    # ---------- 3. 图：Sharpe 热力图（10 组） ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, title in [(axes[0], "sharpe", "Sharpe (full sample)"),
                           (axes[1], "sharpe_ex2009", "Sharpe (excluding 2009)")]:
        piv = grid[grid.buckets == 10].pivot(index="skip", columns="lookback", values=col)
        im = ax.imshow(piv.values, cmap="RdBu", vmin=-0.6, vmax=0.6, aspect="auto")
        ax.set_xticks(range(len(piv.columns)), [f"{c}m" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)), [f"skip={i}" for i in piv.index])
        for (i, j), v in np.ndenumerate(piv.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10)
        ax.set_xlabel("formation period (lookback)")
        ax.set_title(title + ", 10 buckets")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(OUT / "robustness_heatmap.png", dpi=140, bbox_inches="tight")

    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    pd.set_option("display.width", 200)
    print("\n=== Parameter grid ===\n", grid.to_string(index=False))
    print("\n=== Cost sensitivity (12-1, 10 buckets) ===\n", costs.to_string(index=False))
    print(f"\nBreak-even one-way cost: {breakeven:.1f} bp")


if __name__ == "__main__":
    main()
