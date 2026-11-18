"""复现 12-1 动量因子并与 Ken French UMD 因子对比。

用法:
    python run.py                       # 默认 2000-01 起，单边成本 10bp
    python run.py --start 1995-01-01 --cost-bps 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from momentum.analysis import factor_regression, perf_stats
from momentum.data import get_french_factors, get_prices, get_sp500_tickers
from momentum.factor import build_wml

OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--lookback", type=int, default=12)
    p.add_argument("--skip", type=int, default=1)
    p.add_argument("--buckets", type=int, default=10)
    p.add_argument("--cost-bps", type=float, default=10.0, help="单边交易成本（基点）")
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)

    tickers = get_sp500_tickers()
    print(f"Universe: {len(tickers)} current S&P 500 tickers")
    daily = get_prices(tickers, args.start, args.end)
    print(f"Prices: {daily.shape[1]} tickers, {daily.index[0].date()} -> {daily.index[-1].date()}")

    res = build_wml(daily, args.lookback, args.skip, args.buckets, args.cost_bps)
    rets, wml, wml_net = res["bucket_returns"], res["wml"], res["wml_net"]

    # 最后一个月可能不完整，剔除
    last_full = daily.index[-1] + pd.offsets.MonthEnd(0)
    if daily.index[-1] < last_full:
        rets, wml, wml_net = (x.loc[x.index < last_full] for x in (rets, wml, wml_net))

    ff = get_french_factors()
    common = wml.index.intersection(ff.index)
    print(f"Overlap with French data: {common[0]:%Y-%m} -> {common[-1]:%Y-%m} ({len(common)} months)")

    # ---------- 1. 绩效 ----------
    stats = pd.DataFrame({
        "WML (gross)": perf_stats(wml.loc[common]),
        f"WML (net {args.cost_bps:g}bp)": perf_stats(wml_net.loc[common]),
        "French UMD": perf_stats(ff.loc[common, "UMD"]),
        "Mkt-RF": perf_stats(ff.loc[common, "Mkt-RF"]),
    })
    stats.to_csv(OUT / "performance.csv")

    # ---------- 2. 十分位单调性 ----------
    dec = pd.DataFrame({
        "ann_return": rets.loc[common].mean() * 12,
        "ann_vol": rets.loc[common].std() * np.sqrt(12),
    })
    dec["sharpe"] = dec["ann_return"] / dec["ann_vol"]
    dec.index.name = "decile"
    dec.to_csv(OUT / "decile_returns.csv")

    # ---------- 3. 与 UMD 相关 & 回归 ----------
    corr = wml.loc[common].corr(ff.loc[common, "UMD"])
    reg_umd = factor_regression(wml, ff[["UMD"]])
    reg_ff3 = factor_regression(wml, ff[["Mkt-RF", "SMB", "HML"]])

    def reg_table(fit) -> pd.DataFrame:
        t = pd.DataFrame({"coef": fit.params, "t_stat": fit.tvalues})
        t.loc["const", "coef"] *= 12  # 年化 alpha
        t = t.rename(index={"const": "alpha (ann.)"})
        t.loc["R2", "coef"] = fit.rsquared
        return t

    reg = pd.concat({"WML ~ UMD": reg_table(reg_umd), "WML ~ FF3": reg_table(reg_ff3)})
    reg.to_csv(OUT / "regressions.csv")

    # ---------- 4. 图 ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    for s, lbl in [(wml, "WML (S&P 500, gross)"), (wml_net, f"WML (net {args.cost_bps:g}bp)"),
                   (ff["UMD"], "Ken French UMD")]:
        np.log1p(s.loc[common]).cumsum().plot(ax=ax, label=lbl, lw=1.4)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title("12-1 Momentum: cumulative log return")
    ax.set_ylabel("cumulative log return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cumulative.png", dpi=140)

    fig, ax = plt.subplots(figsize=(8, 4))
    (dec["ann_return"] * 100).plot.bar(ax=ax, color="#4C72B0")
    ax.set_title("Annualized return by momentum decile (1 = losers, 10 = winners)")
    ax.set_ylabel("%")
    ax.set_xlabel("decile")
    fig.tight_layout()
    fig.savefig(OUT / "deciles.png", dpi=140)

    roll = wml.loc[common].rolling(36).corr(ff.loc[common, "UMD"])
    fig, ax = plt.subplots(figsize=(10, 3.5))
    roll.plot(ax=ax, color="#C44E52")
    ax.set_ylim(0, 1)
    ax.set_title("Rolling 36-month correlation: our WML vs French UMD")
    fig.tight_layout()
    fig.savefig(OUT / "rolling_corr.png", dpi=140)

    # ---------- 输出 ----------
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("\n=== Performance ===\n", stats)
    print("\n=== Deciles ===\n", dec)
    print(f"\nCorrelation(WML, UMD) = {corr:.3f}")
    print(f"Average monthly turnover (long+short legs) = {res['turnover'].mean():.2%}")
    print("\n=== Regressions (Newey-West t) ===\n", reg)
    pd.Series({"corr_wml_umd": corr, "avg_turnover": res["turnover"].mean()}).to_csv(OUT / "summary.csv")


if __name__ == "__main__":
    main()
