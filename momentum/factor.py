"""Jegadeesh & Titman (1993) 截面动量因子：12-1 动量，十分位排序，持有 1 个月。

时间约定（避免前视偏差）：
  在月末 t 计算信号  mom_t = P_{t-1} / P_{t-12} - 1   （跳过最近一个月，规避短期反转）
  在月末 t 建仓，持有至月末 t+1，获得收益 r_{t+1}
  组合收益记在 t+1 月。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_prices(daily: pd.DataFrame) -> pd.DataFrame:
    """日度价格 -> 月末价格。"""
    return daily.resample("ME").last()


def momentum_signal(prices_m: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """t 时刻的动量信号，仅使用 t-skip 及更早的价格。

    要求 t-lookback 与 t-skip 两端价格都存在，且 t 时刻有价格（股票仍在交易）。
    """
    sig = prices_m.shift(skip) / prices_m.shift(lookback) - 1.0
    return sig.where(prices_m.notna())


def assign_buckets(signal: pd.DataFrame, n_buckets: int = 10, min_stocks: int = 50) -> pd.DataFrame:
    """每个月按信号截面排序分组：1=输家(最低)，n=赢家(最高)。样本不足的月份置空。"""
    def _row(s: pd.Series) -> pd.Series:
        s = s.dropna()
        if len(s) < min_stocks:
            return pd.Series(dtype=float)
        ranks = s.rank(method="first")
        return np.ceil(ranks / len(s) * n_buckets).clip(1, n_buckets)

    return signal.apply(_row, axis=1).reindex(columns=signal.columns)


def bucket_returns(buckets: pd.DataFrame, prices_m: pd.DataFrame) -> pd.DataFrame:
    """各组下月等权收益。返回 index=持有月 (t+1)，columns=组号。

    t 月形成的组合用 t+1 月收益；若 t+1 月收益缺失（如退市），该股票从当月组合中剔除。
    """
    fwd_ret = (prices_m.shift(-1) / prices_m - 1.0)  # r_{t+1} 对齐在 t
    out = {}
    for b in sorted(pd.unique(buckets.stack().dropna())):
        mask = buckets == b
        out[int(b)] = fwd_ret.where(mask).mean(axis=1)
    res = pd.DataFrame(out)
    res.index = res.index + pd.offsets.MonthEnd(1)  # 移到持有月
    return res.dropna(how="all")


def turnover(buckets: pd.DataFrame, bucket: int) -> pd.Series:
    """某一组等权组合的月度单边换手率 = 0.5 * Σ|w_t - w_{t-1}|（忽略组内价格漂移）。"""
    w = (buckets == bucket).astype(float)
    w = w.div(w.sum(axis=1), axis=0).fillna(0.0)
    to = 0.5 * (w - w.shift(1)).abs().sum(axis=1)
    to.iloc[0] = np.nan
    return to


def build_wml(daily_prices: pd.DataFrame, lookback: int = 12, skip: int = 1,
              n_buckets: int = 10, cost_bps: float = 0.0) -> dict:
    """完整流水线：返回各组收益、WML（赢家减输家）以及换手率。

    cost_bps：单边交易成本（基点），WML 扣除多空两腿成本：
        cost = (to_long + to_short) * 2 * cost_bps/1e4   （买入 + 卖出）
    """
    pm = monthly_prices(daily_prices)
    sig = momentum_signal(pm, lookback, skip)
    buckets = assign_buckets(sig, n_buckets)
    rets = bucket_returns(buckets, pm)

    wml = rets[n_buckets] - rets[1]
    to_long = turnover(buckets, n_buckets)
    to_short = turnover(buckets, 1)
    to = (to_long + to_short)
    to.index = to.index + pd.offsets.MonthEnd(1)
    to = to.reindex(wml.index)
    wml_net = wml - to.fillna(0) * 2 * cost_bps / 1e4

    return {
        "monthly_prices": pm,
        "signal": sig,
        "buckets": buckets,
        "bucket_returns": rets,
        "wml": wml.rename("WML"),
        "wml_net": wml_net.rename("WML_net"),
        "turnover": to.rename("turnover"),
    }
