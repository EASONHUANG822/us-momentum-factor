"""绩效统计与因子回归。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def perf_stats(r: pd.Series) -> pd.Series:
    """月度收益序列的年化绩效指标。"""
    r = r.dropna()
    n = len(r)
    mean, std = r.mean(), r.std()
    t_stat = mean / (std / np.sqrt(n))
    wealth = (1 + r).cumprod()
    max_dd = (wealth / wealth.cummax() - 1).min()
    return pd.Series({
        "months": n,
        "ann_return": mean * 12,
        "ann_vol": std * np.sqrt(12),
        "sharpe": mean / std * np.sqrt(12),
        "t_stat": t_stat,
        "max_drawdown": max_dd,
        "worst_month": r.min(),
        "skew": r.skew(),
        "hit_rate": (r > 0).mean(),
    })


def factor_regression(y: pd.Series, X: pd.DataFrame, nw_lags: int = 6):
    """OLS + Newey-West 标准误。alpha 为月度值。"""
    df = pd.concat([y, X], axis=1, join="inner").dropna()
    model = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1:]))
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
