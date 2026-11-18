import numpy as np
import pandas as pd

from momentum.factor import assign_buckets, bucket_returns, build_wml, momentum_signal


def _toy_prices(n_stocks=60, n_months=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rets = rng.normal(0.01, 0.08, size=(n_months, n_stocks))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx,
                        columns=[f"S{i}" for i in range(n_stocks)])


def test_signal_skips_last_month():
    p = _toy_prices()
    sig = momentum_signal(p, 12, 1)
    t = p.index[15]
    expected = p.loc[p.index[14]] / p.loc[p.index[3]] - 1
    pd.testing.assert_series_equal(sig.loc[t], expected, check_names=False)
    assert sig.iloc[:12].isna().all().all()


def test_signal_has_no_lookahead():
    """篡改 t 月及之后的价格，t 月信号不应改变。"""
    p = _toy_prices()
    t = p.index[20]
    sig1 = momentum_signal(p, 12, 1).loc[t]
    p2 = p.copy()
    p2.loc[t:] *= 3.0
    sig2 = momentum_signal(p2, 12, 1).loc[t]
    pd.testing.assert_series_equal(sig1, sig2)


def test_buckets_balanced():
    p = _toy_prices(n_stocks=100)
    b = assign_buckets(momentum_signal(p), 10)
    counts = b.iloc[-1].value_counts()
    assert sorted(counts.index) == list(range(1, 11))
    assert (counts == 10).all()


def test_bucket_returns_use_next_month():
    """组合在 t 形成、收益记在 t+1。"""
    p = _toy_prices(n_stocks=60)
    b = assign_buckets(momentum_signal(p), 10)
    r = bucket_returns(b, p)
    t = p.index[15]
    members = b.loc[t][b.loc[t] == 10].index
    expected = (p.loc[p.index[16], members] / p.loc[t, members] - 1).mean()
    assert np.isclose(r.loc[p.index[16], 10], expected)


def test_wml_recovers_planted_momentum():
    """构造有持续性的收益：过去赢家下月继续跑赢，WML 应显著为正。"""
    rng = np.random.default_rng(1)
    n, T = 100, 120
    drift = rng.normal(0, 0.02, n)  # 每只股票固定的漂移 → 动量持续
    rets = drift + rng.normal(0, 0.03, (T, n))
    idx = pd.date_range("2000-01-31", periods=T, freq="ME")
    daily = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx)
    wml = build_wml(daily)["wml"]
    assert wml.mean() > 0.02
