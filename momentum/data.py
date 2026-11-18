"""数据获取：S&P 500 成分股、日度复权价格（yfinance）、Ken French 因子库。

所有下载结果缓存在 ./data 目录下，重复运行不会重新下载。
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script)"}

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def get_sp500_tickers(refresh: bool = False) -> list[str]:
    """当前 S&P 500 成分股列表（注意：使用当前成分股会引入幸存者偏差）。"""
    cache = DATA_DIR / "sp500_tickers.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)["ticker"].tolist()

    html = requests.get(SP500_URL, headers=HEADERS, timeout=30).text
    table = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]
    # Yahoo 使用 "-" 代替 "."，例如 BRK.B -> BRK-B
    tickers = sorted(table["Symbol"].str.replace(".", "-", regex=False).unique())
    DATA_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(cache, index=False)
    return tickers


def get_prices(tickers: list[str], start: str, end: str | None = None,
               refresh: bool = False) -> pd.DataFrame:
    """日度复权收盘价，index=日期，columns=股票代码。"""
    import yfinance as yf

    cache = DATA_DIR / f"prices_{start}_{end or 'latest'}.csv.gz"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, threads=True, group_by="column")
    prices = raw["Close"].dropna(how="all", axis=1).sort_index()
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(cache, compression="gzip")
    return prices


def _read_french_zip(filename: str) -> str:
    cache = DATA_DIR / filename
    if not cache.exists():
        resp = requests.get(FRENCH_BASE + filename, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        DATA_DIR.mkdir(exist_ok=True)
        cache.write_bytes(resp.content)
    with zipfile.ZipFile(cache) as zf:
        return zf.read(zf.namelist()[0]).decode("latin-1")


def _parse_french_monthly(text: str) -> pd.DataFrame:
    """解析 French CSV 中的月度部分（第一张表，YYYYMM 索引，单位 %）。"""
    lines = text.splitlines()
    rows, header = [], None
    for line in lines:
        cells = [c.strip() for c in line.split(",")]
        if header is None:
            # 表头行：首格为空，其余为因子名
            if len(cells) > 1 and cells[0] == "" and all(cells[1:]):
                header = cells[1:]
            continue
        if len(cells[0]) == 6 and cells[0].isdigit():
            rows.append(cells)
        elif rows:  # 月度表结束（后面是年度表）
            break
    df = pd.DataFrame(rows).set_index(0).astype(float) / 100.0
    df.columns = header
    df.index = pd.to_datetime(df.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    df.index.name = "date"
    return df


def get_french_factors() -> pd.DataFrame:
    """Fama-French 三因子 + 动量因子 UMD（月度，小数形式）。"""
    ff3 = _parse_french_monthly(_read_french_zip("F-F_Research_Data_Factors_CSV.zip"))
    mom = _parse_french_monthly(_read_french_zip("F-F_Momentum_Factor_CSV.zip"))
    mom.columns = ["UMD"]
    return ff3.join(mom, how="inner")
