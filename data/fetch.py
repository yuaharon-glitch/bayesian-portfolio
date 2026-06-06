"""yfinance download → log returns → Parquet cache (7-day TTL)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

START: str = "2010-01-01"
END: str = "2023-12-31"
CACHE_DIR: Path = Path(__file__).parent / "cache"
CACHE_MAX_AGE_SECONDS: int = 7 * 24 * 3600

UNIVERSE_10: list[str] = [
    "SPY",   # US large-cap equity
    "QQQ",   # US tech equity
    "IWM",   # US small-cap equity
    "EFA",   # Developed international equity
    "EEM",   # Emerging market equity
    "TLT",   # US 20+ year Treasuries
    "IEF",   # US 7-10 year Treasuries
    "GLD",   # Gold
    "GSG",   # Commodities
    "VNQ",   # US real estate
]

# 30 S&P 500 constituents: 5 per sector × 6 sectors
# Sectors: Technology, Healthcare, Financials, Consumer Discretionary,
#          Industrials, Energy
UNIVERSE_30: list[str] = [
    # Technology (5)
    "AAPL", "MSFT", "NVDA", "AVGO", "CRM",
    # Healthcare (5)
    "JNJ", "UNH", "LLY", "ABBV", "MRK",
    # Financials (5)
    "JPM", "BAC", "WFC", "GS", "MS",
    # Consumer Discretionary (5)
    "AMZN", "HD", "MCD", "NKE", "SBUX",
    # Industrials (5)
    "HON", "UPS", "CAT", "DE", "LMT",
    # Energy (5)
    "XOM", "CVX", "COP", "SLB", "EOG",
]


def _cache_path(label: str, start: str, end: str) -> Path:
    safe_label = label.replace(" ", "_").replace("/", "_")
    return CACHE_DIR / f"{safe_label}_{start}_{end}.parquet"


def _is_fresh(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_MAX_AGE_SECONDS


def _download_and_clean(
    tickers: list[str],
    start: str,
    end: str,
    max_missing_pct: float = 0.05,
    ffill_limit: int = 3,
) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for tickers: {tickers}")

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]

    missing_frac = prices.isna().mean()
    good_tickers = missing_frac[missing_frac < max_missing_pct].index.tolist()
    prices = prices[good_tickers].ffill(limit=ffill_limit)

    return np.log(prices / prices.shift(1)).dropna()


def fetch_returns(
    tickers: list[str] | None = None,
    start: str = START,
    end: str = END,
    label: str | None = None,
) -> pd.DataFrame:
    """Log returns for `tickers` from `start` to `end`, cached as Parquet."""
    if tickers is None:
        tickers = UNIVERSE_10

    if label is None:
        label = f"universe{len(tickers)}"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(label, start, end)

    if _is_fresh(path):
        return pd.read_parquet(path)

    df = _download_and_clean(tickers, start, end)
    df.to_parquet(path)
    return df


def fetch_universe_10(start: str = START, end: str = END) -> pd.DataFrame:
    return fetch_returns(tickers=UNIVERSE_10, start=start, end=end, label="universe10")


def fetch_universe_30(start: str = START, end: str = END) -> pd.DataFrame:
    return fetch_returns(tickers=UNIVERSE_30, start=start, end=end, label="universe30")
