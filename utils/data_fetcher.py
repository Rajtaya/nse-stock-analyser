"""
data_fetcher.py
---------------
Fetches OHLCV + info from Yahoo Finance.
Supports:
  - NSE stocks       (e.g. BEL  → BEL.NS)
  - Commodity futures (e.g. NG=F — passed as-is, no .NS suffix)

v4 changes:
  - COMMODITY_TICKERS set: tickers in this set skip the .NS suffix
  - is_commodity() auto-detects based on ticker pattern
  - Volume missing for some futures → gracefully filled with 0
"""

import yfinance as yf
import pandas as pd
from typing import Optional

NSE_SUFFIX = ".NS"

TIMEFRAMES = {
    "1 Month":  {"period": "1mo",  "interval": "1d"},
    "6 Months": {"period": "6mo",  "interval": "1d"},
    "1 Year":   {"period": "1y",   "interval": "1d"},
    "2 Years":  {"period": "2y",   "interval": "1wk"},
    "5 Years":  {"period": "5y",   "interval": "1wk"},
}

# Commodity / futures / index tickers — no .NS suffix needed
COMMODITY_TICKERS = {
    "NG=F",   # Natural Gas NYMEX
    "CL=F",   # Crude Oil WTI
    "BZ=F",   # Brent Crude
    "GC=F",   # Gold
    "SI=F",   # Silver
    "HG=F",   # Copper
    "ZW=F",   # Wheat
    "ZC=F",   # Corn
    "^NSEI",  # Nifty 50
    "^BSESN", # Sensex
}


def is_commodity(code: str) -> bool:
    c = code.strip().upper()
    return (c in COMMODITY_TICKERS or c.endswith("=F")
            or c.startswith("^") or "." in c)


def get_ticker_symbol(stock_code: str) -> str:
    code = stock_code.strip().upper()
    return code if is_commodity(code) else code + NSE_SUFFIX


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col
                      for col in df.columns]
    rename_map = {c: c.title() for c in df.columns if c.lower() in
                  {"open", "high", "low", "close", "volume", "adj close"}}
    return df.rename(columns=rename_map)


def fetch_stock_data(
    stock_code: str,
    period: str = "1y",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    ticker = get_ticker_symbol(stock_code)
    try:
        raw = yf.download(
            ticker, period=period, interval=interval,
            progress=False, auto_adjust=True,
        )
        if raw is None or raw.empty:
            return None

        df = _flatten_columns(raw.copy())

        # Commodities may have no Volume column
        if "Volume" not in df.columns:
            df["Volume"] = 0

        needed = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in df.columns for c in needed):
            return None

        df = df[needed].copy()
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype("int64")
        return df.sort_index()

    except Exception as exc:
        print(f"[data_fetcher] Could not fetch {ticker}: {exc}")
        return None


def fetch_stock_info(stock_code: str) -> dict:
    ticker = get_ticker_symbol(stock_code)
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}
