"""
Convert minute-bar dicts from price_generator into the OHLCV DataFrame format
used by VWAP and cleaning pipelines.
"""

from __future__ import annotations

import pandas as pd

REQUIRED = ("Open", "High", "Low", "Close", "Volume")


def bars_to_ohlcv_dataframe(bars: list[dict]) -> pd.DataFrame:
    """
    Build a DataFrame indexed by bar time from a list of dicts.

    Each dict should include:
      - timestamp: datetime-like (index)
      - Open, High, Low, Close, Volume: floats

    The optional \"price\" field (typically == Close) is ignored for the frame;
    it is kept on the wire for lightweight consumers.
    """
    if not bars:
        return pd.DataFrame(columns=list(REQUIRED))

    rows = []
    for b in bars:
        ts = b.get("timestamp")
        if ts is None:
            continue
        try:
            rows.append(
                {
                    "timestamp": ts,
                    "Open": float(b["Open"]),
                    "High": float(b["High"]),
                    "Low": float(b["Low"]),
                    "Close": float(b["Close"]),
                    "Volume": float(b.get("Volume", 0) or 0),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    if not rows:
        return pd.DataFrame(columns=list(REQUIRED))

    df = pd.DataFrame(rows)
    df = df.set_index("timestamp").sort_index()
    return df[list(REQUIRED)]
