import yfinance as yf
import pandas as pd
from datetime import datetime


class YahooFetcher:
    """
    Yahoo Finance OHLCV fetcher
    """
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        df = yf.download(
            self.symbol,
            start=start,
            end=end,
            interval=self.interval,
            auto_adjust=False,
            progress=False
        )

        if df.empty:
            raise ValueError(f"No data fetched for {self.symbol}")

        return df
