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


class LocalFileFetcher:
    """
    Fetcher for a local CSV file in the format produced by StocksPhi /
    standard Yahoo-export style:

        Date, Open, High, Low, Close, Adj Close, Volume

    where Date is a naive datetime string: 2008-01-01 09:15:00

    Steps performed:
        1. Read CSV, parse Date column as datetime.
        2. Drop 'Adj Close' — not used downstream.
        3. Localize naive timestamps to Asia/Kolkata.
        4. Optionally filter to [start, end] if supplied.
        5. Return DataFrame with DatetimeIndex and columns
           [Open, High, Low, Close, Volume] — same contract as YahooFetcher.

    Resampling from 1m to a coarser timeframe (e.g. "5min") is NOT done
    here; it remains the responsibility of OHLCVResampler called from
    run_data_pipeline, consistent with how the Yahoo path works.
    """

    TIMEZONE = "Asia/Kolkata"
    DATE_COL = "Date"
    DROP_COLS = ["Adj Close"]
    REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._tz = pytz.timezone(self.TIMEZONE)

    def fetch(
            self,
            start: datetime | None = None,
            end: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Load the CSV and return a tz-aware OHLCV DataFrame.

        Parameters
        ----------
        start : optional lower bound (inclusive). If naive, treated as IST.
        end   : optional upper bound (exclusive). If naive, treated as IST.

        Returns
        -------
        pd.DataFrame with DatetimeIndex (Asia/Kolkata) and columns
        [Open, High, Low, Close, Volume], sorted ascending.
        """
        df = pd.read_csv(
            self.file_path,
            parse_dates=[self.DATE_COL],
        )

        if df.empty:
            raise ValueError(f"LocalFileFetcher: CSV at '{self.file_path}' is empty.")

        # --- Validate columns ---
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"LocalFileFetcher: CSV missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        # --- Set index ---
        df = df.set_index(self.DATE_COL)
        df.index = pd.to_datetime(df.index)

        # --- Localize naive index to IST ---
        if df.index.tz is None:
            df.index = df.index.tz_localize(self._tz, ambiguous="infer", nonexistent="shift_forward")
        else:
            df.index = df.index.tz_convert(self._tz)

        # --- Drop unused columns ---
        for col in self.DROP_COLS:
            if col in df.columns:
                df = df.drop(columns=[col])

        # --- Keep only the standard OHLCV columns ---
        df = df[self.REQUIRED_COLS].copy()

        # --- Coerce numeric types (guard against string artefacts in CSV) ---
        for col in self.REQUIRED_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # --- Optional date filtering ---
        if start is not None:
            start_ts = self._to_tz_aware(start)
            df = df[df.index >= start_ts]

        if end is not None:
            end_ts = self._to_tz_aware(end)
            df = df[df.index < end_ts]

        if df.empty:
            raise ValueError(
                f"LocalFileFetcher: no data remains after filtering "
                f"[{start}, {end})."
            )

        df = df.sort_index()
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tz_aware(self, dt: datetime) -> datetime:
        """Ensure a datetime is tz-aware in IST."""
        if dt.tzinfo is None:
            return self._tz.localize(dt)
        return dt.astimezone(self._tz)
