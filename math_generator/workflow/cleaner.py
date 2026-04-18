import pandas as pd


class OHLVCleaner:
    REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # Flatten MultiIndex (Yahoo intraday quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Validate required columns
        for col in OHLVCleaner.REQUIRED_COLS:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Forward-fill OHLC
        price_cols = ["Open", "High", "Low", "Close"]
        df[price_cols] = df[price_cols].ffill()

        # Volume cleanup
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        df = df[df["Volume"] > 0]

        # Drop NaNs
        df = df.dropna(subset=OHLVCleaner.REQUIRED_COLS)

        # Sort time
        df = df.sort_index()

        return df[OHLVCleaner.REQUIRED_COLS]
