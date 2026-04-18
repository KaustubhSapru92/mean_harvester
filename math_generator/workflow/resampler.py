import pandas as pd


class OHLCVResampler:
    @staticmethod
    def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if df.empty:
            return df

        agg = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }

        resampled = df.resample(timeframe).agg(agg)
        resampled = resampled.dropna()

        return resampled