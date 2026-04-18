import pandas as pd

class VWAPCalculator:
    """
    Rolling VWAP using price proxy = (H + L + C) / 3
    """

    @staticmethod
    def compute_rolling_vwap(
        df: pd.DataFrame,
        windows: list[int]
    ) -> pd.DataFrame:

        if df.empty:
            return df

        out = df.copy()

        # Price proxy
        out["price_proxy"] = (
            out["High"] + out["Low"] + out["Close"]
        ) / 3.0

        for w in windows:
            num = (out["price_proxy"] * out["Volume"]).rolling(w).sum()
            den = out["Volume"].rolling(w).sum()
            out[f"VWAP_{w}"] = num / den

            # Deviation (used in diagnostics)
            out[f"DEV_{w}"] = out["price_proxy"] - out[f"VWAP_{w}"]

        return out