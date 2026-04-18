import pandas as pd

class RollingVWAPStability:
    """
    Rolling stability diagnostics for VWAP deviations.
    """

    @staticmethod
    def compute(
        df: pd.DataFrame,
        window: int,
        roll_len: int = 100,
        price_col: str = "Close"
    ) -> pd.DataFrame:
        """
        Computes rolling stability statistics for VWAP deviations.

        Parameters
        ----------
        df : DataFrame with VWAP_{window}
        window : VWAP window length (bars)
        roll_len : rolling diagnostic window (bars)
        price_col : price proxy column

        Returns
        -------
        DataFrame with rolling stability metrics
        """

        vwap_col = f"VWAP_{window}"
        if vwap_col not in df.columns:
            raise ValueError(f"{vwap_col} not found")

        dev = df[price_col] - df[vwap_col]

        out = pd.DataFrame(index=df.index)
        out["dev_mean"] = dev.rolling(roll_len).mean()
        out["dev_std"] = dev.rolling(roll_len).std()
        out["dev_z"] = dev / out["dev_std"]

        # Stability-of-stability
        out["mean_instability"] = out["dev_mean"].rolling(roll_len).std()
        out["std_instability"] = out["dev_std"].rolling(roll_len).std()

        # Scale-normalized noise
        out["cv"] = out["dev_std"] / dev.abs().rolling(roll_len).mean()

        return out.dropna()
