import pandas as pd

class VWAPStabilityDiagnostics:
    """
    Quantifies statistical stability of VWAP deviations.
    No plotting. No trading logic.
    """

    @staticmethod
    def window_stability(
        df: pd.DataFrame,
        windows: list[int]
    ) -> pd.DataFrame:
        """
        Compare DEV statistics across VWAP windows
        using the full available history.
        """

        rows = []

        for W in windows:
            dev_col = f"DEV_{W}"

            if dev_col not in df.columns:
                continue

            series = df[dev_col].dropna()

            rows.append({
                "window": W,
                "mean": series.mean(),
                "std": series.std(),
                "skew": series.skew(),
                "kurtosis": series.kurtosis(),
                "n_obs": len(series)
            })

        return pd.DataFrame(rows).set_index("window")

    @staticmethod
    def history_stability(
        df: pd.DataFrame,
        window: int,
        history_days: list[int]
    ) -> pd.DataFrame:
        """
        Measure how DEV statistics change as we increase
        available history length.
        """

        dev_col = f"DEV_{window}"
        if dev_col not in df.columns:
            raise ValueError(f"Missing {dev_col} in DataFrame")

        rows = []
        last_ts = df.index.max()

        for days in history_days:
            cutoff = last_ts - pd.Timedelta(days=days)
            subset = df.loc[df.index >= cutoff, dev_col].dropna()

            if len(subset) == 0:
                continue

            rows.append({
                "days": days,
                "mean": subset.mean(),
                "std": subset.std(),
                "skew": subset.skew(),
                "kurtosis": subset.kurtosis(),
                "n_obs": len(subset)
            })

        return pd.DataFrame(rows).set_index("days")
