import pandas as pd


class DeviationNormalizer:
    """
    Stage 1 of Step 3.

    Validates that NDEV_W columns exist on the base_vwap DataFrame
    and exposes them as clean per-window series for downstream
    econometric analysis (ADF, AR1 half-life).
    """

    @staticmethod
    def extract(
        df: pd.DataFrame,
        windows: list[int]
    ) -> dict[int, pd.Series]:
        """
        Returns a dict mapping each window to its NDEV series (dropna applied).

        Parameters
        ----------
        df      : base_vwap DataFrame output from VWAPCalculator
        windows : list of window integers

        Returns
        -------
        { window: pd.Series of normalised deviations }
        """
        result = {}

        for w in windows:
            col = f"NDEV_{w}"
            if col not in df.columns:
                raise ValueError(
                    f"Column {col} not found. "
                    f"Ensure VWAPCalculator has been run with window={w}."
                )
            series = df[col].dropna()
            if series.empty:
                raise ValueError(f"NDEV_{w} is entirely NaN after dropna.")

            result[w] = series

        return result

    @staticmethod
    def summary(
        ndev_map: dict[int, pd.Series]
    ) -> pd.DataFrame:
        """
        Quick sanity table: mean, std, min, max per window.
        Useful for a fast visual check before running ADF.
        """
        rows = []
        for w, series in ndev_map.items():
            rows.append({
                "window": w,
                "n_obs":  len(series),
                "mean":   series.mean(),
                "std":    series.std(),
                "min":    series.min(),
                "max":    series.max(),
            })
        return pd.DataFrame(rows).set_index("window")