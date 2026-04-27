import pandas as pd
from statsmodels.tsa.stattools import adfuller


class ADFTester:
    """
    Stage 2 of Step 3.

    Runs the Augmented Dickey-Fuller test on each NDEV_W series.
    Stationary series (p < 0.05) are candidates for AR(1) half-life.
    Non-stationary windows are flagged and excluded from Step 4 ranking.
    """

    P_VALUE_THRESHOLD = 0.05

    @staticmethod
    def test_window(
        series: pd.Series,
        window: int
    ) -> dict:
        """
        Run ADF on a single NDEV series.

        Parameters
        ----------
        series : normalised deviation series for one window
        window : the VWAP window integer (for labelling only)

        Returns
        -------
        dict with keys:
            window, adf_stat, p_value, n_lags_used,
            is_stationary, critical_values
        """
        if len(series) < 20:
            return {
                "window":         window,
                "adf_stat":       None,
                "p_value":        None,
                "n_lags_used":    None,
                "is_stationary":  False,
                "critical_values": {},
                "skip_reason":    "insufficient observations (<20)",
            }

        adf_stat, p_value, n_lags, _, crit_values, _ = adfuller(
            series,
            autolag="AIC",
            regression="c",   # constant only — deviations are mean-reverting, not trended
        )

        return {
            "window":          window,
            "adf_stat":        round(adf_stat, 6),
            "p_value":         round(p_value, 6),
            "n_lags_used":     n_lags,
            "is_stationary":   p_value < ADFTester.P_VALUE_THRESHOLD,
            "critical_values": {k: round(v, 4) for k, v in crit_values.items()},
            "skip_reason":     None,
        }

    @staticmethod
    def test_all(
        ndev_map: dict[int, pd.Series]
    ) -> pd.DataFrame:
        """
        Run ADF across all windows in ndev_map.

        Parameters
        ----------
        ndev_map : { window: pd.Series } from DeviationNormalizer.extract()

        Returns
        -------
        DataFrame indexed by window with columns:
            adf_stat | p_value | n_lags_used | is_stationary | skip_reason
        """
        rows = []

        for w, series in ndev_map.items():
            row = ADFTester.test_window(series, w)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("window")

        # Promote critical values to readable columns
        df["cv_1pct"]  = df["critical_values"].apply(lambda x: x.get("1%"))
        df["cv_5pct"]  = df["critical_values"].apply(lambda x: x.get("5%"))
        df["cv_10pct"] = df["critical_values"].apply(lambda x: x.get("10%"))
        df = df.drop(columns=["critical_values"])

        return df

    @staticmethod
    def stationary_windows(adf_results: pd.DataFrame) -> list[int]:
        """
        Returns list of windows that passed the stationarity test.
        Used by Stage 3 to know which windows get half-life computed.
        """
        return list(
            adf_results[adf_results["is_stationary"] == True].index
        )

    @staticmethod
    def flag_report(adf_results: pd.DataFrame) -> str:
        """
        Human-readable summary string.
        Useful for a quick console print before proceeding.
        """
        lines = []
        for w, row in adf_results.iterrows():
            status = "PASS" if row["is_stationary"] else "FAIL"
            p = row["p_value"]
            reason = f"  ← {row['skip_reason']}" if row["skip_reason"] else ""
            lines.append(f"  W={w:>4}  [{status}]  p={p}{reason}")
        return "ADF Results:\n" + "\n".join(lines)