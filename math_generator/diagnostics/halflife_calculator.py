import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


class HalfLifeCalculator:
    """
    Stage 3 of Step 3.

    Fits an AR(1) model on each stationary NDEV_W series and derives
    the mean-reversion half-life in bars.

    Model:  ΔNDEV_t = α + (φ − 1)·NDEV_(t-1) + ε
    Equivalently: NDEV_t = α + φ·NDEV_(t-1) + ε

    Half-life = −ln(2) / ln(φ)

    φ must satisfy 0 < φ < 1 for a valid mean-reverting process.
    If φ ≥ 1 the series is explosive; if φ ≤ 0 it oscillates.
    Both are flagged as invalid.
    """

    @staticmethod
    def compute_window(
        series: pd.Series,
        window: int
    ) -> dict:
        """
        Fit AR(1) on a single NDEV series and return half-life metadata.

        Parameters
        ----------
        series : stationary NDEV series for one window
        window : VWAP window integer (for labelling only)

        Returns
        -------
        dict with keys:
            window, phi, half_life_bars, valid, skip_reason
        """
        if len(series) < 20:
            return {
                "window":         window,
                "phi":            None,
                "half_life_bars": None,
                "valid":          False,
                "skip_reason":    "insufficient observations (<20)",
            }

        y = series.values[1:]        # NDEV_t
        x = series.values[:-1]       # NDEV_(t-1)

        x_const = add_constant(x)
        model   = OLS(y, x_const).fit()

        phi = model.params[1]        # AR(1) coefficient

        # Validity gate: φ must be in (0, 1) for mean reversion
        if phi <= 0:
            return {
                "window":         window,
                "phi":            round(float(phi), 6),
                "half_life_bars": None,
                "valid":          False,
                "skip_reason":    f"phi={phi:.4f} ≤ 0 (oscillating, not mean-reverting)",
            }

        if phi >= 1:
            return {
                "window":         window,
                "phi":            round(float(phi), 6),
                "half_life_bars": None,
                "valid":          False,
                "skip_reason":    f"phi={phi:.4f} ≥ 1 (explosive process)",
            }

        half_life = -np.log(2) / np.log(phi)

        return {
            "window":         window,
            "phi":            round(float(phi), 6),
            "half_life_bars": round(float(half_life), 2),
            "valid":          True,
            "skip_reason":    None,
        }

    @staticmethod
    def compute_all(
        ndev_map: dict[int, pd.Series],
        stationary_windows: list[int]
    ) -> pd.DataFrame:
        """
        Compute half-life for all windows that passed the ADF test.
        Non-stationary windows are recorded as skipped.

        Parameters
        ----------
        ndev_map           : { window: pd.Series } from DeviationNormalizer
        stationary_windows : windows that passed ADF (from ADFTester)

        Returns
        -------
        DataFrame indexed by window with columns:
            phi | half_life_bars | valid | skip_reason
        """
        rows = []

        for w, series in ndev_map.items():
            if w not in stationary_windows:
                rows.append({
                    "window":         w,
                    "phi":            None,
                    "half_life_bars": None,
                    "valid":          False,
                    "skip_reason":    "skipped — did not pass ADF test",
                })
                continue

            row = HalfLifeCalculator.compute_window(series, w)
            rows.append(row)

        return pd.DataFrame(rows).set_index("window")

    @staticmethod
    def flag_report(hl_results: pd.DataFrame) -> str:
        """
        Human-readable summary string for console output.
        """
        lines = []
        for w, row in hl_results.iterrows():
            if row["valid"]:
                lines.append(
                    f"  W={w:>4}  [VALID]  "
                    f"φ={row['phi']:.4f}  "
                    f"half_life={row['half_life_bars']:.1f} bars"
                )
            else:
                lines.append(
                    f"  W={w:>4}  [SKIP ]  {row['skip_reason']}"
                )
        return "AR(1) Half-Life Results:\n" + "\n".join(lines)