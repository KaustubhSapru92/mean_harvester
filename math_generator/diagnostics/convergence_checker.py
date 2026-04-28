import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


class ConvergenceChecker:
    """
    Stage 4 of Step 3.

    Re-runs ADF and AR(1) half-life on growing prefix slices of each
    stationary NDEV_W series to determine how much history is needed
    before both metrics stabilise.

    A window is "converged" when both ADF p-value and half-life change
    by less than STABILITY_THRESHOLD across the final TAIL_FRACTION
    of the slice sequence.
    """

    STABILITY_THRESHOLD = 0.10   # max allowed relative change (10%)
    TAIL_FRACTION       = 0.10   # final 10% of slices used for stability check
    MIN_SLICE_OBS       = 20     # minimum bars needed to run ADF + OLS
    N_SLICES            = 20     # how many prefix slices to evaluate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adf_p(series: pd.Series) -> float | None:
        if len(series) < ConvergenceChecker.MIN_SLICE_OBS:
            return None
        try:
            _, p, *_ = adfuller(series, autolag="AIC", regression="c")
            return float(p)
        except Exception:
            return None

    @staticmethod
    def _half_life(series: pd.Series) -> float | None:
        if len(series) < ConvergenceChecker.MIN_SLICE_OBS:
            return None
        try:
            y = series.values[1:]
            x = add_constant(series.values[:-1])
            phi = OLS(y, x).fit().params[1]
            if not (0 < phi < 1):
                return None
            return float(-np.log(2) / np.log(phi))
        except Exception:
            return None

    @staticmethod
    def _is_stable(values: list[float | None]) -> bool:
        """
        Check whether the tail of a metric sequence is stable.
        Stable = max relative change across tail < STABILITY_THRESHOLD.
        """
        clean = [v for v in values if v is not None]
        if len(clean) < 2:
            return False

        n_tail = max(2, int(len(clean) * ConvergenceChecker.TAIL_FRACTION))
        tail   = clean[-n_tail:]

        base = abs(tail[0])
        if base == 0:
            return all(abs(v) < 1e-8 for v in tail)

        rel_changes = [abs(tail[i] - tail[i - 1]) / base for i in range(1, len(tail))]
        return max(rel_changes) < ConvergenceChecker.STABILITY_THRESHOLD

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def check_window(
        series: pd.Series,
        window: int
    ) -> dict:
        """
        Run prefix-slice convergence analysis on a single NDEV series.

        Parameters
        ----------
        series : stationary NDEV series for one window
        window : VWAP window integer (for labelling only)

        Returns
        -------
        dict with keys:
            window, converged, n_slices_run,
            slice_fractions, p_values, half_lives,
            p_stable, hl_stable
        """
        n        = len(series)
        n_slices = ConvergenceChecker.N_SLICES

        # Build evenly spaced prefix lengths from MIN_SLICE_OBS to full length
        min_obs   = ConvergenceChecker.MIN_SLICE_OBS
        lengths   = np.linspace(min_obs, n, n_slices, dtype=int)
        lengths   = np.unique(lengths)          # remove duplicates at short series

        fractions  = []
        p_values   = []
        half_lives = []

        for length in lengths:
            prefix = series.iloc[:length]
            fractions.append(round(length / n, 3))
            p_values.append(ConvergenceChecker._adf_p(prefix))
            half_lives.append(ConvergenceChecker._half_life(prefix))

        p_stable  = ConvergenceChecker._is_stable(p_values)
        hl_stable = ConvergenceChecker._is_stable(half_lives)
        converged = p_stable and hl_stable

        return {
            "window":          window,
            "converged":       converged,
            "p_stable":        p_stable,
            "hl_stable":       hl_stable,
            "n_slices_run":    len(lengths),
            "slice_fractions": fractions,
            "p_values":        p_values,
            "half_lives":      half_lives,
        }

    @staticmethod
    def check_all(
        ndev_map: dict[int, pd.Series],
        stationary_windows: list[int]
    ) -> tuple[pd.DataFrame, dict[int, dict]]:
        """
        Run convergence check for all stationary windows.

        Parameters
        ----------
        ndev_map           : { window: pd.Series } from DeviationNormalizer
        stationary_windows : windows that passed ADF (from ADFTester)

        Returns
        -------
        summary_df  : DataFrame indexed by window with columns:
                        converged | p_stable | hl_stable | n_slices_run
        detail_map  : { window: full check_window dict } for plotting
        """
        summary_rows = []
        detail_map   = {}

        for w, series in ndev_map.items():
            if w not in stationary_windows:
                summary_rows.append({
                    "window":       w,
                    "converged":    False,
                    "p_stable":     False,
                    "hl_stable":    False,
                    "n_slices_run": 0,
                })
                continue

            result = ConvergenceChecker.check_window(series, w)
            detail_map[w] = result

            summary_rows.append({
                "window":       w,
                "converged":    result["converged"],
                "p_stable":     result["p_stable"],
                "hl_stable":    result["hl_stable"],
                "n_slices_run": result["n_slices_run"],
            })

        summary_df = pd.DataFrame(summary_rows).set_index("window")
        return summary_df, detail_map

    @staticmethod
    def flag_report(summary_df: pd.DataFrame) -> str:
        """
        Human-readable summary string for console output.
        """
        lines = []
        for w, row in summary_df.iterrows():
            status   = "CONVERGED" if row["converged"]  else "NOT CONVERGED"
            p_tag    = "p✓"        if row["p_stable"]   else "p✗"
            hl_tag   = "hl✓"       if row["hl_stable"]  else "hl✗"
            lines.append(
                f"  W={w:>4}  [{status:<13}]  {p_tag}  {hl_tag}  "
                f"slices={int(row['n_slices_run'])}"
            )
        return "Convergence Results:\n" + "\n".join(lines)