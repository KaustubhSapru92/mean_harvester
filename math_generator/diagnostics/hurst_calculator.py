import numpy as np
import pandas as pd


class HurstCalculator:
    """
    Step 5 — Stage 2.

    Computes the Hurst exponent via Rescaled Range (R/S) analysis
    on the low-volatility subset of each eligible NDEV_W series.

    Interpretation:
        H < 0.5  — mean-reverting (desired)
        H = 0.5  — random walk
        H > 0.5  — trending / persistent

    Method (R/S analysis):
        For a range of sub-period lengths n, compute:
            R(n) = max(cumulative deviation) - min(cumulative deviation)
            S(n) = std(series subset)
            RS(n) = R(n) / S(n)

        H = slope of log(RS) vs log(n) via OLS.

    A minimum of hurst_window bars is required after applying the
    low-vol mask. Windows with fewer surviving bars are flagged as
    insufficient and excluded from Step 6.
    """

    H_THRESHOLD = 0.5   # H must be strictly below this to confirm mean reversion

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rs_for_length(series: np.ndarray, n: int) -> float | None:
        """
        Compute mean R/S statistic for sub-period length n.
        Splits series into non-overlapping blocks of length n.
        """
        n_blocks = len(series) // n
        if n_blocks < 1:
            return None

        rs_values = []
        for i in range(n_blocks):
            block = series[i * n: (i + 1) * n]
            mean  = block.mean()
            dev   = np.cumsum(block - mean)
            r     = dev.max() - dev.min()
            s     = block.std(ddof=1)
            if s == 0:
                continue
            rs_values.append(r / s)

        if not rs_values:
            return None

        return float(np.mean(rs_values))

    @staticmethod
    def _compute_hurst(series: np.ndarray, min_window: int) -> float | None:
        """
        Estimate H from OLS slope of log(RS) vs log(n).

        Sub-period lengths are spaced geometrically between
        min_window // 4 and len(series) // 2.
        """
        n     = len(series)
        low   = max(10, min_window // 4)
        high  = n // 2

        if high <= low:
            return None

        lengths = np.unique(
            np.geomspace(low, high, num=20).astype(int)
        )

        log_n  = []
        log_rs = []

        for length in lengths:
            rs = HurstCalculator._rs_for_length(series, length)
            if rs is not None and rs > 0:
                log_n.append(np.log(length))
                log_rs.append(np.log(rs))

        if len(log_n) < 4:
            return None

        # OLS: log_rs = H * log_n + const
        log_n_arr  = np.array(log_n)
        log_rs_arr = np.array(log_rs)
        A          = np.vstack([log_n_arr, np.ones(len(log_n_arr))]).T
        H, _       = np.linalg.lstsq(A, log_rs_arr, rcond=None)[0]

        return float(round(H, 6))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def compute_window(
        ndev_series: pd.Series,
        low_vol_mask: pd.Series,
        window: int,
        hurst_window: int = 100,
    ) -> dict:
        """
        Apply low-vol mask and compute Hurst exponent for one window.

        Parameters
        ----------
        ndev_series  : full NDEV_W series (unmasked)
        low_vol_mask : boolean Series aligned to ndev_series index
        window       : VWAP window integer (for labelling only)
        hurst_window : minimum bars required after masking

        Returns
        -------
        dict with keys:
            window, n_low_vol_bars, hurst_H, hurst_pass,
            valid, skip_reason
        """
        # Align mask to series index
        aligned_mask = low_vol_mask.reindex(ndev_series.index).fillna(False)
        low_vol_series = ndev_series[aligned_mask].dropna()
        n = len(low_vol_series)

        if n < hurst_window:
            return {
                "window":         window,
                "n_low_vol_bars": n,
                "hurst_H":        None,
                "hurst_pass":     False,
                "valid":          False,
                "skip_reason":    (
                    f"insufficient low-vol bars after masking "
                    f"({n} < {hurst_window})"
                ),
            }

        H = HurstCalculator._compute_hurst(
            low_vol_series.values,
            hurst_window
        )

        if H is None:
            return {
                "window":         window,
                "n_low_vol_bars": n,
                "hurst_H":        None,
                "hurst_pass":     False,
                "valid":          False,
                "skip_reason":    "R/S analysis failed — insufficient variance in blocks",
            }

        hurst_pass = H < HurstCalculator.H_THRESHOLD

        return {
            "window":         window,
            "n_low_vol_bars": n,
            "hurst_H":        H,
            "hurst_pass":     hurst_pass,
            "valid":          True,
            "skip_reason":    None if hurst_pass else (
                f"H={H:.4f} ≥ {HurstCalculator.H_THRESHOLD} "
                f"(not mean-reverting in low-vol regime)"
            ),
        }

    @staticmethod
    def compute_all(
        ndev_map: dict[int, pd.Series],
        low_vol_mask: pd.Series,
        eligible_windows: list[int],
        hurst_window: int = 100,
    ) -> pd.DataFrame:
        """
        Compute Hurst exponent for all eligible windows.
        Ineligible windows are recorded as skipped.

        Parameters
        ----------
        ndev_map         : { window: pd.Series } from DeviationNormalizer
        low_vol_mask     : boolean Series from ATRCalculator
        eligible_windows : windows that passed Step 4 eligibility gate
        hurst_window     : minimum low-vol bars required

        Returns
        -------
        DataFrame indexed by window with columns:
            n_low_vol_bars | hurst_H | hurst_pass | valid | skip_reason
        """
        rows = []

        for w, series in ndev_map.items():
            if w not in eligible_windows:
                rows.append({
                    "window":         w,
                    "n_low_vol_bars": None,
                    "hurst_H":        None,
                    "hurst_pass":     False,
                    "valid":          False,
                    "skip_reason":    "skipped — not eligible from Step 4",
                })
                continue

            row = HurstCalculator.compute_window(
                series, low_vol_mask, w, hurst_window
            )
            rows.append(row)

        return pd.DataFrame(rows).set_index("window")

    @staticmethod
    def flag_report(hurst_results: pd.DataFrame) -> str:
        """
        Human-readable Hurst exponent summary for console output.
        """
        lines = []
        for w, row in hurst_results.iterrows():
            if not row["valid"] and row["hurst_H"] is None:
                lines.append(
                    f"  W={w:>4}  [SKIP ]  {row['skip_reason']}"
                )
            elif row["hurst_pass"]:
                lines.append(
                    f"  W={w:>4}  [PASS ]  "
                    f"H={row['hurst_H']:.4f}  "
                    f"low_vol_bars={int(row['n_low_vol_bars'])}"
                )
            else:
                lines.append(
                    f"  W={w:>4}  [FAIL ]  "
                    f"H={row['hurst_H']:.4f}  "
                    f"({row['skip_reason']})"
                )
        return "Hurst Exponent Results:\n" + "\n".join(lines)