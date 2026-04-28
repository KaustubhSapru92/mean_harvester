import pandas as pd


class VolNormalizer:
    """
    Step 5 — Stage 3.

    Produces volatility-normalised NDEV (VNDEV) for each eligible window
    by dividing NDEV_W by its own rolling standard deviation, applied
    only to bars that survive the ATR low-vol mask.

    VNDEV_W_t = NDEV_W_t / rolling_std(NDEV_W, vol_norm_window)_t

    Applied only where low_vol_mask == True.
    All masked (high-vol) bars receive NaN.

    Interpretation:
        VNDEV is a z-score-like signal — a value of +2.0 means the
        deviation is two rolling standard deviations above its recent
        mean, making entry thresholds directly comparable across windows
        and across different volatility regimes.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def compute_window(
        ndev_series: pd.Series,
        low_vol_mask: pd.Series,
        window: int,
        vol_norm_window: int = 20,
    ) -> pd.Series:
        """
        Compute VNDEV for a single NDEV_W series.

        Parameters
        ----------
        ndev_series     : full NDEV_W series (unmasked, from base_vwap)
        low_vol_mask    : boolean Series aligned to base_vwap index
        window          : VWAP window integer (for column naming only)
        vol_norm_window : rolling window for std computation (bars)

        Returns
        -------
        pd.Series named VNDEV_{window}, NaN on high-vol bars,
        indexed same as ndev_series.
        """
        # Rolling std computed on the full series first —
        # uses all available data so the denominator is stable
        rolling_std = ndev_series.rolling(vol_norm_window).std()

        # Raw VNDEV — full series
        vndev_raw = ndev_series / rolling_std

        # Align mask and apply — high-vol bars become NaN
        aligned_mask = low_vol_mask.reindex(ndev_series.index).fillna(False)
        vndev = vndev_raw.where(aligned_mask)

        vndev.name = f"VNDEV_{window}"
        return vndev

    @staticmethod
    def compute_all(
        ndev_map: dict[int, pd.Series],
        low_vol_mask: pd.Series,
        eligible_windows: list[int],
        hurst_results: pd.DataFrame,
        vol_norm_window: int = 20,
    ) -> dict[int, pd.Series]:
        """
        Compute VNDEV for all windows that are eligible AND passed Hurst.

        Windows that failed Hurst or were ineligible from Step 4 are
        excluded — their VNDEV would not be used in Step 6 anyway.

        Parameters
        ----------
        ndev_map         : { window: pd.Series } from DeviationNormalizer
        low_vol_mask     : boolean Series from ATRCalculator
        eligible_windows : windows that passed Step 4 eligibility gate
        hurst_results    : DataFrame from HurstCalculator.compute_all()
        vol_norm_window  : rolling window for std (bars)

        Returns
        -------
        { window: pd.Series of VNDEV } for windows passing all filters.
        Windows excluded at any prior stage are absent from the dict.
        """
        vndev_map = {}

        for w, series in ndev_map.items():

            # Must be eligible from Step 4
            if w not in eligible_windows:
                continue

            # Must have passed Hurst check
            if w not in hurst_results.index:
                continue
            if not bool(hurst_results.loc[w, "hurst_pass"]):
                continue

            vndev = VolNormalizer.compute_window(
                series, low_vol_mask, w, vol_norm_window
            )
            vndev_map[w] = vndev

        return vndev_map

    @staticmethod
    def summary(vndev_map: dict[int, pd.Series]) -> pd.DataFrame:
        """
        Quick sanity table per window:
            n_valid | mean | std | min | max

        n_valid = number of non-NaN bars (i.e. low-vol bars with
                  enough history for rolling std).
        """
        rows = []
        for w, series in vndev_map.items():
            clean = series.dropna()
            rows.append({
                "window":  w,
                "n_valid": len(clean),
                "mean":    round(clean.mean(), 6) if len(clean) else None,
                "std":     round(clean.std(),  6) if len(clean) else None,
                "min":     round(clean.min(),  6) if len(clean) else None,
                "max":     round(clean.max(),  6) if len(clean) else None,
            })
        return pd.DataFrame(rows).set_index("window")

    @staticmethod
    def flag_report(vndev_map: dict[int, pd.Series]) -> str:
        """
        Human-readable VNDEV summary for console output.
        """
        if not vndev_map:
            return "VNDEV: no windows passed all filters."

        summary = VolNormalizer.summary(vndev_map)
        lines   = ["VNDEV Summary (low-vol bars only):"]

        for w, row in summary.iterrows():
            lines.append(
                f"  W={w:>4}  n={int(row['n_valid']):>5}  "
                f"mean={row['mean']:+.4f}  "
                f"std={row['std']:.4f}  "
                f"[{row['min']:.3f}, {row['max']:.3f}]"
            )
        return "\n".join(lines)