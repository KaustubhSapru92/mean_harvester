import numpy as np
import pandas as pd


class ATRCalculator:
    """
    Step 5 — Stage 1.

    Computes Average True Range (ATR) on the base_vwap OHLCV DataFrame
    and produces a low-volatility boolean mask for use in Stages 2–4.

    True Range at bar t:
        TR_t = max(High_t - Low_t,
                   |High_t - Close_(t-1)|,
                   |Low_t  - Close_(t-1)|)

    ATR_t = rolling mean of TR over atr_window bars.

    A bar is flagged high-volatility when:
        ATR_t > percentile(ATR, atr_percentile)

    The percentile threshold is computed over the full available ATR
    history (not rolling) so the mask is stable and reproducible.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def compute(df: pd.DataFrame,
                atr_window: int = 14,
                atr_percentile: float = 75.0,
                ) -> pd.DataFrame:
        """
        Compute ATR and low-vol mask on a full OHLCV DataFrame.

        Parameters
        ----------
        df             : base_vwap DataFrame with High, Low, Close columns
        atr_window     : rolling window for ATR mean (bars)
        atr_percentile : percentile above which a bar is high-vol (0–100)

        Returns
        -------
        DataFrame with same index as df and columns:
            true_range    : raw TR per bar
            atr           : rolling mean of TR (NaN for first atr_window bars)
            atr_threshold : scalar threshold broadcast as a constant series
            high_vol      : bool — True if bar is high-volatility
            low_vol       : bool — True if bar is safe to trade (inverse of high_vol)
        """
        if df.empty:
            raise ValueError("ATRCalculator.compute: input DataFrame is empty.")

        for col in ("High", "Low", "Close"):
            if col not in df.columns:
                raise ValueError(f"ATRCalculator.compute: missing column '{col}'.")

        out = pd.DataFrame(index=df.index)

        # True range — three-component max
        prev_close   = df["Close"].shift(1)
        hl           = df["High"] - df["Low"]
        h_pc         = (df["High"] - prev_close).abs()
        l_pc         = (df["Low"]  - prev_close).abs()

        out["true_range"] = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)

        # ATR — simple rolling mean (Wilder uses EWM; simple mean is standard here)
        out["atr"] = out["true_range"].rolling(atr_window).mean()

        # Threshold — fixed percentile over full ATR history (dropna)
        threshold = float(np.nanpercentile(out["atr"].dropna().values, atr_percentile))
        out["atr_threshold"] = threshold

        # Regime flags
        out["high_vol"] = out["atr"] > threshold
        out["low_vol"]  = ~out["high_vol"]

        # First atr_window bars have NaN ATR — treat conservatively as high-vol
        out.loc[out["atr"].isna(), ["high_vol", "low_vol"]] = [True, False]

        return out

    @staticmethod
    def low_vol_mask(atr_df: pd.DataFrame) -> pd.Series:
        """
        Convenience method — returns just the low_vol boolean Series.
        Aligned to atr_df.index, ready to apply to base_vwap.

        Parameters
        ----------
        atr_df : output of ATRCalculator.compute()

        Returns
        -------
        pd.Series of bool, indexed same as atr_df
        """
        return atr_df["low_vol"]

    @staticmethod
    def summary(atr_df: pd.DataFrame) -> dict:
        """
        Quick summary stats on the ATR regime filter result.

        Returns
        -------
        dict with keys:
            n_total, n_low_vol, n_high_vol,
            low_vol_pct, atr_threshold, atr_mean, atr_max
        """
        n_total   = len(atr_df)
        n_low_vol = int(atr_df["low_vol"].sum())
        n_high_vol = n_total - n_low_vol

        return {
            "n_total":      n_total,
            "n_low_vol":    n_low_vol,
            "n_high_vol":   n_high_vol,
            "low_vol_pct":  round(n_low_vol / n_total * 100, 2),
            "atr_threshold": round(float(atr_df["atr_threshold"].iloc[0]), 6),
            "atr_mean":     round(float(atr_df["atr"].mean()), 6),
            "atr_max":      round(float(atr_df["atr"].max()), 6),
        }

    @staticmethod
    def flag_report(atr_df: pd.DataFrame) -> str:
        """
        Human-readable ATR regime summary for console output.
        """
        s = ATRCalculator.summary(atr_df)
        return (
            f"ATR Regime Filter:\n"
            f"  threshold (p{int(atr_df['atr_threshold'].iloc[0] > 0 and 75)}) "
            f"= {s['atr_threshold']:.6f}\n"
            f"  ATR mean  = {s['atr_mean']:.6f}  "
            f"ATR max = {s['atr_max']:.6f}\n"
            f"  low-vol bars  : {s['n_low_vol']:>5} / {s['n_total']} "
            f"({s['low_vol_pct']:.1f}%)\n"
            f"  high-vol bars : {s['n_high_vol']:>5} / {s['n_total']} "
            f"({100 - s['low_vol_pct']:.1f}%)"
        )