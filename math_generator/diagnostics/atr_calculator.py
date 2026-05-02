import numpy as np
import pandas as pd


class ATRCalculator:
    """
    Computes Average True Range (ATR) and low-volatility masks.

    Train data may learn a percentile threshold with compute().
    Test data must use compute_with_threshold() so it does not learn from
    its own future volatility distribution.
    """

    @staticmethod
    def compute(
        df: pd.DataFrame,
        atr_window: int = 14,
        atr_percentile: float = 75.0,
    ) -> pd.DataFrame:
        out = ATRCalculator._compute_atr(df, atr_window)
        atr_values = out["atr"].dropna().values
        if len(atr_values) == 0:
            raise ValueError("ATRCalculator.compute: no valid ATR values.")

        threshold = float(np.nanpercentile(atr_values, atr_percentile))
        return ATRCalculator._apply_threshold(out, threshold)

    @staticmethod
    def compute_with_threshold(
        df: pd.DataFrame,
        atr_window: int,
        atr_threshold: float,
    ) -> pd.DataFrame:
        out = ATRCalculator._compute_atr(df, atr_window)
        return ATRCalculator._apply_threshold(out, atr_threshold)

    @staticmethod
    def low_vol_mask(atr_df: pd.DataFrame) -> pd.Series:
        return atr_df["low_vol"]

    @staticmethod
    def summary(atr_df: pd.DataFrame) -> dict:
        n_total = len(atr_df)
        n_low_vol = int(atr_df["low_vol"].sum())
        n_high_vol = n_total - n_low_vol

        return {
            "n_total": n_total,
            "n_low_vol": n_low_vol,
            "n_high_vol": n_high_vol,
            "low_vol_pct": round(n_low_vol / n_total * 100, 2),
            "atr_threshold": round(float(atr_df["atr_threshold"].iloc[0]), 6),
            "atr_mean": round(float(atr_df["atr"].mean()), 6),
            "atr_max": round(float(atr_df["atr"].max()), 6),
        }

    @staticmethod
    def flag_report(atr_df: pd.DataFrame) -> str:
        s = ATRCalculator.summary(atr_df)
        return (
            "ATR Regime Filter:\n"
            f"  threshold = {s['atr_threshold']:.6f}\n"
            f"  ATR mean = {s['atr_mean']:.6f}  "
            f"ATR max = {s['atr_max']:.6f}\n"
            f"  low-vol bars  : {s['n_low_vol']:>5} / {s['n_total']} "
            f"({s['low_vol_pct']:.1f}%)\n"
            f"  high-vol bars : {s['n_high_vol']:>5} / {s['n_total']} "
            f"({100 - s['low_vol_pct']:.1f}%)"
        )

    @staticmethod
    def _compute_atr(df: pd.DataFrame, atr_window: int) -> pd.DataFrame:
        if df.empty:
            raise ValueError("ATRCalculator.compute: input DataFrame is empty.")

        for col in ("High", "Low", "Close"):
            if col not in df.columns:
                raise ValueError(f"ATRCalculator.compute: missing column '{col}'.")

        out = pd.DataFrame(index=df.index)
        prev_close = df["Close"].shift(1)
        hl = df["High"] - df["Low"]
        h_pc = (df["High"] - prev_close).abs()
        l_pc = (df["Low"] - prev_close).abs()

        out["true_range"] = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(atr_window).mean()
        return out

    @staticmethod
    def _apply_threshold(out: pd.DataFrame, threshold: float) -> pd.DataFrame:
        out = out.copy()
        out["atr_threshold"] = float(threshold)
        out["high_vol"] = out["atr"] > threshold
        out["low_vol"] = ~out["high_vol"]
        out.loc[out["atr"].isna(), ["high_vol", "low_vol"]] = [True, False]
        return out
