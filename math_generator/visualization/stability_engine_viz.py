import matplotlib.pyplot as plt
import pandas as pd


class StabilityEngineViz:
    """
    Engine-grade visualization layer for stability diagnostics.
    Returns figure objects instead of showing plots.
    """

    # -------------------------------------------------
    # Rolling Stability Line Plot
    # -------------------------------------------------
    @staticmethod
    def rolling_stability_line(
        df: pd.DataFrame,
        symbol: str,
        interval: str
    ):
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111)

        for col in df.columns:
            if col.startswith("RollingStability_"):
                ax.plot(df.index, df[col], label=col)

        ax.set_title(f"{symbol} Rolling VWAP Stability ({interval})")
        ax.set_ylabel("Stability Score")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        return fig

    # -------------------------------------------------
    # Rolling Stability Heatmap (Window vs Time)
    # -------------------------------------------------
    @staticmethod
    def rolling_stability_heatmap(
        df: pd.DataFrame,
        symbol: str,
        interval: str
    ):
        stability_cols = [c for c in df.columns if c.startswith("RollingStability_")]
        heat_df = df[stability_cols].copy()

        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111)

        im = ax.imshow(
            heat_df.T.values,
            aspect="auto",
            interpolation="nearest"
        )

        ax.set_yticks(range(len(stability_cols)))
        ax.set_yticklabels(stability_cols)
        ax.set_title(f"{symbol} Rolling Stability Heatmap ({interval})")

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        return fig

    # -------------------------------------------------
    # Stability Regime Detection Overlay
    # -------------------------------------------------
    @staticmethod
    def stability_regime_overlay(
        price_df: pd.DataFrame,
        stability_df: pd.DataFrame,
        symbol: str,
        interval: str,
        threshold: float = 0.8
    ):
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111)

        ax.plot(price_df.index, price_df["Close"], label="Close")

        regime_series = stability_df.mean(axis=1)
        unstable = regime_series < threshold

        ax.scatter(
            price_df.index[unstable],
            price_df["Close"][unstable],
            marker="x",
            label="Unstable Regime"
        )

        ax.set_title(f"{symbol} Stability Regime Overlay ({interval})")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        return fig
