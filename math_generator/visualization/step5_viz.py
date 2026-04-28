import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Step5Viz:
    """
    Step 5 visualisation layer.

    Three plots:
      1. atr_overlay       — ATR time series with threshold line + high-vol shading
      2. hurst_bar         — Hurst exponent per window with H=0.5 reference line
      3. vndev_distributions — VNDEV distribution per surviving window

    All methods return figure objects — no plt.show() calls.
    Consistent with Step3Viz and Step4Viz patterns.
    """

    # ------------------------------------------------------------------
    # 1. ATR overlay
    # ------------------------------------------------------------------

    @staticmethod
    def atr_overlay(
        base_vwap: pd.DataFrame,
        atr_df: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> plt.Figure:
        """
        Two-panel figure:
          top    — Close price with high-vol periods shaded red
          bottom — ATR series with threshold line

        Parameters
        ----------
        base_vwap : OHLCV DataFrame with Close column
        atr_df    : output of ATRCalculator.compute()
        symbol    : ticker string for title
        interval  : timeframe string for title

        Returns
        -------
        matplotlib Figure
        """
        fig = plt.figure(figsize=(14, 7))
        gs  = gridspec.GridSpec(2, 1, hspace=0.35, height_ratios=[2, 1])

        ax_price = fig.add_subplot(gs[0])
        ax_atr   = fig.add_subplot(gs[1], sharex=ax_price)

        # ---- Price panel ----
        ax_price.plot(
            base_vwap.index,
            base_vwap["Close"],
            linewidth=0.8,
            color="#2196f3",
            label="Close"
        )

        # Shade high-vol regions
        high_vol = atr_df["high_vol"]
        in_region = False
        start_idx = None

        for i, (ts, is_hv) in enumerate(high_vol.items()):
            if is_hv and not in_region:
                start_idx = ts
                in_region = True
            elif not is_hv and in_region:
                ax_price.axvspan(start_idx, ts, alpha=0.15, color="red", label="_nolegend_")
                in_region = False

        if in_region:
            ax_price.axvspan(start_idx, high_vol.index[-1], alpha=0.15, color="red")

        ax_price.set_title(
            f"{symbol} — Price with High-Vol Regime ({interval})\n"
            f"red shading = ATR above {int(atr_df['atr_threshold'].iloc[0] > 0 and 75)}th percentile",
            fontsize=10
        )
        ax_price.set_ylabel("Price")
        ax_price.legend(fontsize=8)
        ax_price.grid(True, alpha=0.3)

        # ---- ATR panel ----
        ax_atr.plot(
            atr_df.index,
            atr_df["atr"],
            linewidth=0.9,
            color="#ff7043",
            label="ATR"
        )
        ax_atr.axhline(
            atr_df["atr_threshold"].iloc[0],
            linestyle="--",
            linewidth=1.0,
            color="red",
            label=f"Threshold"
        )
        ax_atr.fill_between(
            atr_df.index,
            atr_df["atr"],
            atr_df["atr_threshold"].iloc[0],
            where=atr_df["atr"] > atr_df["atr_threshold"].iloc[0],
            alpha=0.2,
            color="red"
        )

        ax_atr.set_ylabel("ATR")
        ax_atr.set_xlabel("Time")
        ax_atr.legend(fontsize=8)
        ax_atr.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 2. Hurst exponent bar chart
    # ------------------------------------------------------------------

    @staticmethod
    def hurst_bar(
        hurst_results: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> plt.Figure:
        """
        Horizontal bar chart of Hurst exponent per window.
        H=0.5 reference line separates mean-reverting from random/trending.
        Bars coloured green (pass) / red (fail) / grey (skipped).

        Parameters
        ----------
        hurst_results : output of HurstCalculator.compute_all()
        symbol        : ticker string for title
        interval      : timeframe string for title

        Returns
        -------
        matplotlib Figure
        """
        windows = list(hurst_results.index)
        h_vals  = []
        colors  = []
        labels  = []

        for w, row in hurst_results.iterrows():
            labels.append(f"W={w}")
            if row["hurst_H"] is None or pd.isna(row["hurst_H"]):
                h_vals.append(0.0)
                colors.append("#bdbdbd")   # grey — skipped
            elif row["hurst_pass"]:
                h_vals.append(float(row["hurst_H"]))
                colors.append("#4caf82")   # green — pass
            else:
                h_vals.append(float(row["hurst_H"]))
                colors.append("#d9534f")   # red — fail

        fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(windows))))

        bars = ax.barh(
            labels[::-1],
            h_vals[::-1],
            color=colors[::-1],
            height=0.55
        )

        # H value annotations
        for bar, val, col in zip(bars, h_vals[::-1], colors[::-1]):
            if col != "#bdbdbd":
                ax.text(
                    bar.get_width() + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}",
                    va="center",
                    fontsize=9
                )
            else:
                ax.text(
                    0.01,
                    bar.get_y() + bar.get_height() / 2,
                    "skipped",
                    va="center",
                    fontsize=8,
                    color="#757575"
                )

        ax.axvline(
            0.5,
            linestyle="--",
            linewidth=1.0,
            color="black",
            label="H=0.5 (random walk)"
        )

        ax.set_xlim(0, max(0.8, max(h_vals) + 0.1) if h_vals else 0.8)
        ax.set_xlabel("Hurst Exponent (H)")
        ax.set_title(
            f"{symbol} — Hurst Exponent by Window ({interval})\n"
            f"green = mean-reverting (H<0.5)   red = not mean-reverting",
            fontsize=10
        )
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 3. VNDEV distributions
    # ------------------------------------------------------------------

    @staticmethod
    def vndev_distributions(
        vndev_map: dict[int, pd.Series],
        symbol: str,
        interval: str,
    ) -> plt.Figure:
        """
        One subplot per surviving window showing the VNDEV distribution
        on low-vol bars only. Overlays a standard normal reference.

        Parameters
        ----------
        vndev_map : { window: pd.Series } from VolNormalizer.compute_all()
        symbol    : ticker string for title
        interval  : timeframe string for title

        Returns
        -------
        matplotlib Figure
        """
        if not vndev_map:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(
                0.5, 0.5,
                "No windows passed regime filter.",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=11
            )
            ax.axis("off")
            fig.suptitle(f"{symbol} — VNDEV Distributions ({interval})")
            return fig

        windows = list(vndev_map.keys())
        n_cols  = min(3, len(windows))
        n_rows  = int(np.ceil(len(windows) / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 4 * n_rows),
            squeeze=False
        )

        for idx, w in enumerate(windows):
            ax     = axes[idx // n_cols][idx % n_cols]
            series = vndev_map[w].dropna()

            ax.hist(
                series,
                bins=50,
                density=True,
                alpha=0.6,
                color="#5c6bc0",
                label="VNDEV"
            )

            # Standard normal reference
            mu, sigma = series.mean(), series.std()
            x_range   = np.linspace(series.min(), series.max(), 200)
            normal_pdf = (
                1 / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
            )
            ax.plot(
                x_range, normal_pdf,
                linewidth=1.5,
                linestyle="--",
                color="black",
                label="Normal ref"
            )

            # Zero line
            ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")

            # Entry threshold guides at ±2σ
            ax.axvline( 2.0, color="#e53935", linewidth=0.7,
                        linestyle=":", label="±2σ")
            ax.axvline(-2.0, color="#e53935", linewidth=0.7, linestyle=":")

            ax.set_title(
                f"W={w}  μ={mu:.3f}  σ={sigma:.3f}  n={len(series)}",
                fontsize=9
            )
            ax.set_xlabel("VNDEV (vol-normalised)")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(windows), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        fig.suptitle(
            f"{symbol} — VNDEV Distributions ({interval})\n"
            f"low-vol bars only · dashed = ±2σ entry guides",
            fontsize=12,
            y=1.01
        )
        fig.tight_layout()
        return fig