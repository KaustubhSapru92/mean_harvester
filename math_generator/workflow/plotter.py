import matplotlib.pyplot as plt
import pandas as pd


class OHLCVPlotter:
    """
    Diagnostic plotting utilities.
    """

    @staticmethod
    def plot_price_vs_vwap(
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        windows=(20, 50)
    ):
        plt.figure(figsize=(14, 6))

        plt.plot(df.index, df["Close"], label="Close", linewidth=1)

        for w in windows:
            col = f"VWAP_{w}"
            if col in df.columns:
                plt.plot(df.index, df[col], label=col, linewidth=1.5)

        plt.title(f"{symbol} – Price vs VWAP ({interval})")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_vwap_deviation(
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        window=20
    ):
        col = f"DEV_{window}"
        if col not in df.columns:
            return

        plt.figure(figsize=(14, 4))
        plt.plot(df.index, df[col])
        plt.axhline(0, linestyle="--", alpha=0.5)

        plt.title(f"{symbol} – VWAP Deviation ({interval}, W={window})")
        plt.xlabel("Time")
        plt.ylabel("Price − VWAP")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_vwap_deviation_distribution(
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        window=20
    ):
        col = f"DEV_{window}"
        if col not in df.columns:
            return

        plt.figure(figsize=(8, 4))
        plt.hist(df[col].dropna(), bins=60, density=True)

        plt.title(f"{symbol} – VWAP Deviation Distribution ({interval}, W={window})")
        plt.xlabel("Deviation")
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_resampled_vs_base(
        base_df: pd.DataFrame,
        resampled_df: pd.DataFrame,
        symbol: str,
        base_interval: str,
        resampled_interval: str
    ):
        plt.figure(figsize=(14, 6))

        plt.plot(
            base_df.index,
            base_df["Close"],
            label=f"Close ({base_interval})",
            alpha=0.4
        )

        plt.plot(
            resampled_df.index,
            resampled_df["Close"],
            label=f"Close ({resampled_interval})",
            linewidth=2
        )

        plt.title(
            f"{symbol} – {base_interval} vs {resampled_interval}"
        )
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_history_stability(history_stats: pd.DataFrame, window: int):
        """
        Plot convergence of DEV statistics vs history length.
        """

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(
            history_stats.index,
            history_stats["mean"],
            marker="o",
            label="Mean DEV"
        )
        ax1.set_xlabel("History (days)")
        ax1.set_ylabel("Mean DEV")
        ax1.axhline(0, linestyle="--", alpha=0.5)
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(
            history_stats.index,
            history_stats["std"],
            marker="s",
            linestyle="--",
            label="Std DEV"
        )
        ax2.set_ylabel("Std DEV")

        fig.suptitle(f"VWAP Stability vs History Length (W={window})")

        fig.legend(loc="upper right")
        plt.show()

    @staticmethod
    def plot_window_stability(window_stats: pd.DataFrame):
        """
        Plot DEV dispersion vs VWAP window size.
        """

        plt.figure(figsize=(8, 4))
        plt.plot(
            window_stats.index,
            window_stats["std"],
            marker="o"
        )

        plt.xlabel("VWAP Window (bars)")
        plt.ylabel("Std DEV")
        plt.title("VWAP Deviation Stability vs Window Size")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_rolling_vwap_stability(
            stability_df: pd.DataFrame,
            window: int,
            roll_len: int
    ):
        """
        Visualizes rolling VWAP stability metrics.
        """

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # -------------------------
        # Rolling deviation stats
        # -------------------------
        axes[0].plot(
            stability_df.index,
            stability_df["dev_mean"],
            label="Rolling Mean Deviation",
            linewidth=1
        )
        axes[0].plot(
            stability_df.index,
            stability_df["dev_std"],
            label="Rolling Std Deviation",
            linewidth=1
        )
        axes[0].set_title(
            f"VWAP({window}) Rolling Deviation Stats (L={roll_len})"
        )
        axes[0].legend()
        axes[0].grid(True)

        # -------------------------
        # Instability of stability
        # -------------------------
        axes[1].plot(
            stability_df.index,
            stability_df["mean_instability"],
            label="Mean Instability",
            linewidth=1
        )
        axes[1].plot(
            stability_df.index,
            stability_df["std_instability"],
            label="Std Instability",
            linewidth=1
        )
        axes[1].set_title("Stability Drift Metrics")
        axes[1].legend()
        axes[1].grid(True)

        # -------------------------
        # Coefficient of variation
        # -------------------------
        axes[2].plot(
            stability_df.index,
            stability_df["cv"],
            label="Coefficient of Variation",
            linewidth=1
        )
        axes[2].set_title("Scale-Normalized Noise (CV)")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_vwap_window_comparison(
            stability_map: dict,
            roll_len: int
    ):
        """
        Compare rolling VWAP stability across multiple windows.
        stability_map: {window: stability_df}
        """

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # -------------------------
        # Mean instability
        # -------------------------
        for w, df in stability_map.items():
            axes[0].plot(
                df.index,
                df["mean_instability"],
                label=f"W={w}",
                linewidth=1
            )

        axes[0].set_title(
            f"VWAP Mean Instability Comparison (rolling L={roll_len})"
        )
        axes[0].legend()
        axes[0].grid(True)

        # -------------------------
        # Std instability
        # -------------------------
        for w, df in stability_map.items():
            axes[1].plot(
                df.index,
                df["std_instability"],
                label=f"W={w}",
                linewidth=1
            )

        axes[1].set_title("VWAP Std Instability Comparison")
        axes[1].legend()
        axes[1].grid(True)

        # -------------------------
        # Coefficient of variation
        # -------------------------
        for w, df in stability_map.items():
            axes[2].plot(
                df.index,
                df["cv"],
                label=f"W={w}",
                linewidth=1
            )

        axes[2].set_title("VWAP CV Comparison")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()