import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


class Step6Viz:
    """
    Step 6 visualisation layer.

    Three plot types:
      1. equity_and_drawdown — equity curve + drawdown subplot per window
      2. vndev_signal_overlay — VNDEV series with entry/exit markers
      3. metrics_summary_table — cross-window ranked metrics table

    All methods return figure objects — no plt.show() calls.
    Consistent with Step3Viz, Step4Viz, Step5Viz patterns.
    """

    # ------------------------------------------------------------------
    # 1. Equity curve + drawdown
    # ------------------------------------------------------------------

    @staticmethod
    def equity_and_drawdown(
        equity_curves: pd.DataFrame,
        drawdown_curves: pd.DataFrame,
        cross_summary: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> plt.Figure:
        """
        One two-panel subplot per window (equity top, drawdown bottom).
        Windows sorted by Sharpe ratio descending.

        Parameters
        ----------
        equity_curves   : multi-window equity DataFrame from TradeBook
        drawdown_curves : multi-window drawdown DataFrame from TradeBook
        cross_summary   : ranked summary DataFrame from TradeBook
        symbol          : ticker string for title
        interval        : timeframe string for title

        Returns
        -------
        matplotlib Figure
        """
        windows = list(equity_curves.columns)
        if not windows:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No windows to plot.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return fig

        # Sort by Sharpe if available
        if not cross_summary.empty and "sharpe_ratio" in cross_summary.columns:
            windows = [
                w for w in cross_summary.sort_values(
                    "sharpe_ratio", ascending=False
                ).index
                if w in equity_curves.columns
            ]

        n_windows = len(windows)
        fig = plt.figure(figsize=(14, 5 * n_windows))
        gs  = gridspec.GridSpec(
            n_windows * 2, 1,
            hspace=0.45,
            height_ratios=[3, 1] * n_windows
        )

        for idx, w in enumerate(windows):
            equity   = equity_curves[w].dropna()
            drawdown = drawdown_curves[w].dropna()

            sharpe = (
                cross_summary.loc[w, "sharpe_ratio"]
                if w in cross_summary.index else float("nan")
            )
            total_pnl = float(equity.iloc[-1]) if len(equity) else 0.0
            max_dd    = float(drawdown.min())   if len(drawdown) else 0.0

            ax_eq = fig.add_subplot(gs[idx * 2])
            ax_dd = fig.add_subplot(gs[idx * 2 + 1], sharex=ax_eq)

            # ---- Equity panel ----
            color = "#4caf82" if total_pnl >= 0 else "#d9534f"
            ax_eq.plot(equity.index, equity.values, linewidth=1.0, color=color)
            ax_eq.axhline(0, linewidth=0.7, linestyle="--", color="gray", alpha=0.6)
            ax_eq.fill_between(
                equity.index, equity.values, 0,
                where=equity.values >= 0,
                alpha=0.12, color="#4caf82"
            )
            ax_eq.fill_between(
                equity.index, equity.values, 0,
                where=equity.values < 0,
                alpha=0.12, color="#d9534f"
            )
            ax_eq.set_title(
                f"W={w}  Sharpe={sharpe:+.3f}  "
                f"Total PnL={total_pnl:+.4f}  "
                f"Max DD={max_dd:.4f}",
                fontsize=9
            )
            ax_eq.set_ylabel("Cumulative PnL")
            ax_eq.grid(True, alpha=0.3)

            # ---- Drawdown panel ----
            ax_dd.fill_between(
                drawdown.index, drawdown.values, 0,
                alpha=0.4, color="#d9534f"
            )
            ax_dd.plot(
                drawdown.index, drawdown.values,
                linewidth=0.7, color="#d9534f"
            )
            ax_dd.set_ylabel("Drawdown")
            ax_dd.set_xlabel("Time")
            ax_dd.grid(True, alpha=0.3)

        fig.suptitle(
            f"{symbol} — Equity Curves & Drawdown ({interval})",
            fontsize=12, y=1.01
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 2. VNDEV signal overlay
    # ------------------------------------------------------------------

    @staticmethod
    def vndev_signal_overlay(
        vndev_map: dict[int, pd.Series],
        trade_log_df: pd.DataFrame,
        entry_threshold: float,
        stop_threshold: float,
        symbol: str,
        interval: str,
    ) -> plt.Figure:
        """
        VNDEV time series per window with entry/exit trade markers.

        Entry markers : triangle up (long) / triangle down (short)
        Exit markers  : circle — green (reversion), orange (timeout),
                        red (stop-loss)
        Threshold lines at ±entry and ±stop.

        Parameters
        ----------
        vndev_map       : { window: pd.Series } from VolNormalizer
        trade_log_df    : consolidated trade log from TradeBook
        entry_threshold : entry σ level (from config)
        stop_threshold  : stop σ level (from config)
        symbol          : ticker string for title
        interval        : timeframe string for title

        Returns
        -------
        matplotlib Figure
        """
        windows = list(vndev_map.keys())
        if not windows:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No windows to plot.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return fig

        n_cols = min(2, len(windows))
        n_rows = int(np.ceil(len(windows) / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(12 * n_cols, 4 * n_rows),
            squeeze=False
        )

        exit_colors = {
            "reversion":    "#4caf82",
            "timeout":      "#ff9800",
            "stop_loss":    "#d9534f",
            "nan_boundary": "#9e9e9e",
            "end_of_series": "#9e9e9e",
        }

        for idx, w in enumerate(windows):
            ax     = axes[idx // n_cols][idx % n_cols]
            vndev  = vndev_map[w].dropna()
            trades = (
                trade_log_df[trade_log_df["window"] == w]
                if not trade_log_df.empty else pd.DataFrame()
            )

            # VNDEV line
            ax.plot(
                vndev.index, vndev.values,
                linewidth=0.7, color="#5c6bc0", alpha=0.8, label="VNDEV"
            )

            # Threshold lines
            for lvl, ls, lbl in [
                ( entry_threshold, "--", f"+{entry_threshold}σ entry"),
                (-entry_threshold, "--", f"-{entry_threshold}σ entry"),
                ( stop_threshold,  ":",  f"+{stop_threshold}σ stop"),
                (-stop_threshold,  ":",  f"-{stop_threshold}σ stop"),
            ]:
                ax.axhline(
                    lvl, linestyle=ls, linewidth=0.8,
                    color="#e53935" if abs(lvl) == stop_threshold else "#ff9800",
                    alpha=0.7, label=lbl
                )

            # Zero line
            ax.axhline(0, linewidth=0.6, color="gray", alpha=0.5)

            # Trade markers
            if not trades.empty:
                for _, trade in trades.iterrows():
                    entry_ts = trade["entry_bar"]
                    exit_ts  = trade["exit_bar"]
                    direc    = trade["direction"]
                    reason   = trade["exit_reason"]

                    # Entry marker
                    try:
                        entry_v = float(vndev.asof(entry_ts))
                        marker  = "^" if direc == 1 else "v"
                        color   = "#4caf82" if direc == 1 else "#d9534f"
                        ax.scatter(
                            entry_ts, entry_v,
                            marker=marker, color=color,
                            s=40, zorder=5
                        )
                    except Exception:
                        pass

                    # Exit marker
                    try:
                        exit_v = float(vndev.asof(exit_ts))
                        ax.scatter(
                            exit_ts, exit_v,
                            marker="o",
                            color=exit_colors.get(reason, "#9e9e9e"),
                            s=30, zorder=5
                        )
                    except Exception:
                        pass

            ax.set_title(f"W={w}  VNDEV Signal", fontsize=9)
            ax.set_ylabel("VNDEV")
            ax.set_xlabel("Time")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(windows), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        # Legend
        legend_elements = [
            mpatches.Patch(color="#4caf82", label="Long entry (▲)"),
            mpatches.Patch(color="#d9534f", label="Short entry (▼)"),
            mpatches.Patch(color="#4caf82", label="Exit: reversion"),
            mpatches.Patch(color="#ff9800", label="Exit: timeout"),
            mpatches.Patch(color="#d9534f", label="Exit: stop-loss"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=5,
            fontsize=8,
            bbox_to_anchor=(0.5, -0.02)
        )

        fig.suptitle(
            f"{symbol} — VNDEV Signal Overlay ({interval})",
            fontsize=12, y=1.01
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 3. Metrics summary table
    # ------------------------------------------------------------------

    @staticmethod
    def metrics_summary_table(
        cross_summary: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> plt.Figure:
        """
        Renders the cross-window ranked metrics as a matplotlib table.
        Rows sorted by Sharpe ratio descending.
        Sharpe and total PnL cells colour-coded green/red.

        Parameters
        ----------
        cross_summary : output of TradeBook.cross_window_summary()
        symbol        : ticker string for title
        interval      : timeframe string for title

        Returns
        -------
        matplotlib Figure
        """
        if cross_summary.empty:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "No metrics to display.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return fig

        display_cols = [
            "sharpe_ratio", "total_pnl", "max_drawdown",
            "win_rate", "profit_factor", "n_trades",
            "avg_bars_held", "pct_long", "pct_stop",
        ]
        col_labels = [
            "Sharpe", "Total PnL", "Max DD",
            "Win%", "Prof. Factor", "Trades",
            "Avg Bars", "% Long", "% Stop",
        ]

        existing_cols = [c for c in display_cols if c in cross_summary.columns]
        existing_lbls = [
            col_labels[display_cols.index(c)] for c in existing_cols
        ]

        sorted_df = cross_summary.sort_values("sharpe_ratio", ascending=False)
        cell_text = []

        for w, row in sorted_df.iterrows():
            cells = []
            for c in existing_cols:
                val = row[c]
                if pd.isna(val):
                    cells.append("—")
                elif c in ("win_rate", "pct_long", "pct_stop"):
                    cells.append(f"{val:.1%}")
                elif c in ("n_trades", "avg_bars_held"):
                    cells.append(f"{val:.1f}")
                else:
                    sign = "+" if val > 0 else ""
                    cells.append(f"{sign}{val:.4f}")
            cell_text.append(cells)

        row_labels = [f"W={w}" for w in sorted_df.index]

        fig, ax = plt.subplots(
            figsize=(max(10, 1.3 * len(existing_cols)),
                     1.2 + 0.55 * len(sorted_df))
        )
        ax.axis("off")

        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=existing_lbls,
            loc="center",
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.6)

        # Colour Sharpe and Total PnL cells
        sharpe_col_idx = existing_cols.index("sharpe_ratio") if "sharpe_ratio" in existing_cols else None
        pnl_col_idx    = existing_cols.index("total_pnl")    if "total_pnl"    in existing_cols else None

        for row_idx, (_, row) in enumerate(sorted_df.iterrows(), start=1):
            if sharpe_col_idx is not None:
                v = row.get("sharpe_ratio", 0)
                table[row_idx, sharpe_col_idx].set_facecolor(
                    "#d4edda" if v > 0 else "#f8d7da"
                )
            if pnl_col_idx is not None:
                v = row.get("total_pnl", 0)
                table[row_idx, pnl_col_idx].set_facecolor(
                    "#d4edda" if v > 0 else "#f8d7da"
                )

        fig.suptitle(
            f"{symbol} — Backtester Results Summary ({interval})",
            fontsize=11, y=0.98
        )
        fig.tight_layout()
        return fig