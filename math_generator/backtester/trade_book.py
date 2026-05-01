import pandas as pd


class TradeBook:
    """
    Step 6 — Stage 5.

    Assembles the final trade log and equity curve outputs from the
    per-window results accumulated across Stages 1-4.

    Responsibilities:
        1. Flatten per-window trade logs into a single consolidated
           DataFrame with PnL attached per trade.
        2. Build a multi-window equity curve DataFrame — one column
           per window — aligned to base_vwap timestamps.
        3. Build a multi-window drawdown DataFrame — same structure.
        4. Produce a cross-window summary suitable for Step 7 input.

    The consolidated trade log is the primary deliverable — it includes
    every trade across all windows with full metadata so Step 7 can
    slice by window, direction, exit reason, or any other dimension.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def build_trade_log(
        pnl_results: dict[int, dict],
        metrics_map: dict[int, dict],
    ) -> pd.DataFrame:
        """
        Flatten all per-window trade logs into one consolidated DataFrame.
        Attaches per-trade PnL derived from bar_pnl slices.

        Parameters
        ----------
        pnl_results : { window: pnl_result_dict } from PnLEngine
        metrics_map : { window: metrics_dict }   from PerformanceMetrics

        Returns
        -------
        DataFrame with columns:
            window | direction | entry_bar | exit_bar | entry_vndev |
            exit_vndev | bars_held | exit_reason | trade_pnl |
            trade_pnl_pct | cumulative_pnl_window
        Sorted by entry_bar ascending.
        """
        all_rows = []

        for w, res in pnl_results.items():
            trade_log = res["trade_log"]
            bar_pnl   = res["bar_pnl"]
            equity    = res["equity"]

            running_pnl = 0.0

            for trade in trade_log:
                entry = trade["entry_bar"]
                exit_ = trade["exit_bar"]

                # Slice bar_pnl to get accurate per-trade PnL
                mask      = (bar_pnl.index >= entry) & (bar_pnl.index <= exit_)
                trade_pnl = float(bar_pnl[mask].sum())
                running_pnl += trade_pnl

                # Equity at exit bar
                try:
                    equity_at_exit = float(equity.loc[exit_])
                except KeyError:
                    equity_at_exit = running_pnl

                all_rows.append({
                    "window":               w,
                    "direction":            trade["direction"],
                    "entry_bar":            entry,
                    "exit_bar":             exit_,
                    "entry_vndev":          trade["entry_vndev"],
                    "exit_vndev":           trade["exit_vndev"],
                    "bars_held":            trade["bars_held"],
                    "exit_reason":          trade["exit_reason"],
                    "trade_pnl":            round(trade_pnl, 6),
                    "cumulative_pnl_window": round(equity_at_exit, 6),
                })

        if not all_rows:
            return pd.DataFrame()

        df = (
            pd.DataFrame(all_rows)
            .sort_values("entry_bar")
            .reset_index(drop=True)
        )

        # Add win flag and trade number per window
        df["is_win"] = df["trade_pnl"] > 0
        df["trade_n"] = df.groupby("window").cumcount() + 1

        return df

    @staticmethod
    def build_equity_curves(
        pnl_results: dict[int, dict],
        base_vwap: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build a multi-window equity curve DataFrame.

        Each column is one window's cumulative PnL, aligned to
        base_vwap's full timestamp index. NaN before first signal.

        Parameters
        ----------
        pnl_results : { window: pnl_result_dict }
        base_vwap   : OHLCV DataFrame (provides the master index)

        Returns
        -------
        DataFrame indexed by timestamp, columns = window integers.
        """
        curves = {}

        for w, res in pnl_results.items():
            equity = res["equity"].reindex(base_vwap.index)
            curves[w] = equity

        return pd.DataFrame(curves)

    @staticmethod
    def build_drawdown_curves(
        pnl_results: dict[int, dict],
        base_vwap: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build a multi-window drawdown DataFrame.

        Same structure as equity curves — one column per window.
        Values are always ≤ 0.

        Parameters
        ----------
        pnl_results : { window: pnl_result_dict }
        base_vwap   : OHLCV DataFrame (provides the master index)

        Returns
        -------
        DataFrame indexed by timestamp, columns = window integers.
        """
        curves = {}

        for w, res in pnl_results.items():
            dd = res["drawdown"].reindex(base_vwap.index)
            curves[w] = dd

        return pd.DataFrame(curves)

    @staticmethod
    def cross_window_summary(
        trade_log_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build a cross-window summary table merging trade log stats
        with performance metrics.

        Adds columns derived from the trade log that aren't in metrics_df:
            avg_bars_held    — mean trade duration
            pct_long         — fraction of trades that were long
            pct_stop         — fraction of trades stopped out

        Returns
        -------
        DataFrame indexed by window, all columns from metrics_df plus
        the three derived columns above. Sorted by sharpe_ratio desc.
        """
        if trade_log_df.empty or metrics_df.empty:
            return metrics_df.copy()

        derived = (
            trade_log_df
            .groupby("window")
            .apply(lambda g: pd.Series({
                "avg_bars_held": round(g["bars_held"].mean(), 2),
                "pct_long":      round((g["direction"] == 1).mean(), 4),
                "pct_stop":      round(
                    (g["exit_reason"] == "stop_loss").mean(), 4
                ),
            }))
        )

        merged = metrics_df.join(derived, how="left")
        return merged.sort_values("sharpe_ratio", ascending=False)

    @staticmethod
    def flag_report(
        trade_log_df: pd.DataFrame,
        equity_curves: pd.DataFrame,
    ) -> str:
        """
        Human-readable trade book summary for console output.
        """
        if trade_log_df.empty:
            return "TradeBook: no trades to report."

        lines = ["Trade Book Summary:"]

        # Overall stats
        n_total  = len(trade_log_df)
        n_wins   = int(trade_log_df["is_win"].sum())
        n_windows = trade_log_df["window"].nunique()

        lines.append(
            f"  total trades : {n_total} across {n_windows} window(s)"
        )
        lines.append(
            f"  overall win% : {n_wins / n_total:.1%}  "
            f"({n_wins} wins / {n_total - n_wins} losses)"
        )

        # Exit reason breakdown
        exit_counts = trade_log_df["exit_reason"].value_counts()
        lines.append("  exit reasons :")
        for reason, count in exit_counts.items():
            lines.append(f"    {reason:<18} {count:>4}  ({count/n_total:.1%})")

        # Per-window final equity
        lines.append("  final equity per window:")
        for w in sorted(equity_curves.columns):
            final = equity_curves[w].dropna()
            if len(final):
                val  = float(final.iloc[-1])
                sign = "+" if val >= 0 else ""
                lines.append(f"    W={w:>4}  {sign}{val:.4f}")

        return "\n".join(lines)