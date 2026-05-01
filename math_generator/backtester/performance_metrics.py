import numpy as np
import pandas as pd


class PerformanceMetrics:
    """
    Step 6 — Stage 4.

    Computes risk-adjusted performance metrics from the PnL series
    and trade log produced in Stages 2 and 3.

    Metrics computed per window:
        sharpe_ratio   — annualised Sharpe on bar PnL series
        max_drawdown   — largest peak-to-trough decline in equity
        win_rate       — fraction of completed trades with positive PnL
        avg_win        — mean PnL of winning trades
        avg_loss       — mean PnL of losing trades (negative value)
        profit_factor  — abs(sum wins) / abs(sum losses)
        total_pnl      — final equity curve value
        n_trades       — total completed trades
        n_bars_traded  — bars with non-zero position

    Sharpe annualisation:
        bars_per_year = bars_per_day * trading_days_per_year
        sharpe = mean(bar_pnl) / std(bar_pnl) * sqrt(bars_per_year)

        Only bars with non-zero position are used in mean/std so flat
        periods don't dilute the ratio.
    """

    TRADING_DAYS_PER_YEAR = 252

    def __init__(self, bars_per_day: int = 78):
        """
        Parameters
        ----------
        bars_per_day : number of bars in a trading session.
                       Default 78 = 6.5 hour session at 5-minute bars.
        """
        self.bars_per_day  = bars_per_day
        self.bars_per_year = bars_per_day * self.TRADING_DAYS_PER_YEAR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_window(
        self,
        pnl_result: dict,
        window: int,
    ) -> dict:
        """
        Compute full metrics for a single window.

        Parameters
        ----------
        pnl_result : single window dict from PnLEngine.compute_all()
        window     : VWAP window integer (for labelling only)

        Returns
        -------
        dict of scalar metrics
        """
        bar_pnl   = pnl_result["bar_pnl"]
        equity    = pnl_result["equity"]
        drawdown  = pnl_result["drawdown"]
        trade_log = pnl_result["trade_log"]
        position  = pnl_result["position"]

        # --- Sharpe ratio ---
        active_pnl = bar_pnl[position.reindex(bar_pnl.index).fillna(0) != 0]

        if len(active_pnl) < 2 or active_pnl.std() == 0:
            sharpe = 0.0
        else:
            sharpe = (
                float(active_pnl.mean())
                / float(active_pnl.std())
                * np.sqrt(self.bars_per_year)
            )

        # --- Drawdown ---
        max_dd = float(drawdown.min())

        # --- Trade-level metrics from PnL engine bar returns ---
        # Re-derive trade PnLs from bar_pnl slices for accuracy
        trade_pnls = self._trade_pnls_from_bar(bar_pnl, trade_log)

        n_trades = len(trade_pnls)
        wins     = [p for p in trade_pnls if p > 0]
        losses   = [p for p in trade_pnls if p <= 0]
        n_wins   = len(wins)
        n_losses = len(losses)

        win_rate = round(n_wins / n_trades, 4) if n_trades > 0 else 0.0

        avg_win  = round(float(np.mean(wins)),   6) if wins   else 0.0
        avg_loss = round(float(np.mean(losses)), 6) if losses else 0.0

        sum_wins   = sum(wins)
        sum_losses = abs(sum(losses))
        profit_factor = (
            round(sum_wins / sum_losses, 4)
            if sum_losses > 0
            else float("inf")
        )

        # Exit reason breakdown
        exit_counts = {}
        for trade in trade_log:
            reason = trade.get("exit_reason", "unknown")
            exit_counts[reason] = exit_counts.get(reason, 0) + 1

        return {
            "window":         window,
            "sharpe_ratio":   round(float(sharpe), 4),
            "max_drawdown":   round(max_dd, 6),
            "total_pnl":      round(float(equity.iloc[-1]), 6),
            "win_rate":       win_rate,
            "avg_win":        avg_win,
            "avg_loss":       avg_loss,
            "profit_factor":  profit_factor,
            "n_trades":       n_trades,
            "n_wins":         n_wins,
            "n_losses":       n_losses,
            "n_bars_traded":  int((position != 0).sum()),
            "exit_reversion": exit_counts.get("reversion",    0),
            "exit_timeout":   exit_counts.get("timeout",      0),
            "exit_stop":      exit_counts.get("stop_loss",    0),
            "exit_other":     exit_counts.get("nan_boundary", 0)
                              + exit_counts.get("end_of_series", 0),
        }

    def compute_all(
        self,
        pnl_results: dict[int, dict],
    ) -> tuple[pd.DataFrame, dict[int, dict]]:
        """
        Compute metrics for all windows.

        Parameters
        ----------
        pnl_results : { window: pnl_result_dict } from PnLEngine

        Returns
        -------
        metrics_df  : DataFrame indexed by window — one row per window
        metrics_map : { window: metrics_dict } — for downstream access
        """
        metrics_map = {}
        rows        = []

        for w, res in pnl_results.items():
            m = self.compute_window(res, w)
            metrics_map[w] = m
            rows.append(m)

        if not rows:
            return pd.DataFrame(), {}

        metrics_df = pd.DataFrame(rows).set_index("window")
        return metrics_df, metrics_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trade_pnls_from_bar(
        bar_pnl: pd.Series,
        trade_log: list[dict],
    ) -> list[float]:
        """
        Slice bar_pnl by each trade's entry/exit timestamps to get
        accurate per-trade PnL from the actual vectorised computation.
        """
        trade_pnls = []
        for trade in trade_log:
            try:
                entry = trade["entry_bar"]
                exit_ = trade["exit_bar"]
                mask  = (bar_pnl.index >= entry) & (bar_pnl.index <= exit_)
                pnl   = float(bar_pnl[mask].sum())
                trade_pnls.append(pnl)
            except Exception:
                continue
        return trade_pnls

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def flag_report(metrics_df: pd.DataFrame) -> str:
        """
        Human-readable metrics summary for console output.
        Sorted by Sharpe ratio descending.
        """
        if metrics_df.empty:
            return "Performance Metrics: no results."

        sorted_df = metrics_df.sort_values("sharpe_ratio", ascending=False)
        lines     = ["Performance Metrics (sorted by Sharpe):"]

        for w, row in sorted_df.iterrows():
            pnl_sign = "+" if row["total_pnl"] >= 0 else ""
            lines.append(
                f"  W={w:>4}  "
                f"sharpe={row['sharpe_ratio']:+.3f}  "
                f"pnl={pnl_sign}{row['total_pnl']:.4f}  "
                f"max_dd={row['max_drawdown']:.4f}  "
                f"win%={row['win_rate']:.1%}  "
                f"pf={row['profit_factor']:.2f}  "
                f"trades={int(row['n_trades'])}"
            )

        return "\n".join(lines)

    @staticmethod
    def metrics_table(metrics_df: pd.DataFrame) -> str:
        """
        Compact aligned table of all metrics for console review.
        """
        if metrics_df.empty:
            return "No metrics available."

        cols = [
            "sharpe_ratio", "total_pnl", "max_drawdown",
            "win_rate", "profit_factor", "n_trades",
            "exit_reversion", "exit_timeout", "exit_stop"
        ]
        return metrics_df[cols].to_string(float_format=lambda x: f"{x:.4f}")