import numpy as np
import pandas as pd


class OOSEngine:
    """
    Step 7 — Stage 3.

    Runs the out-of-sample backtest using parameters frozen on the
    train set. No re-optimisation is permitted on test data.

    Frozen parameters (derived from train, applied unchanged to test):
        kelly_capped     : position size fraction from PositionSizer
        entry_threshold  : VNDEV level to enter (from config — same as IS)
        stop_threshold   : VNDEV level to stop (from config — same as IS)
        half_life_bars   : timeout from window_scores (same as IS)

    The OOS run is a clean re-execution of the signal → position →
    PnL pipeline on the test VNDEV slice, using the frozen kelly_capped
    from the train Kelly computation.

    Anti-overfitting discipline:
        - Kelly fraction is re-computed on train trades only
        - Entry/stop thresholds are NOT swept — they remain fixed at
          the config values used in Step 6
        - half_life timeout is NOT changed between IS and OOS
        - The OOS result is what the strategy earns on unseen data
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_window(
        self,
        test_vndev: pd.Series,
        test_trades: list[dict],
        train_trades: list[dict],
        cost_test: dict,
        window_scores_row: pd.Series,
        sized_result: dict,
        base_vwap: pd.DataFrame,
        signal_gen,
        window: int,
    ) -> dict:
        """
        Run OOS evaluation for one window.

        Parameters
        ----------
        test_vndev        : VNDEV series for the test segment
        test_trades       : trade log sliced to test segment
        train_trades      : trade log sliced to train segment (for IS metrics)
        cost_test         : cost_dict for test segment from TransactionCostEngine
        window_scores_row : row from window_scores DataFrame for this window
        sized_result      : sized_dict for this window from PositionSizer
        base_vwap         : full OHLCV DataFrame
        signal_gen        : SignalGenerator instance (carries thresholds)
        window            : VWAP window integer

        Returns
        -------
        dict with keys:
            window, is_metrics, oos_metrics, degradation
        """
        kelly_capped   = sized_result["kelly_meta"]["kelly_capped"]
        half_life_bars = float(window_scores_row["half_life_bars"])

        # ---- Re-run signal on test VNDEV with frozen params ----
        oos_signal_result = signal_gen.generate(
            vndev          = test_vndev,
            half_life_bars = half_life_bars,
            window         = window,
        )

        oos_signal   = oos_signal_result["signal"]
        oos_position = oos_signal.astype(float) * kelly_capped
        oos_position.name = f"position_{window}"

        # ---- OOS PnL (on test price slice) ----
        test_close = base_vwap["Close"].reindex(test_vndev.index).ffill()

        bar_returns  = test_close.pct_change().fillna(0.0)
        pos_lagged   = oos_position.shift(1).fillna(0.0)
        raw_bar_pnl  = pos_lagged * bar_returns

        # Apply cost-adjusted PnL from Stage 2
        oos_bar_pnl  = cost_test["cost_adjusted_pnl"].reindex(
            raw_bar_pnl.index
        ).fillna(raw_bar_pnl)

        oos_equity   = oos_bar_pnl.cumsum()
        running_peak = oos_equity.cummax()
        oos_drawdown = oos_equity - running_peak

        # ---- IS metrics (from train cost-adjusted PnL) ----
        is_metrics = self._compute_metrics(
            bar_pnl   = cost_test.get("__train_pnl__",
                        pd.Series(dtype=float)),
            trade_log = train_trades,
            position  = sized_result["position"],
            label     = "IS",
        )

        # ---- OOS metrics ----
        oos_metrics = self._compute_metrics(
            bar_pnl   = oos_bar_pnl,
            trade_log = oos_signal_result["trade_log"],
            position  = oos_position,
            label     = "OOS",
        )

        # ---- Degradation ratio ----
        degradation = self._degradation(is_metrics, oos_metrics)

        return {
            "window":        window,
            "is_metrics":    is_metrics,
            "oos_metrics":   oos_metrics,
            "degradation":   degradation,
            "oos_bar_pnl":   oos_bar_pnl,
            "oos_equity":    oos_equity,
            "oos_drawdown":  oos_drawdown,
            "oos_signal":    oos_signal,
            "oos_position":  oos_position,
            "oos_trade_log": oos_signal_result["trade_log"],
        }

    def run_all(
        self,
        wf_splits: dict[int, dict],
        cost_results: dict[int, dict],
        sized_results: dict[int, dict],
        window_scores: pd.DataFrame,
        base_vwap: pd.DataFrame,
        signal_gen,
        pnl_results: dict[int, dict],
    ) -> dict[int, dict]:
        """
        Run OOS evaluation for all windows.

        Parameters
        ----------
        wf_splits     : { window: split_dict } from WalkForwardSplitter
        cost_results  : { window: { train, test } } from TransactionCostEngine
        sized_results : { window: sized_dict } from PositionSizer
        window_scores : enriched DataFrame from vwap_pipeline
        base_vwap     : OHLCV DataFrame
        signal_gen    : SignalGenerator instance
        pnl_results   : { window: pnl_dict } from PnLEngine

        Returns
        -------
        { window: oos_result_dict }
        """
        oos_results = {}

        for w, split in wf_splits.items():
            if split.get("skip_reason"):
                oos_results[w] = {"skip_reason": split["skip_reason"]}
                continue

            cost = cost_results.get(w, {})
            if cost.get("skip_reason"):
                oos_results[w] = {"skip_reason": cost["skip_reason"]}
                continue

            if w not in sized_results or w not in window_scores.index:
                oos_results[w] = {"skip_reason": "missing sizing or scores data"}
                continue

            # Attach train bar_pnl to cost_test dict for IS metric access
            cost_test = dict(cost["test"])
            train_pnl = cost["train"]["cost_adjusted_pnl"]
            cost_test["__train_pnl__"] = train_pnl

            oos_results[w] = self.run_window(
                test_vndev        = split["test_vndev"],
                test_trades       = split["test_trades"],
                train_trades      = split["train_trades"],
                cost_test         = cost_test,
                window_scores_row = window_scores.loc[w],
                sized_result      = sized_results[w],
                base_vwap         = base_vwap,
                signal_gen        = signal_gen,
                window            = w,
            )

        return oos_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        bar_pnl: pd.Series,
        trade_log: list[dict],
        position: pd.Series,
        label: str,
        bars_per_year: int = 19656,  # 78 bars/day * 252 days
    ) -> dict:
        """
        Compute Sharpe, PnL, win rate, max drawdown from a bar PnL series.
        Mirrors PerformanceMetrics.compute_window() on a segment slice.
        """
        if bar_pnl.empty or len(bar_pnl) < 2:
            return {
                "label":        label,
                "sharpe":       0.0,
                "total_pnl":    0.0,
                "max_drawdown": 0.0,
                "win_rate":     0.0,
                "n_trades":     0,
            }

        active = bar_pnl[bar_pnl != 0.0]
        if len(active) < 2 or active.std() == 0:
            sharpe = 0.0
        else:
            sharpe = float(
                active.mean() / active.std() * np.sqrt(bars_per_year)
            )

        equity       = bar_pnl.cumsum()
        running_peak = equity.cummax()
        max_dd       = float((equity - running_peak).min())

        # Trade-level win rate from trade_log
        trade_pnls = []
        for trade in trade_log:
            try:
                entry = trade["entry_bar"]
                exit_ = trade["exit_bar"]
                mask  = (bar_pnl.index >= entry) & (bar_pnl.index <= exit_)
                trade_pnls.append(float(bar_pnl[mask].sum()))
            except Exception:
                continue

        n_trades = len(trade_pnls)
        n_wins   = sum(1 for p in trade_pnls if p > 0)
        win_rate = round(n_wins / n_trades, 4) if n_trades > 0 else 0.0

        return {
            "label":        label,
            "sharpe":       round(sharpe, 4),
            "total_pnl":    round(float(equity.iloc[-1]), 6),
            "max_drawdown": round(max_dd, 6),
            "win_rate":     win_rate,
            "n_trades":     n_trades,
        }

    @staticmethod
    def _degradation(is_metrics: dict, oos_metrics: dict) -> dict:
        """
        Compute degradation ratios between IS and OOS metrics.

        sharpe_ratio  = oos_sharpe / is_sharpe  (1.0 = no degradation)
        pnl_ratio     = oos_pnl    / is_pnl
        flag          = True if sharpe_ratio < 0.5 (likely curve-fitted)
        """
        is_sharpe  = is_metrics["sharpe"]
        oos_sharpe = oos_metrics["sharpe"]
        is_pnl     = is_metrics["total_pnl"]
        oos_pnl    = oos_metrics["total_pnl"]

        sharpe_ratio = (
            round(oos_sharpe / is_sharpe, 4)
            if is_sharpe != 0 else None
        )
        pnl_ratio = (
            round(oos_pnl / is_pnl, 4)
            if is_pnl != 0 else None
        )

        overfit_flag = (
            sharpe_ratio is not None and sharpe_ratio < 0.5
        )

        return {
            "sharpe_ratio":  sharpe_ratio,
            "pnl_ratio":     pnl_ratio,
            "overfit_flag":  overfit_flag,
        }

    @staticmethod
    def flag_report(oos_results: dict[int, dict]) -> str:
        """
        Human-readable OOS summary for console output.
        """
        lines = ["Out-of-Sample Results:"]

        for w, res in oos_results.items():
            if res.get("skip_reason"):
                lines.append(f"  W={w:>4}  [SKIP]  {res['skip_reason']}")
                continue

            is_m   = res["is_metrics"]
            oos_m  = res["oos_metrics"]
            deg    = res["degradation"]
            flag   = " *** OVERFIT ***" if deg["overfit_flag"] else ""

            lines.append(
                f"  W={w:>4}  "
                f"IS  sharpe={is_m['sharpe']:+.3f}  pnl={is_m['total_pnl']:+.4f}  "
                f"win%={is_m['win_rate']:.1%}  "
                f"| OOS sharpe={oos_m['sharpe']:+.3f}  pnl={oos_m['total_pnl']:+.4f}  "
                f"win%={oos_m['win_rate']:.1%}  "
                f"ratio={deg['sharpe_ratio']}{flag}"
            )

        return "\n".join(lines)