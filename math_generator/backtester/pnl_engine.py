import numpy as np
import pandas as pd


class PnLEngine:
    """
    Step 6 — Stage 3.

    Computes vectorised bar-by-bar PnL from scaled position series
    and Close-to-Close bar returns.

    Core formula (no lookahead):
        bar_return_t = Close_t / Close_(t-1) - 1
        trade_pnl_t  = position_(t-1) * bar_return_t

    Using the LAGGED position is critical — the position decided at bar
    t-1 earns the return that occurs between t-1 and t. Using position_t
    would imply knowledge of the signal at the same bar as the return,
    which is lookahead.

    Cumulative PnL:
        equity_t = sum(trade_pnl_0 ... trade_pnl_t)

    Outputs per window:
        bar_pnl    : pd.Series — PnL earned each bar
        equity     : pd.Series — cumulative PnL (equity curve)
        drawdown   : pd.Series — rolling drawdown from equity peak
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def compute_window(
        position: pd.Series,
        base_vwap: pd.DataFrame,
        window: int,
    ) -> dict:
        """
        Compute PnL series for a single window.

        Parameters
        ----------
        position  : scaled position Series from PositionSizer (float, lagged
                    internally — caller passes the raw series)
        base_vwap : OHLCV DataFrame with Close column
        window    : VWAP window integer (for labelling only)

        Returns
        -------
        dict with keys:
            bar_pnl  : pd.Series — per-bar PnL
            equity   : pd.Series — cumulative PnL
            drawdown : pd.Series — drawdown from running peak (negative values)
        """
        # Align position to base_vwap index — fill gaps with zero (flat)
        pos_aligned = position.reindex(base_vwap.index).fillna(0.0)

        # Bar returns — Close to Close, no lookahead
        bar_returns = base_vwap["Close"].pct_change().fillna(0.0)

        # Lagged position — position decided at t-1 earns return at t
        pos_lagged = pos_aligned.shift(1).fillna(0.0)

        # Bar PnL
        bar_pnl = pos_lagged * bar_returns
        bar_pnl.name = f"bar_pnl_{window}"

        # Equity curve — cumulative sum
        equity = bar_pnl.cumsum()
        equity.name = f"equity_{window}"

        # Drawdown — rolling peak minus current value
        running_peak = equity.cummax()
        drawdown     = equity - running_peak
        drawdown.name = f"drawdown_{window}"

        return {
            "bar_pnl":  bar_pnl,
            "equity":   equity,
            "drawdown": drawdown,
        }

    @staticmethod
    def compute_all(
        sized_results: dict[int, dict],
        base_vwap: pd.DataFrame,
    ) -> dict[int, dict]:
        """
        Compute PnL for all windows in sized_results.

        Parameters
        ----------
        sized_results : { window: { signal, trade_log, position, kelly_meta } }
                        from PositionSizer.size_all()
        base_vwap     : OHLCV DataFrame

        Returns
        -------
        { window: { signal, trade_log, position, kelly_meta,
                    bar_pnl, equity, drawdown } }
        """
        pnl_results = {}

        for w, res in sized_results.items():
            pnl = PnLEngine.compute_window(
                position  = res["position"],
                base_vwap = base_vwap,
                window    = w,
            )

            pnl_results[w] = {
                **res,
                "bar_pnl":  pnl["bar_pnl"],
                "equity":   pnl["equity"],
                "drawdown": pnl["drawdown"],
            }

        return pnl_results

    @staticmethod
    def summary(pnl_results: dict[int, dict]) -> pd.DataFrame:
        """
        Quick PnL summary table across all windows.

        Columns:
            total_pnl | mean_bar_pnl | std_bar_pnl | max_drawdown | n_bars_traded
        """
        rows = []

        for w, res in pnl_results.items():
            bar_pnl  = res["bar_pnl"]
            equity   = res["equity"]
            drawdown = res["drawdown"]

            active = bar_pnl[bar_pnl != 0.0]

            rows.append({
                "window":         w,
                "total_pnl":      round(float(equity.iloc[-1]), 6),
                "mean_bar_pnl":   round(float(active.mean()), 6) if len(active) else 0.0,
                "std_bar_pnl":    round(float(active.std()),  6) if len(active) else 0.0,
                "max_drawdown":   round(float(drawdown.min()), 6),
                "n_bars_traded":  int((res["position"] != 0).sum()),
            })

        return pd.DataFrame(rows).set_index("window")

    @staticmethod
    def flag_report(pnl_results: dict[int, dict]) -> str:
        """
        Human-readable PnL summary for console output.
        """
        if not pnl_results:
            return "PnL Engine: no windows to report."

        summary = PnLEngine.summary(pnl_results)
        lines   = ["Vectorised PnL Summary:"]

        for w, row in summary.iterrows():
            pnl_sign = "+" if row["total_pnl"] >= 0 else ""
            lines.append(
                f"  W={w:>4}  "
                f"total={pnl_sign}{row['total_pnl']:.4f}  "
                f"mean_bar={row['mean_bar_pnl']:+.6f}  "
                f"std={row['std_bar_pnl']:.6f}  "
                f"max_dd={row['max_drawdown']:.4f}  "
                f"bars_traded={int(row['n_bars_traded'])}"
            )

        return "\n".join(lines)