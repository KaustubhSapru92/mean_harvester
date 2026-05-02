import numpy as np
import pandas as pd


class OOSEngine:
    """
    Compares train and test artifacts after the test run has already been
    executed with frozen train parameters.
    """

    @staticmethod
    def compare_window(
        window: int,
        train_result: dict,
        test_result: dict,
        frozen_params,
        min_test_trades: int = 5,
    ) -> dict:
        train_metrics = OOSEngine._compute_metrics(
            bar_pnl=train_result["cost_adjusted_pnl"],
            trade_log=train_result["trade_log"],
            label="IS",
        )
        test_metrics = OOSEngine._compute_metrics(
            bar_pnl=test_result["cost_adjusted_pnl"],
            trade_log=test_result["trade_log"],
            label="OOS",
        )
        degradation = OOSEngine._degradation(
            train_metrics,
            test_metrics,
            min_test_trades=min_test_trades,
        )

        return {
            "window": window,
            "frozen_params": frozen_params,
            "is_metrics": train_metrics,
            "oos_metrics": test_metrics,
            "degradation": degradation,
            "is_bar_pnl": train_result["cost_adjusted_pnl"],
            "is_equity": train_result["cost_adjusted_equity"],
            "is_drawdown": train_result["cost_drawdown"],
            "oos_bar_pnl": test_result["cost_adjusted_pnl"],
            "oos_equity": test_result["cost_adjusted_equity"],
            "oos_drawdown": test_result["cost_drawdown"],
            "oos_signal": test_result["signal"],
            "oos_position": test_result["position"],
            "oos_trade_log": test_result["trade_log"],
        }

    @staticmethod
    def run_all(*args, **kwargs) -> dict:
        raise RuntimeError(
            "OOSEngine.run_all was removed from the validation path. "
            "Run train learning first, apply frozen parameters to test, then "
            "call OOSEngine.compare_window()."
        )

    @staticmethod
    def _compute_metrics(
        bar_pnl: pd.Series,
        trade_log: list[dict],
        label: str,
        bars_per_year: int = 19656,
    ) -> dict:
        if bar_pnl.empty or len(bar_pnl) < 2:
            return {
                "label": label,
                "sharpe": 0.0,
                "total_pnl": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "n_trades": 0,
            }

        active = bar_pnl[bar_pnl != 0.0]
        if len(active) < 2 or active.std() == 0:
            sharpe = 0.0
        else:
            sharpe = float(active.mean() / active.std() * np.sqrt(bars_per_year))

        equity = bar_pnl.cumsum()
        running_peak = equity.cummax()
        max_dd = float((equity - running_peak).min())

        trade_pnls = []
        for trade in trade_log:
            try:
                entry = trade["entry_bar"]
                exit_ = trade["exit_bar"]
                mask = (bar_pnl.index >= entry) & (bar_pnl.index <= exit_)
                trade_pnls.append(float(bar_pnl[mask].sum()))
            except Exception:
                continue

        n_trades = len(trade_pnls)
        n_wins = sum(1 for pnl in trade_pnls if pnl > 0)
        win_rate = round(n_wins / n_trades, 4) if n_trades > 0 else 0.0

        return {
            "label": label,
            "sharpe": round(float(sharpe), 4),
            "total_pnl": round(float(equity.iloc[-1]), 6),
            "max_drawdown": round(max_dd, 6),
            "win_rate": win_rate,
            "n_trades": n_trades,
        }

    @staticmethod
    def _degradation(
        train_metrics: dict,
        test_metrics: dict,
        min_test_trades: int,
    ) -> dict:
        train_sharpe = train_metrics["sharpe"]
        test_sharpe = test_metrics["sharpe"]
        train_pnl = train_metrics["total_pnl"]
        test_pnl = test_metrics["total_pnl"]
        test_trades = test_metrics["n_trades"]

        sharpe_ratio = (
            round(test_sharpe / train_sharpe, 4)
            if train_sharpe != 0 else None
        )
        pnl_ratio = round(test_pnl / train_pnl, 4) if train_pnl != 0 else None

        weak_sharpe = sharpe_ratio is not None and sharpe_ratio < 0.5
        too_few_trades = test_trades < min_test_trades
        pnl_flip = train_pnl > 0 and test_pnl < 0

        return {
            "sharpe_ratio": sharpe_ratio,
            "pnl_ratio": pnl_ratio,
            "overfit_flag": weak_sharpe or too_few_trades or pnl_flip,
            "too_few_test_trades": too_few_trades,
            "pnl_flip": pnl_flip,
        }

    @staticmethod
    def flag_report(oos_results: dict[int, dict]) -> str:
        lines = ["Out-of-Sample Results:"]

        for w, res in oos_results.items():
            if res.get("skip_reason"):
                lines.append(f"  W={w:>4}  [SKIP]  {res['skip_reason']}")
                continue

            is_m = res["is_metrics"]
            oos_m = res["oos_metrics"]
            deg = res["degradation"]
            flag = " *** OVERFIT ***" if deg["overfit_flag"] else ""

            lines.append(
                f"  W={w:>4}  "
                f"IS sharpe={is_m['sharpe']:+.3f} pnl={is_m['total_pnl']:+.4f} "
                f"trades={is_m['n_trades']} | "
                f"OOS sharpe={oos_m['sharpe']:+.3f} pnl={oos_m['total_pnl']:+.4f} "
                f"trades={oos_m['n_trades']} ratio={deg['sharpe_ratio']}{flag}"
            )

        return "\n".join(lines)
