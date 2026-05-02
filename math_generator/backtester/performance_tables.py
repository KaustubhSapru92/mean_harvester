import pandas as pd


class PerformanceTables:
    """
    Step 7 — Stage 4.

    Builds structured comparison tables from IS and OOS metrics.

    Three tables produced:
        1. summary_table    — one row per window, IS vs OOS side by side
        2. degradation_table — degradation ratios with overfit flags
        3. cost_impact_table — gross vs net PnL showing cost drag

    The summary table is the primary deliverable for the
    "no curve-fitting" requirement — it makes IS/OOS comparison
    immediate and unambiguous.

    Degradation ratio interpretation:
        ratio >= 0.8  : strong — OOS performs close to IS
        ratio 0.5-0.8 : acceptable — some degradation, monitor closely
        ratio < 0.5   : overfit flag — IS performance likely not replicable
        ratio < 0     : strategy reverses sign OOS — severe overfit
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def build_summary_table(oos_results: dict[int, dict]) -> pd.DataFrame:
        """
        Build the main IS vs OOS comparison table.

        Columns:
            is_sharpe | is_pnl | is_win_rate | is_max_dd | is_trades
            oos_sharpe | oos_pnl | oos_win_rate | oos_max_dd | oos_trades
            sharpe_ratio | overfit_flag

        Indexed by window, sorted by OOS Sharpe descending.

        Parameters
        ----------
        oos_results : { window: oos_result_dict } from OOSEngine

        Returns
        -------
        pd.DataFrame
        """
        rows = []

        for w, res in oos_results.items():
            if res.get("skip_reason"):
                rows.append({
                    "window":        w,
                    "is_sharpe":     None,
                    "is_pnl":        None,
                    "is_win_rate":   None,
                    "is_max_dd":     None,
                    "is_trades":     None,
                    "oos_sharpe":    None,
                    "oos_pnl":       None,
                    "oos_win_rate":  None,
                    "oos_max_dd":    None,
                    "oos_trades":    None,
                    "sharpe_ratio":  None,
                    "overfit_flag":  None,
                    "skip_reason":   res["skip_reason"],
                })
                continue

            is_m  = res["is_metrics"]
            oos_m = res["oos_metrics"]
            deg   = res["degradation"]

            rows.append({
                "window":       w,
                "is_sharpe":    is_m["sharpe"],
                "is_pnl":       is_m["total_pnl"],
                "is_win_rate":  is_m["win_rate"],
                "is_max_dd":    is_m["max_drawdown"],
                "is_trades":    is_m["n_trades"],
                "oos_sharpe":   oos_m["sharpe"],
                "oos_pnl":      oos_m["total_pnl"],
                "oos_win_rate": oos_m["win_rate"],
                "oos_max_dd":   oos_m["max_drawdown"],
                "oos_trades":   oos_m["n_trades"],
                "sharpe_ratio": deg["sharpe_ratio"],
                "overfit_flag": deg["overfit_flag"],
                "skip_reason":  None,
            })

        df = pd.DataFrame(rows).set_index("window")

        # Sort by OOS Sharpe descending — best window first
        return df.sort_values("oos_sharpe", ascending=False, na_position="last")

    @staticmethod
    def build_degradation_table(
        oos_results: dict[int, dict]
    ) -> pd.DataFrame:
        """
        Build a focused degradation table with quality ratings.

        Columns:
            sharpe_ratio | pnl_ratio | overfit_flag | quality_rating

        Quality rating:
            STRONG     : sharpe_ratio >= 0.8
            ACCEPTABLE : sharpe_ratio 0.5 - 0.8
            WEAK       : sharpe_ratio 0.0 - 0.5
            OVERFIT    : sharpe_ratio < 0.0
            SKIP       : window was skipped

        Parameters
        ----------
        oos_results : { window: oos_result_dict } from OOSEngine

        Returns
        -------
        pd.DataFrame indexed by window
        """
        rows = []

        for w, res in oos_results.items():
            if res.get("skip_reason"):
                rows.append({
                    "window":         w,
                    "sharpe_ratio":   None,
                    "pnl_ratio":      None,
                    "overfit_flag":   None,
                    "quality_rating": "SKIP",
                })
                continue

            deg = res["degradation"]
            r   = deg["sharpe_ratio"]

            if r is None:
                rating = "SKIP"
            elif r >= 0.8:
                rating = "STRONG"
            elif r >= 0.5:
                rating = "ACCEPTABLE"
            elif r >= 0.0:
                rating = "WEAK"
            else:
                rating = "OVERFIT"

            rows.append({
                "window":         w,
                "sharpe_ratio":   r,
                "pnl_ratio":      deg["pnl_ratio"],
                "overfit_flag":   deg["overfit_flag"],
                "quality_rating": rating,
            })

        return (
            pd.DataFrame(rows)
            .set_index("window")
            .sort_values("sharpe_ratio", ascending=False, na_position="last")
        )

    @staticmethod
    def build_cost_impact_table(
        cost_results: dict[int, dict],
        oos_results:  dict[int, dict],
    ) -> pd.DataFrame:
        """
        Build a table showing gross vs net PnL and total cost drag.

        Columns:
            train_cost | test_cost | total_cost
            oos_gross_pnl | oos_net_pnl | cost_drag_pct

        cost_drag_pct = total_cost / abs(oos_gross_pnl) * 100
        Shows how much of the raw PnL is consumed by transaction costs.

        Parameters
        ----------
        cost_results : { window: { train, test } } from TransactionCostEngine
        oos_results  : { window: oos_result_dict } from OOSEngine

        Returns
        -------
        pd.DataFrame indexed by window
        """
        rows = []

        for w, cost in cost_results.items():
            if cost.get("skip_reason"):
                rows.append({
                    "window":        w,
                    "train_cost":    None,
                    "test_cost":     None,
                    "total_cost":    None,
                    "oos_net_pnl":   None,
                    "cost_drag_pct": None,
                })
                continue

            train_cost = cost["train"]["total_cost_paid"]
            test_cost  = cost["test"]["total_cost_paid"]
            total_cost = train_cost + test_cost

            oos_res = oos_results.get(w, {})
            if oos_res.get("skip_reason"):
                oos_net_pnl   = None
                cost_drag_pct = None
            else:
                oos_net_pnl = oos_res["oos_metrics"]["total_pnl"]
                gross        = oos_net_pnl + test_cost
                cost_drag_pct = (
                    round(test_cost / abs(gross) * 100, 2)
                    if gross != 0 else None
                )

            rows.append({
                "window":        w,
                "train_cost":    round(train_cost, 6),
                "test_cost":     round(test_cost,  6),
                "total_cost":    round(total_cost,  6),
                "oos_net_pnl":   oos_net_pnl,
                "cost_drag_pct": cost_drag_pct,
            })

        return (
            pd.DataFrame(rows)
            .set_index("window")
            .sort_values("oos_net_pnl", ascending=False, na_position="last")
        )

    @staticmethod
    def flag_report(
        summary_table:     pd.DataFrame,
        degradation_table: pd.DataFrame,
    ) -> str:
        """
        Human-readable table summary for console output.
        """
        lines = ["Performance Tables:"]

        lines.append("\n  IS vs OOS Summary (sorted by OOS Sharpe):")
        for w, row in summary_table.iterrows():
            if pd.isna(row.get("oos_sharpe")):
                lines.append(f"    W={w:>4}  [SKIP]")
                continue
            flag = " *** OVERFIT ***" if row.get("overfit_flag") else ""
            lines.append(
                f"    W={w:>4}  "
                f"IS({row['is_sharpe']:+.3f} / {row['is_pnl']:+.4f})  "
                f"OOS({row['oos_sharpe']:+.3f} / {row['oos_pnl']:+.4f})  "
                f"ratio={row['sharpe_ratio']}{flag}"
            )

        lines.append("\n  Degradation Ratings:")
        for w, row in degradation_table.iterrows():
            rating = row.get("quality_rating", "—")
            r      = row.get("sharpe_ratio")
            r_str  = f"{r:.4f}" if r is not None else "—"
            lines.append(f"    W={w:>4}  [{rating:<10}]  sharpe_ratio={r_str}")

        return "\n".join(lines)