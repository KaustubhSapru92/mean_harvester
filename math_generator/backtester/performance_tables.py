import pandas as pd


class PerformanceTables:
    """
    Builds train-vs-test validation tables from comparison results.
    """

    @staticmethod
    def build_summary_table(oos_results: dict[int, dict]) -> pd.DataFrame:
        rows = []

        for w, res in oos_results.items():
            if res.get("skip_reason"):
                rows.append({
                    "window": w,
                    "train_sharpe": None,
                    "test_sharpe": None,
                    "skip_reason": res["skip_reason"],
                })
                continue

            train = res["is_metrics"]
            test = res["oos_metrics"]
            deg = res["degradation"]

            rows.append({
                "window": w,
                "train_sharpe": train["sharpe"],
                "test_sharpe": test["sharpe"],
                "train_pnl": train["total_pnl"],
                "test_pnl": test["total_pnl"],
                "is_sharpe": train["sharpe"],
                "oos_sharpe": test["sharpe"],
                "is_pnl": train["total_pnl"],
                "oos_pnl": test["total_pnl"],
                "train_max_drawdown": train["max_drawdown"],
                "test_max_drawdown": test["max_drawdown"],
                "train_trades": train["n_trades"],
                "test_trades": test["n_trades"],
                "sharpe_degradation": deg["sharpe_ratio"],
                "pnl_degradation": deg["pnl_ratio"],
                "sharpe_ratio": deg["sharpe_ratio"],
                "pnl_ratio": deg["pnl_ratio"],
                "overfit_flag": deg["overfit_flag"],
                "too_few_test_trades": deg["too_few_test_trades"],
                "pnl_flip": deg["pnl_flip"],
                "skip_reason": None,
            })

        return (
            pd.DataFrame(rows)
            .set_index("window")
            .sort_values("test_sharpe", ascending=False, na_position="last")
        )

    @staticmethod
    def build_degradation_table(oos_results: dict[int, dict]) -> pd.DataFrame:
        rows = []

        for w, res in oos_results.items():
            if res.get("skip_reason"):
                rows.append({
                    "window": w,
                    "sharpe_ratio": None,
                    "pnl_ratio": None,
                    "overfit_flag": None,
                    "quality_rating": "SKIP",
                })
                continue

            deg = res["degradation"]
            ratio = deg["sharpe_ratio"]

            if ratio is None:
                rating = "UNKNOWN"
            elif ratio >= 0.8:
                rating = "STRONG"
            elif ratio >= 0.5:
                rating = "ACCEPTABLE"
            elif ratio >= 0:
                rating = "WEAK"
            else:
                rating = "OVERFIT"

            if deg["overfit_flag"]:
                rating = "OVERFIT"

            rows.append({
                "window": w,
                "sharpe_ratio": ratio,
                "pnl_ratio": deg["pnl_ratio"],
                "overfit_flag": deg["overfit_flag"],
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
        oos_results: dict[int, dict],
    ) -> pd.DataFrame:
        rows = []

        for w, cost in cost_results.items():
            if cost.get("skip_reason"):
                rows.append({
                    "window": w,
                    "train_cost": None,
                    "test_cost": None,
                    "total_cost": None,
                    "test_net_pnl": None,
                    "cost_drag_pct": None,
                })
                continue

            train_cost = cost["train"]["total_cost_paid"]
            test_cost = cost["test"]["total_cost_paid"]
            total_cost = train_cost + test_cost

            oos_res = oos_results.get(w, {})
            if oos_res.get("skip_reason"):
                test_net_pnl = None
                cost_drag_pct = None
            else:
                test_net_pnl = oos_res["oos_metrics"]["total_pnl"]
                gross = test_net_pnl + test_cost
                cost_drag_pct = (
                    round(test_cost / abs(gross) * 100, 2)
                    if gross != 0 else None
                )

            rows.append({
                "window": w,
                "train_cost": round(train_cost, 6),
                "test_cost": round(test_cost, 6),
                "total_cost": round(total_cost, 6),
                "test_net_pnl": test_net_pnl,
                "cost_drag_pct": cost_drag_pct,
            })

        return (
            pd.DataFrame(rows)
            .set_index("window")
            .sort_values("test_net_pnl", ascending=False, na_position="last")
        )

    @staticmethod
    def flag_report(
        summary_table: pd.DataFrame,
        degradation_table: pd.DataFrame,
    ) -> str:
        lines = ["Performance Tables:"]

        lines.append("\n  Train vs Test Summary:")
        for w, row in summary_table.iterrows():
            if pd.isna(row.get("test_sharpe")):
                lines.append(f"    W={w:>4}  [SKIP]")
                continue
            flag = " *** OVERFIT ***" if row.get("overfit_flag") else ""
            lines.append(
                f"    W={w:>4}  "
                f"train({row['train_sharpe']:+.3f} / {row['train_pnl']:+.4f})  "
                f"test({row['test_sharpe']:+.3f} / {row['test_pnl']:+.4f})  "
                f"ratio={row['sharpe_degradation']}{flag}"
            )

        lines.append("\n  Degradation Ratings:")
        for w, row in degradation_table.iterrows():
            rating = row.get("quality_rating", "-")
            ratio = row.get("sharpe_ratio")
            ratio_str = f"{ratio:.4f}" if ratio is not None else "-"
            lines.append(f"    W={w:>4}  [{rating:<10}]  sharpe_ratio={ratio_str}")

        return "\n".join(lines)
