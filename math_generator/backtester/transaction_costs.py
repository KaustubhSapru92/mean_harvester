import pandas as pd


class TransactionCostEngine:
    """
    Step 7 — Stage 2.

    Applies round-trip transaction costs to PnL series and trade logs
    for both train and test segments.

    Cost model:
        round_trip_cost = 0.1% of position value per trade
        entry_cost      = round_trip_cost / 2  (0.05% on open)
        exit_cost       = round_trip_cost / 2  (0.05% on close)

    Applied as a direct deduction from bar_pnl at the entry and exit
    bar of each trade. This keeps the PnL series properly indexed and
    avoids any lookahead — the cost is incurred at the bar it happens.

    The cost is proportional to kelly_capped (position size) so larger
    positions pay proportionally more in costs — consistent with
    real-world execution where cost scales with notional value.

    Outputs:
        cost_adjusted_pnl : pd.Series — bar PnL with costs deducted
        cost_adjusted_equity : pd.Series — cumulative cost-adjusted PnL
        total_cost_paid   : float — total transaction costs across all trades
        n_trades_costed   : int   — number of trades cost was applied to
    """

    DEFAULT_ROUND_TRIP_PCT = 0.001   # 0.1%

    def __init__(self, round_trip_pct: float = 0.001):
        """
        Parameters
        ----------
        round_trip_pct : total round-trip cost as a fraction (default 0.001 = 0.1%)
        """
        if round_trip_pct < 0:
            raise ValueError(
                f"round_trip_pct must be >= 0, got {round_trip_pct}"
            )
        self.round_trip_pct = round_trip_pct
        self.half_cost      = round_trip_pct / 2.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_to_segment(
        self,
        bar_pnl: pd.Series,
        trade_log: list[dict],
        kelly_capped: float,
        window: int,
    ) -> dict:
        """
        Apply transaction costs to one segment (train or test).

        Cost is deducted at the entry bar (half) and exit bar (half)
        of each trade. If entry and exit fall on the same bar, the
        full round-trip cost is deducted at that bar.

        Parameters
        ----------
        bar_pnl      : per-bar PnL series from PnLEngine for this segment
        trade_log    : list of trade dicts whose entry/exit bars fall
                       within this segment
        kelly_capped : Kelly fraction — cost scales with position size
        window       : VWAP window integer (for labelling only)

        Returns
        -------
        dict with keys:
            cost_adjusted_pnl    : pd.Series
            cost_adjusted_equity : pd.Series
            cost_drawdown        : pd.Series
            total_cost_paid      : float
            n_trades_costed      : int
        """
        cost_series = pd.Series(0.0, index=bar_pnl.index)
        n_costed    = 0

        for trade in trade_log:
            entry_bar = trade["entry_bar"]
            exit_bar  = trade["exit_bar"]

            # Entry cost — deducted at entry bar if it falls in this segment
            if entry_bar in cost_series.index:
                cost_series.loc[entry_bar] += self.half_cost * kelly_capped
                n_costed += 1

            # Exit cost — deducted at exit bar if it falls in this segment
            if exit_bar in cost_series.index:
                cost_series.loc[exit_bar] += self.half_cost * kelly_capped

        # Deduct costs from bar PnL
        adj_pnl    = bar_pnl - cost_series
        adj_equity = adj_pnl.cumsum()

        # Drawdown on cost-adjusted equity
        running_peak = adj_equity.cummax()
        adj_drawdown = adj_equity - running_peak

        total_cost = float(cost_series.sum())

        return {
            "cost_adjusted_pnl":    adj_pnl,
            "cost_adjusted_equity": adj_equity,
            "cost_drawdown":        adj_drawdown,
            "total_cost_paid":      round(total_cost, 8),
            "n_trades_costed":      n_costed,
        }

    def apply_to_splits(
        self,
        wf_splits: dict[int, dict],
        pnl_results: dict[int, dict],
        sized_results: dict[int, dict],
        base_vwap: pd.DataFrame,
    ) -> dict[int, dict]:
        """
        Apply transaction costs to both train and test segments
        for all windows.

        Parameters
        ----------
        wf_splits     : { window: split_dict } from WalkForwardSplitter
        pnl_results   : { window: pnl_result_dict } from PnLEngine
        sized_results : { window: sized_dict } from PositionSizer
                        (provides kelly_capped per window)
        base_vwap     : OHLCV DataFrame for bar return computation

        Returns
        -------
        { window: {
            train: cost_segment_dict,
            test:  cost_segment_dict,
        } }
        """
        cost_results = {}

        for w, split in wf_splits.items():
            if split.get("skip_reason"):
                cost_results[w] = {"skip_reason": split["skip_reason"]}
                continue

            if w not in pnl_results or w not in sized_results:
                cost_results[w] = {"skip_reason": "no PnL or sizing data"}
                continue

            full_bar_pnl  = pnl_results[w]["bar_pnl"]
            kelly_capped  = sized_results[w]["kelly_meta"]["kelly_capped"]

            cut_bar       = split["cut_bar"]

            # Slice bar_pnl into train and test segments
            train_pnl = full_bar_pnl[full_bar_pnl.index <  cut_bar]
            test_pnl  = full_bar_pnl[full_bar_pnl.index >= cut_bar]

            train_segment = self.apply_to_segment(
                bar_pnl    = train_pnl,
                trade_log  = split["train_trades"],
                kelly_capped = kelly_capped,
                window     = w,
            )

            test_segment = self.apply_to_segment(
                bar_pnl    = test_pnl,
                trade_log  = split["test_trades"],
                kelly_capped = kelly_capped,
                window     = w,
            )

            cost_results[w] = {
                "train": train_segment,
                "test":  test_segment,
            }

        return cost_results

    @staticmethod
    def flag_report(cost_results: dict[int, dict]) -> str:
        """
        Human-readable transaction cost summary for console output.
        """
        lines = ["Transaction Costs (0.1% round-trip):"]

        for w, res in cost_results.items():
            if res.get("skip_reason"):
                lines.append(f"  W={w:>4}  [SKIP]  {res['skip_reason']}")
                continue

            train = res["train"]
            test  = res["test"]

            lines.append(
                f"  W={w:>4}  "
                f"train_cost={train['total_cost_paid']:.6f}  "
                f"({train['n_trades_costed']} trades)  "
                f"test_cost={test['total_cost_paid']:.6f}  "
                f"({test['n_trades_costed']} trades)"
            )

        return "\n".join(lines)