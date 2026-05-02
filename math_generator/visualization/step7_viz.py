import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Step7Viz:
    """
    Step 7 visualisation layer.

    Three plot types:
      1. is_oos_equity_overlay  — IS and OOS equity curves on one axis per window
      2. rolling_sharpe         — rolling Sharpe over the OOS period
      3. sensitivity_sweep      — Sharpe vs entry threshold sweep

    All methods return figure objects — no plt.show() calls.
    Consistent with Step3Viz through Step6Viz patterns.
    """

    # ------------------------------------------------------------------
    # 1. IS vs OOS equity curve overlay
    # ------------------------------------------------------------------

    @staticmethod
    def is_oos_equity_overlay(
        oos_results: dict[int, dict],
        cost_results: dict[int, dict],
        summary_table: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> plt.Figure:
        """
        One panel per window showing IS equity curve and OOS equity
        curve on the same axis. Visual similarity = no curve-fitting.

        The IS curve is plotted on a normalised x-axis [0, 1] alongside
        the OOS curve so their shapes can be compared directly regardless
        of different bar counts.

        Parameters
        ----------
        oos_results   : { window: oos_result_dict } from OOSEngine
        cost_results  : { window: cost_dict } from TransactionCostEngine
        summary_table : IS vs OOS comparison DataFrame from PerformanceTables
        symbol        : ticker string for title
        interval      : timeframe string for title

        Returns
        -------
        matplotlib Figure
        """
        windows = [
            w for w, res in oos_results.items()
            if not res.get("skip_reason")
        ]

        if not windows:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No OOS results to plot.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return fig

        n_cols = min(2, len(windows))
        n_rows = int(np.ceil(len(windows) / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(10 * n_cols, 4 * n_rows),
            squeeze=False
        )

        for idx, w in enumerate(windows):
            ax  = axes[idx // n_cols][idx % n_cols]
            res = oos_results[w]

            # IS equity from train cost-adjusted PnL
            is_equity  = cost_results[w]["train"]["cost_adjusted_equity"]
            oos_equity = res["oos_equity"]

            # Normalise x-axis to [0, 1] for shape comparison
            is_x  = np.linspace(0, 1, len(is_equity))
            oos_x = np.linspace(0, 1, len(oos_equity))

            is_color  = "#5c6bc0"
            oos_color = "#4caf82" if float(oos_equity.iloc[-1]) >= 0 else "#d9534f"

            ax.plot(is_x,  is_equity.values,  linewidth=1.0,
                    color=is_color,  label="IS  (train)", alpha=0.8)
            ax.plot(oos_x, oos_equity.values, linewidth=1.2,
                    color=oos_color, label="OOS (test)")

            ax.axhline(0, linewidth=0.6, linestyle="--", color="gray", alpha=0.5)
            ax.fill_between(oos_x, oos_equity.values, 0,
                            where=oos_equity.values >= 0,
                            alpha=0.08, color="#4caf82")
            ax.fill_between(oos_x, oos_equity.values, 0,
                            where=oos_equity.values < 0,
                            alpha=0.08, color="#d9534f")

            # Degradation ratio in title
            ratio = (
                summary_table.loc[w, "sharpe_ratio"]
                if w in summary_table.index else float("nan")
            )
            overfit = (
                summary_table.loc[w, "overfit_flag"]
                if w in summary_table.index else False
            )
            flag_str = "  *** OVERFIT ***" if overfit else ""

            ax.set_title(
                f"W={w}  IS→OOS Sharpe ratio={ratio:.3f}{flag_str}",
                fontsize=9,
                color="#d9534f" if overfit else "black"
            )
            ax.set_xlabel("Normalised time [0=start, 1=end]", fontsize=8)
            ax.set_ylabel("Cumulative PnL", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        for idx in range(len(windows), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        fig.suptitle(
            f"{symbol} — IS vs OOS Equity Overlay ({interval})\n"
            f"blue = in-sample · green/red = out-of-sample",
            fontsize=11, y=1.01
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 2. Rolling Sharpe over OOS period
    # ------------------------------------------------------------------

    @staticmethod
    def rolling_sharpe(
        oos_results: dict[int, dict],
        symbol: str,
        interval: str,
        roll_window: int = 50,
        bars_per_year: int = 19656,
    ) -> plt.Figure:
        """
        Rolling Sharpe ratio computed over the OOS bar PnL series.
        Answers: does the strategy stay profitable throughout the test
        period, or does it degrade over time?

        A flat or rising rolling Sharpe is the target.
        A declining rolling Sharpe signals regime change or overfitting.

        Parameters
        ----------
        oos_results   : { window: oos_result_dict } from OOSEngine
        symbol        : ticker string for title
        interval      : timeframe string for title
        roll_window   : rolling window in bars for Sharpe computation
        bars_per_year : annualisation constant

        Returns
        -------
        matplotlib Figure
        """
        windows = [
            w for w, res in oos_results.items()
            if not res.get("skip_reason")
        ]

        if not windows:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No OOS results to plot.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return fig

        n_cols = min(2, len(windows))
        n_rows = int(np.ceil(len(windows) / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(10 * n_cols, 4 * n_rows),
            squeeze=False
        )

        for idx, w in enumerate(windows):
            ax      = axes[idx // n_cols][idx % n_cols]
            bar_pnl = oos_results[w]["oos_bar_pnl"]

            if len(bar_pnl) < roll_window + 2:
                ax.text(0.5, 0.5, f"W={w}: insufficient OOS bars for rolling Sharpe",
                        ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.axis("off")
                continue

            roll_mean = bar_pnl.rolling(roll_window).mean()
            roll_std  = bar_pnl.rolling(roll_window).std()
            roll_sharpe = (roll_mean / roll_std * np.sqrt(bars_per_year)).dropna()

            color = "#5c6bc0"
            ax.plot(
                range(len(roll_sharpe)),
                roll_sharpe.values,
                linewidth=1.0, color=color
            )
            ax.axhline(0, linewidth=0.8, linestyle="--", color="gray", alpha=0.5)
            ax.fill_between(
                range(len(roll_sharpe)),
                roll_sharpe.values, 0,
                where=roll_sharpe.values >= 0,
                alpha=0.12, color="#4caf82"
            )
            ax.fill_between(
                range(len(roll_sharpe)),
                roll_sharpe.values, 0,
                where=roll_sharpe.values < 0,
                alpha=0.12, color="#d9534f"
            )

            oos_sharpe = oos_results[w]["oos_metrics"]["sharpe"]
            ax.set_title(
                f"W={w}  OOS Sharpe={oos_sharpe:+.3f}  "
                f"(rolling window={roll_window} bars)",
                fontsize=9
            )
            ax.set_xlabel("OOS bar index", fontsize=8)
            ax.set_ylabel("Rolling Sharpe", fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(len(windows), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        fig.suptitle(
            f"{symbol} — Rolling OOS Sharpe ({interval})",
            fontsize=11, y=1.01
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 3. Parameter sensitivity sweep
    # ------------------------------------------------------------------

    @staticmethod
    def sensitivity_sweep(
        oos_results: dict[int, dict],
        wf_splits: dict[int, dict],
        sized_results: dict[int, dict],
        window_scores: pd.DataFrame,
        base_vwap: pd.DataFrame,
        signal_gen,
        symbol: str,
        interval: str,
        threshold_range: tuple[float, float] = (1.0, 3.5),
        n_steps: int = 11,
    ) -> plt.Figure:
        """
        Sharpe vs entry_threshold sweep on the OOS segment.
        Shows whether the strategy is robust to small parameter changes
        or balanced on a knife-edge.

        A smooth, wide hill = robust.
        A sharp spike = overfitted to the exact threshold.

        Parameters
        ----------
        oos_results     : { window: oos_result_dict } from OOSEngine
        wf_splits       : { window: split_dict } from WalkForwardSplitter
        sized_results   : { window: sized_dict } from PositionSizer
        window_scores   : enriched DataFrame from vwap_pipeline
        base_vwap       : OHLCV DataFrame
        signal_gen      : SignalGenerator instance
        symbol          : ticker string for title
        interval        : timeframe string for title
        threshold_range : (min, max) entry threshold to sweep
        n_steps         : number of threshold values to evaluate

        Returns
        -------
        matplotlib Figure
        """
        from backtester.signal_generator import SignalGenerator

        windows = [
            w for w, res in oos_results.items()
            if not res.get("skip_reason")
        ]

        if not windows:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No OOS results to plot.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return fig

        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)

        n_cols = min(2, len(windows))
        n_rows = int(np.ceil(len(windows) / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(9 * n_cols, 4 * n_rows),
            squeeze=False
        )

        bar_returns = base_vwap["Close"].pct_change().fillna(0.0)

        for idx, w in enumerate(windows):
            ax     = axes[idx // n_cols][idx % n_cols]
            split  = wf_splits.get(w, {})

            if split.get("skip_reason"):
                ax.text(0.5, 0.5, "split unavailable",
                        ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.axis("off")
                continue

            test_vndev     = split["test_vndev"]
            kelly_capped   = sized_results[w]["kelly_meta"]["kelly_capped"]
            half_life_bars = float(window_scores.loc[w, "half_life_bars"])
            base_threshold = signal_gen.entry_threshold

            sharpes = []

            for thresh in thresholds:
                # Build a temporary signal generator with swept threshold
                tmp_gen = SignalGenerator(
                    entry_threshold = thresh,
                    stop_threshold  = signal_gen.stop_threshold,
                )
                result   = tmp_gen.generate(test_vndev, half_life_bars, w)
                position = result["signal"].astype(float) * kelly_capped

                test_returns = bar_returns.reindex(test_vndev.index).fillna(0.0)
                pos_lagged   = position.shift(1).fillna(0.0)
                bar_pnl      = pos_lagged * test_returns

                active = bar_pnl[bar_pnl != 0.0]
                if len(active) < 2 or active.std() == 0:
                    sharpes.append(0.0)
                else:
                    s = float(active.mean() / active.std() * np.sqrt(19656))
                    sharpes.append(round(s, 4))

            # Plot
            ax.plot(thresholds, sharpes, linewidth=1.2, color="#5c6bc0", marker="o", markersize=3)
            ax.axvline(
                base_threshold, linestyle="--", linewidth=0.9,
                color="#ff9800", label=f"current={base_threshold}σ"
            )
            ax.axhline(0, linewidth=0.6, color="gray", alpha=0.5, linestyle="--")

            ax.fill_between(
                thresholds, sharpes, 0,
                where=np.array(sharpes) >= 0,
                alpha=0.1, color="#4caf82"
            )
            ax.fill_between(
                thresholds, sharpes, 0,
                where=np.array(sharpes) < 0,
                alpha=0.1, color="#d9534f"
            )

            ax.set_title(f"W={w}  OOS Sharpe vs entry threshold", fontsize=9)
            ax.set_xlabel("Entry threshold (σ)", fontsize=8)
            ax.set_ylabel("OOS Sharpe", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        for idx in range(len(windows), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        fig.suptitle(
            f"{symbol} — Parameter Sensitivity: Entry Threshold ({interval})\n"
            f"wide hill = robust · sharp spike = overfitted",
            fontsize=11, y=1.01
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 4. Performance table figure
    # ------------------------------------------------------------------

    @staticmethod
    def performance_table_figure(
        summary_table: pd.DataFrame,
        degradation_table: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> plt.Figure:
        """
        Renders summary_table and degradation_table as a two-panel
        matplotlib table figure. Cells colour-coded by quality.

        Returns
        -------
        matplotlib Figure
        """
        fig = plt.figure(figsize=(16, max(3, 0.55 * len(summary_table) + 2)))
        gs  = gridspec.GridSpec(1, 2, wspace=0.4)

        # ---- Summary table ----
        ax1 = fig.add_subplot(gs[0])
        ax1.axis("off")

        s_cols = ["is_sharpe", "is_pnl", "oos_sharpe", "oos_pnl", "sharpe_ratio"]
        s_lbls = ["IS Sharpe", "IS PnL", "OOS Sharpe", "OOS PnL", "IS→OOS ratio"]

        existing_s = [c for c in s_cols if c in summary_table.columns]
        existing_l = [s_lbls[s_cols.index(c)] for c in existing_s]

        s_text = []
        for w, row in summary_table.iterrows():
            cells = []
            for c in existing_s:
                v = row[c]
                if pd.isna(v):
                    cells.append("—")
                else:
                    sign = "+" if float(v) > 0 else ""
                    cells.append(f"{sign}{float(v):.4f}")
            s_text.append(cells)

        if s_text:
            tbl1 = ax1.table(
                cellText=s_text,
                rowLabels=[f"W={w}" for w in summary_table.index],
                colLabels=existing_l,
                loc="center",
                cellLoc="center"
            )
            tbl1.auto_set_font_size(False)
            tbl1.set_fontsize(8)
            tbl1.scale(1.0, 1.5)

            oos_sharpe_idx = existing_s.index("oos_sharpe") if "oos_sharpe" in existing_s else None
            ratio_idx      = existing_s.index("sharpe_ratio") if "sharpe_ratio" in existing_s else None

            for r_idx, (w, row) in enumerate(summary_table.iterrows(), start=1):
                if oos_sharpe_idx is not None:
                    v = row.get("oos_sharpe", 0)
                    if pd.notna(v):
                        tbl1[r_idx, oos_sharpe_idx].set_facecolor(
                            "#d4edda" if float(v) > 0 else "#f8d7da"
                        )
                if ratio_idx is not None:
                    v = row.get("sharpe_ratio")
                    if pd.notna(v):
                        v = float(v)
                        color = (
                            "#d4edda" if v >= 0.8 else
                            "#fff3cd" if v >= 0.5 else
                            "#f8d7da"
                        )
                        tbl1[r_idx, ratio_idx].set_facecolor(color)

        ax1.set_title("IS vs OOS Summary", fontsize=9, pad=10)

        # ---- Degradation table ----
        ax2 = fig.add_subplot(gs[1])
        ax2.axis("off")

        d_cols = ["sharpe_ratio", "pnl_ratio", "quality_rating"]
        d_lbls = ["Sharpe ratio", "PnL ratio", "Quality"]

        existing_d = [c for c in d_cols if c in degradation_table.columns]
        existing_dl = [d_lbls[d_cols.index(c)] for c in existing_d]

        d_text = []
        for w, row in degradation_table.iterrows():
            cells = []
            for c in existing_d:
                v = row[c]
                if pd.isna(v) or v is None:
                    cells.append("—")
                elif c == "quality_rating":
                    cells.append(str(v))
                else:
                    try:
                        sign = "+" if float(v) > 0 else ""
                        cells.append(f"{sign}{float(v):.4f}")
                    except Exception:
                        cells.append(str(v))
            d_text.append(cells)

        if d_text:
            tbl2 = ax2.table(
                cellText=d_text,
                rowLabels=[f"W={w}" for w in degradation_table.index],
                colLabels=existing_dl,
                loc="center",
                cellLoc="center"
            )
            tbl2.auto_set_font_size(False)
            tbl2.set_fontsize(8)
            tbl2.scale(1.0, 1.5)

            rating_idx = existing_d.index("quality_rating") if "quality_rating" in existing_d else None

            rating_colors = {
                "STRONG":     "#d4edda",
                "ACCEPTABLE": "#fff3cd",
                "WEAK":       "#fde8cc",
                "OVERFIT":    "#f8d7da",
                "SKIP":       "#e9ecef",
            }

            for r_idx, (_, row) in enumerate(degradation_table.iterrows(), start=1):
                if rating_idx is not None:
                    rating = row.get("quality_rating", "SKIP")
                    tbl2[r_idx, rating_idx].set_facecolor(
                        rating_colors.get(rating, "#e9ecef")
                    )

        ax2.set_title("Degradation Ratings", fontsize=9, pad=10)

        fig.suptitle(
            f"{symbol} — Step 7 Performance Tables ({interval})",
            fontsize=11, y=1.02
        )
        fig.tight_layout()
        return fig