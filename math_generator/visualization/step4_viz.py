import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Step4Viz:
    """
    Step 4 visualisation layer.
    Returns figure objects — no plt.show() calls.
    Consistent with StabilityEngineViz and Step3Viz patterns.
    """

    # ------------------------------------------------------------------
    # 1. Sub-score heatmap (primary ranking visual)
    # ------------------------------------------------------------------

    @staticmethod
    def ranking_heatmap(
        scores: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> plt.Figure:
        """
        Seaborn heatmap of sub-scores and composite per window.

        Rows   = windows (sorted by rank ascending)
        Columns = stat_score | hl_score | conv_score | composite

        Eligible windows are shown with full opacity.
        Ineligible windows are visually separated by a lighter annotation.

        Parameters
        ----------
        scores   : window_scores DataFrame from WindowDiagnostics
        symbol   : ticker string for title
        interval : timeframe string for title

        Returns
        -------
        matplotlib Figure
        """
        col_order = ["stat_score", "hl_score", "conv_score", "composite"]
        col_labels = ["Stationarity", "Half-Life", "Conv. Speed", "Composite"]

        sorted_scores = scores.sort_values("rank")
        heat_data     = sorted_scores[col_order].copy()
        heat_data.index = [
            f"W={w}{'*' if not sorted_scores.loc[w, 'eligible'] else ''}"
            for w in sorted_scores.index
        ]

        fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(heat_data))))

        sns.heatmap(
            heat_data,
            ax=ax,
            annot=True,
            fmt=".3f",
            cmap="YlGn",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Score [0, 1]"},
            xticklabels=col_labels,
        )

        ax.set_title(
            f"{symbol} — Window Ranking Heatmap ({interval})\n"
            f"* = ineligible (excluded from ranking)",
            fontsize=11,
            pad=12
        )
        ax.set_ylabel("VWAP Window")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 2. Composite score bar chart
    # ------------------------------------------------------------------

    @staticmethod
    def composite_bar(
        scores: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> plt.Figure:
        """
        Horizontal bar chart of composite scores, coloured by eligibility.
        Gives a quick visual of score magnitude alongside the heatmap.

        Returns
        -------
        matplotlib Figure
        """
        sorted_scores = scores.sort_values("rank")
        labels   = [f"W={w}" for w in sorted_scores.index]
        values   = sorted_scores["composite"].tolist()
        eligible = sorted_scores["eligible"].tolist()
        colors   = ["#4caf82" if e else "#d9534f" for e in eligible]

        fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(labels))))

        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.55)

        # Annotate composite value on each bar
        for bar, val in zip(bars, values[::-1]):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9
            )

        ax.set_xlim(0, 1.12)
        ax.set_xlabel("Composite Score")
        ax.set_title(
            f"{symbol} — Composite Scores ({interval})\n"
            f"green = eligible   red = excluded",
            fontsize=11
        )
        ax.axvline(0, linewidth=0.8, color="gray")
        ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        return fig