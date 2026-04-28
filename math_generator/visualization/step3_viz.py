import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np


class Step3Viz:
    """
    Stage 5 of Step 3.

    All plots return figure objects (consistent with StabilityEngineViz pattern).
    No plt.show() calls — caller decides when to render or save.

    Two plot types:
      1. deviation_distributions  — histogram + KDE per window
      2. convergence_traces       — ADF p-value and half-life vs slice fraction
    """

    # ------------------------------------------------------------------
    # 1. Deviation Distribution per window
    # ------------------------------------------------------------------

    @staticmethod
    def deviation_distributions(
        ndev_map: dict[int, pd.Series],
        symbol: str,
        interval: str
    ) -> plt.Figure:
        """
        One subplot per window showing the distribution of NDEV_W.
        Overlays a normal reference curve for visual fat-tail detection.

        Parameters
        ----------
        ndev_map : { window: pd.Series } from DeviationNormalizer
        symbol   : ticker string for title
        interval : timeframe string for title

        Returns
        -------
        matplotlib Figure
        """
        windows = list(ndev_map.keys())
        n_cols  = min(3, len(windows))
        n_rows  = int(np.ceil(len(windows) / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 4 * n_rows),
            squeeze=False
        )

        for idx, w in enumerate(windows):
            ax     = axes[idx // n_cols][idx % n_cols]
            series = ndev_map[w].dropna()

            # Histogram
            ax.hist(
                series,
                bins=60,
                density=True,
                alpha=0.6,
                label="NDEV distribution"
            )

            # Normal reference overlay
            mu, sigma = series.mean(), series.std()
            x_range   = np.linspace(series.min(), series.max(), 200)
            normal_pdf = (
                1 / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
            )
            ax.plot(x_range, normal_pdf, linewidth=1.5, linestyle="--", label="Normal ref")

            # Zero line
            ax.axvline(0, color="black", linewidth=0.8, linestyle=":")

            ax.set_title(f"W={w}  μ={mu:.4f}  σ={sigma:.4f}")
            ax.set_xlabel("Normalised Deviation (NDEV)")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(windows), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        fig.suptitle(
            f"{symbol} — NDEV Distributions ({interval})",
            fontsize=13,
            y=1.01
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 2. Convergence traces per window
    # ------------------------------------------------------------------

    @staticmethod
    def convergence_traces(
        convergence_detail: dict[int, dict],
        symbol: str,
        interval: str
    ) -> plt.Figure:
        """
        Two-panel plot per window:
          top    — ADF p-value vs slice fraction (with p=0.05 threshold line)
          bottom — half-life in bars vs slice fraction

        Parameters
        ----------
        convergence_detail : { window: check_window dict } from ConvergenceChecker
        symbol             : ticker string for title
        interval           : timeframe string for title

        Returns
        -------
        matplotlib Figure
        """
        windows = list(convergence_detail.keys())
        n_cols  = min(3, len(windows))
        n_rows  = int(np.ceil(len(windows) / n_cols)) * 2   # 2 panels per window

        fig = plt.figure(figsize=(6 * n_cols, 3.5 * (n_rows // 2)))
        gs  = gridspec.GridSpec(
            n_rows, n_cols,
            hspace=0.55,
            wspace=0.35
        )

        for idx, w in enumerate(windows):
            detail     = convergence_detail[w]
            fractions  = detail["slice_fractions"]
            p_values   = detail["p_values"]
            half_lives = detail["half_lives"]

            col      = idx % n_cols
            top_row  = (idx // n_cols) * 2
            bot_row  = top_row + 1

            # ---- ADF p-value panel ----
            ax_p = fig.add_subplot(gs[top_row, col])

            # Replace None with NaN for clean plotting
            p_clean  = [v if v is not None else np.nan for v in p_values]
            ax_p.plot(fractions, p_clean, marker="o", markersize=3, linewidth=1)
            ax_p.axhline(
                0.05,
                linestyle="--",
                linewidth=0.9,
                color="red",
                label="p=0.05"
            )
            ax_p.set_title(f"W={w}  ADF p-value", fontsize=9)
            ax_p.set_ylabel("p-value", fontsize=8)
            ax_p.set_ylim(-0.02, 1.05)
            ax_p.legend(fontsize=7)
            ax_p.grid(True, alpha=0.3)

            converged_tag = "✓ converged" if detail["converged"] else "✗ not converged"
            ax_p.set_xlabel(converged_tag, fontsize=7)

            # ---- Half-life panel ----
            ax_hl = fig.add_subplot(gs[bot_row, col])

            hl_clean = [v if v is not None else np.nan for v in half_lives]
            ax_hl.plot(
                fractions, hl_clean,
                marker="s",
                markersize=3,
                linewidth=1,
                color="darkorange"
            )
            ax_hl.set_title(f"W={w}  Half-Life (bars)", fontsize=9)
            ax_hl.set_ylabel("bars", fontsize=8)
            ax_hl.set_xlabel("Slice fraction of history", fontsize=8)
            ax_hl.grid(True, alpha=0.3)

        fig.suptitle(
            f"{symbol} — Convergence Traces ({interval})",
            fontsize=13
        )
        return fig

    # ------------------------------------------------------------------
    # 3. Per-window results summary table (text figure)
    # ------------------------------------------------------------------

    @staticmethod
    def results_table(
        adf_results: pd.DataFrame,
        hl_results: pd.DataFrame,
        convergence_summary: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> plt.Figure:
        """
        Renders the consolidated per-window results as a matplotlib table figure.
        Useful for saving as an image alongside other plots.

        Columns: window | p_value | is_stationary | half_life_bars | converged
        """
        # Merge all three result frames on window index
        merged = adf_results[["p_value", "is_stationary"]].copy()
        merged["half_life_bars"] = hl_results["half_life_bars"]
        merged["converged"]      = convergence_summary["converged"]
        merged = merged.reset_index()

        col_labels = ["Window", "ADF p-value", "Stationary", "Half-Life (bars)", "Converged"]
        cell_text  = []

        for _, row in merged.iterrows():
            p    = f"{row['p_value']:.4f}"  if row["p_value"]        is not None else "—"
            hl   = f"{row['half_life_bars']:.1f}" if pd.notna(row["half_life_bars"]) else "—"
            stat = "✓" if row["is_stationary"] else "✗"
            conv = "✓" if row["converged"]     else "✗"
            cell_text.append([str(int(row["window"])), p, stat, hl, conv])

        fig, ax = plt.subplots(figsize=(9, 1.2 + 0.5 * len(merged)))
        ax.axis("off")

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.6)

        # Colour stationary / converged cells
        for row_idx, row in enumerate(cell_text, start=1):
            # Stationary column (index 2)
            color = "#d4edda" if row[2] == "✓" else "#f8d7da"
            table[row_idx, 2].set_facecolor(color)
            # Converged column (index 4)
            color = "#d4edda" if row[4] == "✓" else "#f8d7da"
            table[row_idx, 4].set_facecolor(color)

        fig.suptitle(
            f"{symbol} — Step 3 Results Summary ({interval})",
            fontsize=12,
            y=0.98
        )
        fig.tight_layout()
        return fig