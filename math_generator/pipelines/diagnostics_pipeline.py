import pandas as pd
from workflow.plotter import OHLCVPlotter
from diagnostics.vwap_stability import VWAPStabilityDiagnostics
from workflow.vwap import VWAPCalculator
from visualization.step3_viz import Step3Viz
from visualization.step4_viz import Step4Viz
from visualization.step5_viz import Step5Viz
from visualization.step6_viz import Step6Viz
from visualization.step7_viz import Step7Viz


def run_static_diagnostics(base_vwap, vwap_windows, lookback_days):

    window_stats = VWAPStabilityDiagnostics.window_stability(
        base_vwap,
        windows=vwap_windows
    )

    history_stats = VWAPStabilityDiagnostics.history_stability(
        base_vwap,
        window=50,
        history_days=[1,2,3,4,5,7,lookback_days]
    )

    OHLCVPlotter.plot_window_stability(window_stats)
    OHLCVPlotter.plot_history_stability(history_stats, window=50)

    return window_stats, history_stats


def run_rolling_diagnostics(base_vwap, rolling_stability_map, roll_len):

    OHLCVPlotter.plot_vwap_window_comparison(
        stability_map=rolling_stability_map,
        roll_len=roll_len
    )


def run_base_visual_diagnostics(base_vwap, symbol, interval):

    OHLCVPlotter.plot_price_vs_vwap(
        base_vwap,
        symbol,
        interval,
        windows=(20, 50)
    )

    OHLCVPlotter.plot_vwap_deviation(
        base_vwap,
        symbol,
        interval,
        window=20
    )

    OHLCVPlotter.plot_vwap_deviation_distribution(
        base_vwap,
        symbol,
        interval,
        window=20
    )


def run_resampled_diagnostics(resampled_map, vwap_windows, symbol):

    results = {}

    for tf, df in resampled_map.items():

        resampled_vwap = VWAPCalculator.compute_rolling_vwap(
            df,
            windows=vwap_windows
        )

        OHLCVPlotter.plot_price_vs_vwap(
            resampled_vwap,
            symbol,
            tf,
            windows=(20, 50)
        )

        OHLCVPlotter.plot_vwap_deviation(
            resampled_vwap,
            symbol,
            tf,
            window=20
        )

        results[tf] = resampled_vwap

    return results

def run_step3_visualisations(ndev_map: dict, convergence_detail: dict,
                             adf_results,
                             hl_results,
                             convergence_summary,
                             symbol: str,
                             interval: str
                             ):
    """
    Stage 5 of Step 3.
    Generates and displays all Step 3 diagnostic plots.

    Parameters come directly from the vwap_pipeline return dict.
    """

    fig_dist = Step3Viz.deviation_distributions(ndev_map, symbol, interval)
    fig_dist.show()

    if convergence_detail:
        fig_conv = Step3Viz.convergence_traces(convergence_detail, symbol, interval)
        fig_conv.show()

    fig_table = Step3Viz.results_table(adf_results,
                                       hl_results,
                                       convergence_summary,
                                       symbol,
                                       interval)
    fig_table.show()

def run_step4_visualisations(window_scores: pd.DataFrame,
                             symbol: str,
                             interval: str ):
    """
    Stage 4 of Step 4.
    Generates and displays Step 4 ranking plots.

    Parameters come directly from the vwap_pipeline return dict.
    """
    fig_heatmap = Step4Viz.ranking_heatmap(window_scores, symbol, interval)
    fig_heatmap.show()

    fig_bar = Step4Viz.composite_bar(window_scores, symbol, interval)
    fig_bar.show()

def run_step5_visualisations(base_vwap: pd.DataFrame,
                             atr_df: pd.DataFrame,
                             hurst_results: pd.DataFrame,
                             vndev_map: dict,
                             symbol: str,
                             interval: str ):
    """
    Stage 5 of Step 5.
    Generates and displays all Step 5 regime filter plots.

    Parameters come directly from the vwap_pipeline return dict.
    """
    fig_atr = Step5Viz.atr_overlay(base_vwap, atr_df, symbol, interval)
    fig_atr.show()

    fig_hurst = Step5Viz.hurst_bar(hurst_results, symbol, interval)
    fig_hurst.show()

    fig_vndev = Step5Viz.vndev_distributions(vndev_map, symbol, interval)
    fig_vndev.show()


def run_step6_visualisations(equity_curves: pd.DataFrame, drawdown_curves: pd.DataFrame,
                             cross_summary: pd.DataFrame,
                             vndev_map: dict,
                             trade_log_df: pd.DataFrame,
                             entry_threshold: float,
                             stop_threshold: float,
                             symbol: str,
                             interval: str ):
    """
    Stage 6 of Step 6.
    Generates and displays all Step 6 backtester plots.

    Parameters come directly from the vwap_pipeline return dict.
    """

    fig_equity = Step6Viz.equity_and_drawdown(
        equity_curves, drawdown_curves, cross_summary, symbol, interval
    )
    fig_equity.show()

    fig_signal = Step6Viz.vndev_signal_overlay(
        vndev_map, trade_log_df,
        entry_threshold, stop_threshold,
        symbol, interval
    )
    fig_signal.show()

    fig_table = Step6Viz.metrics_summary_table(cross_summary, symbol, interval)
    fig_table.show()


def run_step7_visualisations(
    oos_results: dict,
    cost_results: dict,
    summary_table,
    degradation_table,
    wf_splits: dict,
    sized_results: dict,
    window_scores,
    base_vwap,
    signal_gen,
    symbol: str,
    interval: str,
    roll_window: int = 50,
):
    """
    Stage 5 of Step 7.
    Generates and displays all Step 7 survival and validation plots.

    Parameters come directly from the vwap_pipeline return dict.
    """
    fig_overlay = Step7Viz.is_oos_equity_overlay(
        oos_results, cost_results, summary_table, symbol, interval
    )
    fig_overlay.show()

    fig_rolling = Step7Viz.rolling_sharpe(
        oos_results, symbol, interval, roll_window=roll_window
    )
    fig_rolling.show()

    fig_sweep = Step7Viz.sensitivity_sweep(
        oos_results, wf_splits, sized_results,
        window_scores, base_vwap, signal_gen,
        symbol, interval
    )
    fig_sweep.show()

    fig_tables = Step7Viz.performance_table_figure(
        summary_table, degradation_table, symbol, interval
    )
    fig_tables.show()