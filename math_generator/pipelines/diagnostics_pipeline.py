from workflow.plotter import OHLCVPlotter
from diagnostics.vwap_stability import VWAPStabilityDiagnostics
from workflow.vwap import VWAPCalculator
from visualization.step3_viz import Step3Viz


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
