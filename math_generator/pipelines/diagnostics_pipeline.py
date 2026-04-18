from workflow.plotter import OHLCVPlotter
from diagnostics.vwap_stability import VWAPStabilityDiagnostics
from workflow.vwap import VWAPCalculator


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
