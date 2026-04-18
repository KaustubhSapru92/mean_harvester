from workflow.vwap import VWAPCalculator
from diagnostics.rolling_vwap_stability import RollingVWAPStability


def run_vwap_pipeline(clean_df, vwap_windows, roll_len=100):

    base_vwap = VWAPCalculator.compute_rolling_vwap(
        clean_df,
        windows=vwap_windows
    )

    rolling_stability_map = {}

    for w in vwap_windows:
        rolling_stability_map[w] = RollingVWAPStability.compute(
            base_vwap,
            window=w,
            roll_len=roll_len
        )

    return {
        "base_vwap": base_vwap,
        "rolling_stability": rolling_stability_map
    }
