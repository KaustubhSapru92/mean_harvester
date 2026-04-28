from workflow.vwap import VWAPCalculator
from diagnostics.rolling_vwap_stability import RollingVWAPStability
from diagnostics.deviation_normalizer import DeviationNormalizer
from diagnostics.adf_tester import ADFTester
from diagnostics.halflife_calculator import HalfLifeCalculator
from diagnostics.convergence_checker import ConvergenceChecker


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
    ndev_map = DeviationNormalizer.extract(base_vwap, vwap_windows)
    adf_results = ADFTester.test_all(ndev_map)

    print(ADFTester.flag_report(adf_results))

    stationary_windows = ADFTester.stationary_windows(adf_results)
    hl_results = HalfLifeCalculator.compute_all(ndev_map, stationary_windows)

    print(HalfLifeCalculator.flag_report(hl_results))

    convergence_summary, convergence_detail = ConvergenceChecker.check_all(
        ndev_map, stationary_windows
    )

    print(ConvergenceChecker.flag_report(convergence_summary))

    return {
        "base_vwap": base_vwap,
        "rolling_stability": rolling_stability_map,
        "ndev_map": ndev_map,        # Stage 1 output
        "adf_results": adf_results,  # Stage 2 output → feeds AR(1)
        "hl_results": hl_results,    # Stage 3 output → feeds convergence
        "convergence_summary": convergence_summary,  # Stage 4 summary → feeds Step 4 ranking
        "convergence_detail": convergence_detail  # Stage 4 detail → feeds Stage 5 plots
    }
