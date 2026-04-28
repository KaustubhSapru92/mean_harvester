import pandas as pd
from workflow.vwap import VWAPCalculator
from diagnostics.rolling_vwap_stability import RollingVWAPStability
from diagnostics.deviation_normalizer import DeviationNormalizer
from diagnostics.adf_tester import ADFTester
from diagnostics.halflife_calculator import HalfLifeCalculator
from diagnostics.convergence_checker import ConvergenceChecker
from diagnostics.window_diagnostics import WindowDiagnostics

@property
def scores(self) -> pd.DataFrame:
    if self._scores is None:
        raise RuntimeError("Scores not computed yet.")
    return self._scores.copy()


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

    diagnostics = WindowDiagnostics(
        adf_results=adf_results,
        hl_results=hl_results,
        convergence_summary=convergence_summary,
        convergence_detail=convergence_detail,
    )

    eligible_set = diagnostics.build_eligible_set()
    print(WindowDiagnostics.eligible_report(eligible_set))

    diagnostics.derive_convergence_speed()
    print(WindowDiagnostics.convergence_speed_report(diagnostics.scores))

    diagnostics.compute_subscores()
    print(WindowDiagnostics.subscores_report(diagnostics.scores))

    window_scores = diagnostics.compute_composite()
    print(WindowDiagnostics.ranking_report(window_scores))

    return {
        "base_vwap": base_vwap,
        "rolling_stability": rolling_stability_map,
        "ndev_map": ndev_map,        # Stage 1 output
        "adf_results": adf_results,  # Stage 2 output → feeds AR(1)
        "hl_results": hl_results,    # Stage 3 output → feeds convergence
        "convergence_summary": convergence_summary,  # Stage 4 summary → feeds Step 4 ranking
        "convergence_detail": convergence_detail,  # Stage 4 detail → feeds Stage 5 plots
        "diagnostics": diagnostics,  # WindowDiagnostics instance — carries state across Stage 2-4
        "eligible_set": eligible_set,  # step 4 Stage 1 output
        "scores": diagnostics.scores,   # Stage 3 output — eligible set + all sub-scores
        "window_scores": window_scores   # Stage 4 output — full ranked DataFrame
    }
