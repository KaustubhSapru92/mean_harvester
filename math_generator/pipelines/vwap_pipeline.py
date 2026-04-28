import pandas as pd
from workflow.vwap import VWAPCalculator
from diagnostics.rolling_vwap_stability import RollingVWAPStability
from diagnostics.deviation_normalizer import DeviationNormalizer
from diagnostics.adf_tester import ADFTester
from diagnostics.halflife_calculator import HalfLifeCalculator
from diagnostics.convergence_checker import ConvergenceChecker
from diagnostics.window_diagnostics import WindowDiagnostics
from diagnostics.atr_calculator import ATRCalculator
from diagnostics.hurst_calculator import HurstCalculator
from diagnostics.vol_normalizer import VolNormalizer

def run_vwap_pipeline(clean_df, vwap_windows, config, roll_len=100):

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

    # Step 5 — Stage 1: ATR computation
    atr_window = config.get("regime", "atr_window") or 14
    atr_percentile = config.get("regime", "atr_percentile") or 75.0

    atr_df = ATRCalculator.compute(base_vwap, atr_window, atr_percentile)
    low_vol_mask = ATRCalculator.low_vol_mask(atr_df)

    print(ATRCalculator.flag_report(atr_df))

    # Step 5 — Stage 2: Hurst exponent on low-vol subsets
    hurst_window = config.get("regime", "hurst_window") or 100
    eligible_windows = list(window_scores[window_scores["eligible"]].index)

    hurst_results = HurstCalculator.compute_all(ndev_map,
                                                low_vol_mask,
                                                eligible_windows,
                                                hurst_window)
    print(HurstCalculator.flag_report(hurst_results))

    # Step 5 — Stage 3: Volatility-normalised NDEV
    vol_norm_window = config.get("regime", "vol_norm_window") or 20

    vndev_map = VolNormalizer.compute_all(
        ndev_map,
        low_vol_mask,
        eligible_windows,
        hurst_results,
        vol_norm_window,
    )

    print(VolNormalizer.flag_report(vndev_map))

    # Step 5 — Stage 4: Output assembly
    # Enrich window_scores with regime columns so Step 6 has one table.
    atr_summary = ATRCalculator.summary(atr_df)

    window_scores["hurst_H"] = hurst_results["hurst_H"]
    window_scores["hurst_pass"] = hurst_results["hurst_pass"]
    window_scores["n_low_vol_bars"] = hurst_results["n_low_vol_bars"]
    window_scores["low_vol_pct"] = round(atr_summary["low_vol_pct"], 2)
    window_scores["regime_pass"] = (
            window_scores["eligible"] & window_scores["hurst_pass"].fillna(False)
    )

    print("\nStep 5 — Regime-filtered window summary:")
    for w, row in window_scores.iterrows():
        tag = "REGIME_PASS" if row["regime_pass"] else "REGIME_FAIL"
        h = f"H={row['hurst_H']:.4f}" if pd.notna(row["hurst_H"]) else "H=—"
        print(
            f"  W={w:>4}  [{tag:<11}]  {h}  "
            f"low_vol_pct={row['low_vol_pct']:.1f}%"
        )

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
        "window_scores": window_scores,   # Stage 4 output — full ranked DataFrame
        "atr_df": atr_df,  # Step 5 Stage 1 — ATR + regime flags
        "low_vol_mask": low_vol_mask,  # Step 5 Stage 1 — boolean mask for low-vol bars
        "hurst_results": hurst_results,  # Step 5 Stage 2 — Hurst H per eligible window
        "vndev_map": vndev_map,  # Step 5 Stage 3 — VNDEV per surviving window
        "atr_summary": atr_summary  # Step 5 Stage 4 — ATR filter summary stats
    }
