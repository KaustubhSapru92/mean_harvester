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
from backtester.signal_generator import SignalGenerator
from backtester.position_sizer import PositionSizer
from backtester.pnl_engine import PnLEngine
from backtester.performance_metrics import PerformanceMetrics
from backtester.trade_book import TradeBook




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

    # Step 6 — Stage 1: Signal generation
    entry_threshold = config.get("backtester", "entry_threshold") or 2.0
    stop_threshold = config.get("backtester", "stop_threshold") or 3.0

    signal_gen = SignalGenerator(entry_threshold, stop_threshold)
    signal_results = signal_gen.generate_all(vndev_map, window_scores)

    print(SignalGenerator.flag_report(signal_results))

    # Step 6 — Stage 2: Kelly position sizing
    max_position_pct = config.get("backtester", "max_position_pct") or 0.25

    sizer = PositionSizer(max_position_pct)
    sized_results = sizer.size_all(signal_results, base_vwap)

    print(PositionSizer.kelly_report(sized_results))
    # Step 6 — Stage 3: Vectorised PnL

    pnl_results = PnLEngine.compute_all(sized_results, base_vwap)

    print(PnLEngine.flag_report(pnl_results))

    # ------------------------------------------------------------------
    # Step 6 — Stage 4: Performance metrics
    # ------------------------------------------------------------------
    bars_per_day = config.get("backtester", "bars_per_day") or 78

    perf = PerformanceMetrics(bars_per_day)
    metrics_df, metrics_map = perf.compute_all(pnl_results)

    print(PerformanceMetrics.flag_report(metrics_df))
    print(PerformanceMetrics.metrics_table(metrics_df))
    # ------------------------------------------------------------------
    # Step 6 — Stage 5: Trade log and equity curve assembly
    # ------------------------------------------------------------------
    trade_log_df = TradeBook.build_trade_log(pnl_results, metrics_map)
    equity_curves = TradeBook.build_equity_curves(pnl_results, base_vwap)
    drawdown_curves = TradeBook.build_drawdown_curves(pnl_results, base_vwap)
    cross_summary = TradeBook.cross_window_summary(trade_log_df, metrics_df)

    print(TradeBook.flag_report(trade_log_df, equity_curves))


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
        "atr_summary": atr_summary,  # Step 5 Stage 4 — ATR filter summary stats
        "signal_gen": signal_gen,  # Step 6 — SignalGenerator instance
        "signal_results": signal_results,  # Step 6 Stage 1 — signals + trade logs per window
        "sizer": sizer,  # Step 6 — PositionSizer instance
        "sized_results": sized_results,  # Step 6 Stage 2 — positions + Kelly meta per window
        "pnl_results": pnl_results,  # Step 6 Stage 3 — bar_pnl, equity, drawdown per window
        "metrics_df": metrics_df,  # Step 6 Stage 4 — metrics DataFrame (one row per window)
        "metrics_map": metrics_map,  # Step 6 Stage 4 — metrics dict per window
        "trade_log_df": trade_log_df,  # Step 6 Stage 5 — consolidated trade log
        "equity_curves": equity_curves,  # Step 6 Stage 5 — multi-window equity DataFrame
        "drawdown_curves": drawdown_curves,  # Step 6 Stage 5 — multi-window drawdown DataFrame
        "cross_summary": cross_summary,  # Step 6 Stage 5 — cross-window ranked summary
    }
