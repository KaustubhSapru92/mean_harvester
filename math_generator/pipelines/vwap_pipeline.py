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
from backtester.transaction_costs import TransactionCostEngine
from backtester.oos_engine import OOSEngine
from backtester.performance_tables import PerformanceTables
from backtester.frozen_params import FrozenStrategyParams


TRAIN_FRACTION = 0.70


def chronological_split(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    if df.empty:
        raise ValueError("Cannot split an empty OHLCV DataFrame.")
    if not (0 < train_fraction < 1):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    cut_idx = int(len(df) * train_fraction)
    cut_idx = max(1, min(cut_idx, len(df) - 1))
    return df.iloc[:cut_idx].copy(), df.iloc[cut_idx:].copy()


def validate_data_sufficiency(
    clean_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vwap_windows: list[int],
):
    max_window = max(vwap_windows)
    min_train_bars = max_window * 5

    if len(clean_df) < min_train_bars + max_window:
        raise ValueError(
            "Insufficient total bars for leak-free validation: "
            f"need at least {min_train_bars + max_window}, got {len(clean_df)}."
        )
    if len(train_df) < min_train_bars:
        raise ValueError(
            "Insufficient train bars for stable window diagnostics: "
            f"need at least {min_train_bars}, got {len(train_df)}."
        )
    if len(test_df) < max_window:
        raise ValueError(
            "Insufficient test bars for the largest VWAP window: "
            f"need at least {max_window}, got {len(test_df)}."
        )


def learn_on_train(train_df, vwap_windows, config, roll_len=100):
    train_vwap = VWAPCalculator.compute_rolling_vwap(train_df, windows=vwap_windows)

    rolling_stability_map = {
        w: RollingVWAPStability.compute(train_vwap, window=w, roll_len=roll_len)
        for w in vwap_windows
    }

    ndev_map = DeviationNormalizer.extract(train_vwap, vwap_windows)
    adf_results = ADFTester.test_all(ndev_map)
    print(ADFTester.flag_report(adf_results))

    stationary_windows = ADFTester.stationary_windows(adf_results)
    hl_results = HalfLifeCalculator.compute_all(ndev_map, stationary_windows)
    print(HalfLifeCalculator.flag_report(hl_results))

    convergence_summary, convergence_detail = ConvergenceChecker.check_all(
        ndev_map,
        stationary_windows,
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

    atr_window = config.get("regime", "atr_window") or 14
    atr_percentile = config.get("regime", "atr_percentile") or 75.0
    train_atr_df = ATRCalculator.compute(train_vwap, atr_window, atr_percentile)
    train_low_vol_mask = ATRCalculator.low_vol_mask(train_atr_df)
    print(ATRCalculator.flag_report(train_atr_df))

    hurst_window = config.get("regime", "hurst_window") or 100
    eligible_windows = list(window_scores[window_scores["eligible"]].index)
    hurst_results = HurstCalculator.compute_all(
        ndev_map,
        train_low_vol_mask,
        eligible_windows,
        hurst_window,
    )
    print(HurstCalculator.flag_report(hurst_results))

    vol_norm_window = config.get("regime", "vol_norm_window") or 20
    vndev_map = VolNormalizer.compute_all(
        ndev_map,
        train_low_vol_mask,
        eligible_windows,
        hurst_results,
        vol_norm_window,
    )
    print(VolNormalizer.flag_report(vndev_map))

    atr_summary = ATRCalculator.summary(train_atr_df)
    window_scores["hurst_H"] = hurst_results["hurst_H"]
    window_scores["hurst_pass"] = hurst_results["hurst_pass"]
    window_scores["n_low_vol_bars"] = hurst_results["n_low_vol_bars"]
    window_scores["low_vol_pct"] = round(atr_summary["low_vol_pct"], 2)
    window_scores["regime_pass"] = (
        window_scores["eligible"] & window_scores["hurst_pass"].fillna(False)
    )

    candidates = window_scores[
        window_scores["regime_pass"] & window_scores.index.isin(vndev_map.keys())
    ].sort_values("rank")
    if candidates.empty:
        raise ValueError("No train window passed eligibility and regime filters.")

    best_window = int(candidates.index[0])
    best_row = window_scores.loc[best_window]

    entry_threshold = config.get("backtester", "entry_threshold") or 2.0
    stop_threshold = config.get("backtester", "stop_threshold") or 3.0
    signal_gen = SignalGenerator(entry_threshold, stop_threshold)

    train_signal_result = signal_gen.generate(
        vndev=vndev_map[best_window],
        half_life_bars=float(best_row["half_life_bars"]),
        window=best_window,
    )
    print(SignalGenerator.flag_report({best_window: train_signal_result}))

    max_position_pct = config.get("backtester", "max_position_pct") or 0.25
    sizer = PositionSizer(max_position_pct)
    train_returns = train_vwap["Close"].pct_change().fillna(0.0)
    kelly_meta = sizer.compute_kelly(
        trade_log=train_signal_result["trade_log"],
        bar_returns=train_returns,
        signal=train_signal_result["signal"],
    )
    train_position = sizer.scale_positions(
        train_signal_result["signal"],
        kelly_meta["kelly_capped"],
    )
    sized_results = {
        best_window: {
            "signal": train_signal_result["signal"],
            "trade_log": train_signal_result["trade_log"],
            "position": train_position,
            "kelly_meta": kelly_meta,
        }
    }
    print(PositionSizer.kelly_report(sized_results))

    train_pnl_raw = PnLEngine.compute_window(
        position=train_position,
        base_vwap=train_vwap,
        window=best_window,
    )
    train_pnl_result = {
        **sized_results[best_window],
        **train_pnl_raw,
    }
    print(PnLEngine.flag_report({best_window: train_pnl_result}))

    cost_engine = TransactionCostEngine(round_trip_pct=0.001)
    train_cost = cost_engine.apply_to_segment(
        bar_pnl=train_pnl_result["bar_pnl"],
        trade_log=train_signal_result["trade_log"],
        kelly_capped=kelly_meta["kelly_capped"],
        window=best_window,
    )

    train_result = {
        **train_pnl_result,
        **train_cost,
    }

    frozen_params = FrozenStrategyParams(
        window=best_window,
        half_life_bars=float(best_row["half_life_bars"]),
        entry_threshold=float(entry_threshold),
        stop_threshold=float(stop_threshold),
        kelly_capped=float(kelly_meta["kelly_capped"]),
        atr_threshold=float(train_atr_df["atr_threshold"].iloc[0]),
        atr_window=int(atr_window),
        hurst_H=(
            float(best_row["hurst_H"])
            if pd.notna(best_row["hurst_H"]) else None
        ),
        vol_norm_window=int(vol_norm_window),
    )

    return {
        "frozen_params": frozen_params,
        "train_vwap": train_vwap,
        "rolling_stability": rolling_stability_map,
        "ndev_map": ndev_map,
        "adf_results": adf_results,
        "hl_results": hl_results,
        "convergence_summary": convergence_summary,
        "convergence_detail": convergence_detail,
        "diagnostics": diagnostics,
        "eligible_set": eligible_set,
        "scores": diagnostics.scores,
        "window_scores": window_scores,
        "atr_df": train_atr_df,
        "low_vol_mask": train_low_vol_mask,
        "hurst_results": hurst_results,
        "vndev_map": vndev_map,
        "atr_summary": atr_summary,
        "signal_gen": signal_gen,
        "signal_results": {best_window: train_signal_result},
        "sizer": sizer,
        "sized_results": sized_results,
        "pnl_results": {best_window: train_pnl_result},
        "train_result": train_result,
        "cost_engine": cost_engine,
    }


def apply_to_test(test_df, frozen_params: FrozenStrategyParams, config):
    test_vwap = VWAPCalculator.compute_rolling_vwap(
        test_df,
        windows=[frozen_params.window],
    )
    test_ndev_map = DeviationNormalizer.extract(test_vwap, [frozen_params.window])
    test_atr_df = ATRCalculator.compute_with_threshold(
        test_vwap,
        atr_window=frozen_params.atr_window,
        atr_threshold=frozen_params.atr_threshold,
    )
    test_low_vol_mask = ATRCalculator.low_vol_mask(test_atr_df)

    test_vndev = VolNormalizer.compute_window(
        test_ndev_map[frozen_params.window],
        test_low_vol_mask,
        frozen_params.window,
        frozen_params.vol_norm_window,
    )

    signal_gen = SignalGenerator(
        frozen_params.entry_threshold,
        frozen_params.stop_threshold,
    )
    test_signal_result = signal_gen.generate(
        vndev=test_vndev,
        half_life_bars=frozen_params.half_life_bars,
        window=frozen_params.window,
    )
    test_position = test_signal_result["signal"].astype(float) * frozen_params.kelly_capped
    test_position.name = f"position_{frozen_params.window}"

    test_pnl_raw = PnLEngine.compute_window(
        position=test_position,
        base_vwap=test_vwap,
        window=frozen_params.window,
    )
    test_pnl_result = {
        "signal": test_signal_result["signal"],
        "trade_log": test_signal_result["trade_log"],
        "position": test_position,
        "kelly_meta": {"kelly_capped": frozen_params.kelly_capped},
        **test_pnl_raw,
    }

    cost_engine = TransactionCostEngine(round_trip_pct=0.001)
    test_cost = cost_engine.apply_to_segment(
        bar_pnl=test_pnl_result["bar_pnl"],
        trade_log=test_signal_result["trade_log"],
        kelly_capped=frozen_params.kelly_capped,
        window=frozen_params.window,
    )

    return {
        "test_vwap": test_vwap,
        "test_ndev_map": test_ndev_map,
        "test_atr_df": test_atr_df,
        "test_low_vol_mask": test_low_vol_mask,
        "test_vndev_map": {frozen_params.window: test_vndev},
        "test_signal_results": {frozen_params.window: test_signal_result},
        "test_pnl_results": {frozen_params.window: test_pnl_result},
        "test_result": {
            **test_pnl_result,
            **test_cost,
        },
    }


def _build_train_diagnostics(train_artifacts, bars_per_day):
    perf = PerformanceMetrics(bars_per_day)
    metrics_df, metrics_map = perf.compute_all(train_artifacts["pnl_results"])
    trade_log_df = TradeBook.build_trade_log(
        train_artifacts["pnl_results"],
        metrics_map,
    )
    equity_curves = TradeBook.build_equity_curves(
        train_artifacts["pnl_results"],
        train_artifacts["train_vwap"],
    )
    drawdown_curves = TradeBook.build_drawdown_curves(
        train_artifacts["pnl_results"],
        train_artifacts["train_vwap"],
    )
    cross_summary = TradeBook.cross_window_summary(trade_log_df, metrics_df)
    return metrics_df, metrics_map, trade_log_df, equity_curves, drawdown_curves, cross_summary


def run_vwap_pipeline(clean_df, vwap_windows, config, roll_len=100):
    train_df, test_df = chronological_split(clean_df, TRAIN_FRACTION)
    validate_data_sufficiency(clean_df, train_df, test_df, vwap_windows)

    train_artifacts = learn_on_train(train_df, vwap_windows, config, roll_len)
    frozen_params = train_artifacts["frozen_params"]
    test_artifacts = apply_to_test(test_df, frozen_params, config)

    bars_per_day = config.get("backtester", "bars_per_day") or 78
    (
        metrics_df,
        metrics_map,
        trade_log_df,
        equity_curves,
        drawdown_curves,
        cross_summary,
    ) = _build_train_diagnostics(train_artifacts, bars_per_day)

    oos_result = OOSEngine.compare_window(
        window=frozen_params.window,
        train_result=train_artifacts["train_result"],
        test_result=test_artifacts["test_result"],
        frozen_params=frozen_params,
    )
    oos_results = {frozen_params.window: oos_result}
    print(OOSEngine.flag_report(oos_results))

    cost_results = {
        frozen_params.window: {
            "train": {
                "cost_adjusted_pnl": train_artifacts["train_result"]["cost_adjusted_pnl"],
                "cost_adjusted_equity": train_artifacts["train_result"]["cost_adjusted_equity"],
                "cost_drawdown": train_artifacts["train_result"]["cost_drawdown"],
                "total_cost_paid": train_artifacts["train_result"]["total_cost_paid"],
                "n_trades_costed": train_artifacts["train_result"]["n_trades_costed"],
            },
            "test": {
                "cost_adjusted_pnl": test_artifacts["test_result"]["cost_adjusted_pnl"],
                "cost_adjusted_equity": test_artifacts["test_result"]["cost_adjusted_equity"],
                "cost_drawdown": test_artifacts["test_result"]["cost_drawdown"],
                "total_cost_paid": test_artifacts["test_result"]["total_cost_paid"],
                "n_trades_costed": test_artifacts["test_result"]["n_trades_costed"],
            },
        }
    }
    wf_splits = {
        frozen_params.window: {
            "window": frozen_params.window,
            "cut_bar": test_df.index[0],
            "n_train": int(train_artifacts["vndev_map"][frozen_params.window].notna().sum()),
            "n_test": int(test_artifacts["test_vndev_map"][frozen_params.window].notna().sum()),
            "train_vndev": train_artifacts["vndev_map"][frozen_params.window],
            "test_vndev": test_artifacts["test_vndev_map"][frozen_params.window],
            "train_trades": train_artifacts["train_result"]["trade_log"],
            "test_trades": test_artifacts["test_result"]["trade_log"],
            "skip_reason": None,
        }
    }
    combined_vwap = pd.concat(
        [train_artifacts["train_vwap"], test_artifacts["test_vwap"]],
        axis=0,
    )

    summary_table = PerformanceTables.build_summary_table(oos_results)
    degradation_table = PerformanceTables.build_degradation_table(oos_results)
    cost_impact_table = PerformanceTables.build_cost_impact_table(
        cost_results,
        oos_results,
    )
    print(PerformanceTables.flag_report(summary_table, degradation_table))

    return {
        "train_df": train_df,
        "test_df": test_df,
        "frozen_params": frozen_params,
        "base_vwap": combined_vwap,
        "train_vwap": train_artifacts["train_vwap"],
        "test_vwap": test_artifacts["test_vwap"],
        "rolling_stability": train_artifacts["rolling_stability"],
        "ndev_map": train_artifacts["ndev_map"],
        "adf_results": train_artifacts["adf_results"],
        "hl_results": train_artifacts["hl_results"],
        "convergence_summary": train_artifacts["convergence_summary"],
        "convergence_detail": train_artifacts["convergence_detail"],
        "diagnostics": train_artifacts["diagnostics"],
        "eligible_set": train_artifacts["eligible_set"],
        "scores": train_artifacts["scores"],
        "window_scores": train_artifacts["window_scores"],
        "atr_df": train_artifacts["atr_df"],
        "test_atr_df": test_artifacts["test_atr_df"],
        "low_vol_mask": train_artifacts["low_vol_mask"],
        "hurst_results": train_artifacts["hurst_results"],
        "vndev_map": train_artifacts["vndev_map"],
        "test_vndev_map": test_artifacts["test_vndev_map"],
        "atr_summary": train_artifacts["atr_summary"],
        "signal_gen": train_artifacts["signal_gen"],
        "signal_results": train_artifacts["signal_results"],
        "test_signal_results": test_artifacts["test_signal_results"],
        "sizer": train_artifacts["sizer"],
        "sized_results": train_artifacts["sized_results"],
        "pnl_results": train_artifacts["pnl_results"],
        "test_pnl_results": test_artifacts["test_pnl_results"],
        "metrics_df": metrics_df,
        "metrics_map": metrics_map,
        "trade_log_df": trade_log_df,
        "equity_curves": equity_curves,
        "drawdown_curves": drawdown_curves,
        "cross_summary": cross_summary,
        "cost_engine": train_artifacts["cost_engine"],
        "cost_results": cost_results,
        "oos_engine": OOSEngine(),
        "oos_results": oos_results,
        "wf_splits": wf_splits,
        "degradation_table": degradation_table,
        "summary_table": summary_table,
        "cost_impact_table": cost_impact_table,
    }
