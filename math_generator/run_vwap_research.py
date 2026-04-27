"""End-to-end VWAP mean-reversion research workflow (Steps 1-8 scaffold)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from strategy.vwap_mean_reversion import (
    acquire_ohlcv,
    analyze_windows,
    backtest_mean_reversion,
    compute_vwap_features,
    detect_regimes,
    rank_windows,
    walk_forward_validate,
)


def main() -> None:
    symbol = "RELIANCE.NS"
    windows = list(range(20, 201, 20))

    bars = acquire_ohlcv(symbol=symbol, interval="5m", period="60d")
    features = compute_vwap_features(bars, windows=windows)
    diagnostics = analyze_windows(features, windows=windows)
    ranked = rank_windows(diagnostics)

    if ranked.empty:
        raise RuntimeError("No windows passed diagnostics. Try more history.")

    best_window = int(ranked.iloc[0]["window"])
    regime_df = detect_regimes(features)
    bt = backtest_mean_reversion(regime_df, window=best_window, transaction_cost=0.001)
    wf = walk_forward_validate(regime_df, ranked_windows=ranked, train_fraction=0.7, top_k=3)

    print("\n=== Ranked Windows ===")
    print(ranked.to_string(index=False))

    print("\n=== Backtest (best window) ===")
    print(f"Window={best_window} | Sharpe={bt.sharpe:.3f} | Return={bt.total_return:.3%} | MDD={bt.max_drawdown:.3%}")

    print("\n=== Walk Forward ===")
    print(wf.to_string(index=False))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    axes[0].plot(bt.equity_curve.index, bt.equity_curve.values)
    axes[0].set_title(f"Equity Curve ({symbol})")
    axes[0].grid(True)

    drawdown = bt.equity_curve / bt.equity_curve.cummax() - 1
    axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.4)
    axes[1].set_title("Drawdown")
    axes[1].grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    sns.heatmap(
        ranked.set_index("window")[["composite_score", "adf_p_value", "half_life", "convergence_speed"]],
        annot=True,
        fmt=".3f",
        cmap="viridis",
    )
    plt.title("Window Ranking Heatmap")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
