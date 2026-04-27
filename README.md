# mean_harvester

A mean-harvesting research system for stable-but-volatile equities and futures.

## VWAP Mean-Reversion Workflow (Step 1 → Step 8)

Implemented in `math_generator/strategy/vwap_mean_reversion.py` with an executable example in `math_generator/run_vwap_research.py`.

1. **Acquire data** (`acquire_ohlcv`)  
   Pulls OHLCV bars from Yahoo Finance (5-minute interval supported), forward-fills missing values, and enforces clean OHLCV schema.

2. **Compute price proxy + rolling VWAP** (`compute_vwap_features`)  
   Uses `price_proxy = (High + Low + Close) / 3` and computes rolling VWAP over candidate windows.

3. **Build deviation series + stationarity diagnostics** (`analyze_windows`)  
   Creates `NDEV_w = (price_proxy - VWAP_w)/VWAP_w`, runs ADF p-values, AR(1) `phi`, and half-life.

4. **Rank windows statistically** (`rank_windows`)  
   Produces a ranked diagnostics table with the composite score:
   `0.4*(1-adf_p) + 0.3*(1/half_life) + 0.3*convergence_speed`.

5. **Add regime filters** (`detect_regimes`)  
   Computes ATR-based volatility regime and rolling Hurst exponent, flagging trending states.

6. **Define trading logic + backtest** (`backtest_mean_reversion`)  
   Entry on ±z-score threshold, exit on mean reversion / stop, no-lookahead position shifting, transaction costs, Sharpe, and drawdown.

7. **Optimize/validate** (`walk_forward_validate`)  
   70/30 train/test walk-forward over top-ranked windows with out-of-sample metrics.

8. **Deploy (paper trading scaffold)**  
   Repository currently provides research/backtest functions. For live deployment, integrate the same signal state machine with broker APIs (e.g., Kite Connect) and an asyncio event loop.

---

## Quick Start

```bash
cd math_generator
python -m pip install -r requirements.txt
python run_vwap_research.py
```

This prints:
- Ranked VWAP window diagnostics.
- Best-window backtest statistics.
- Walk-forward validation table.

And plots:
- Equity curve.
- Drawdown.
- Window ranking heatmap.
