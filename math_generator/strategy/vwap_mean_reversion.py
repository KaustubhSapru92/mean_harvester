from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller


@dataclass
class WindowDiagnostics:
    window: int
    adf_p_value: float
    phi: float
    half_life: float
    convergence_speed: float
    variance_stability: float
    composite_score: float


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    returns: pd.Series
    sharpe: float
    max_drawdown: float
    total_return: float


def acquire_ohlcv(
    symbol: str,
    interval: str = "5m",
    period: str = "60d",
) -> pd.DataFrame:
    """Step 1: fetch and clean OHLCV data."""
    df = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean = df[required].copy()
    clean = clean.dropna(how="all").ffill().dropna()
    clean["Volume"] = clean["Volume"].clip(lower=0)

    return clean


def compute_vwap_features(df: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    """Step 2/3: price proxy, rolling VWAP, normalized deviations."""
    out = df.copy()
    out["price_proxy"] = (out["High"] + out["Low"] + out["Close"]) / 3.0

    for w in windows:
        numerator = (out["price_proxy"] * out["Volume"]).rolling(w).sum()
        denominator = out["Volume"].rolling(w).sum()
        out[f"VWAP_{w}"] = numerator / denominator.replace(0, np.nan)
        out[f"NDEV_{w}"] = (out["price_proxy"] - out[f"VWAP_{w}"]) / out[f"VWAP_{w}"]

    return out


def _ar1_phi(series: pd.Series) -> float:
    aligned = pd.concat([series.shift(1), series], axis=1).dropna()
    if aligned.empty:
        return np.nan
    x = aligned.iloc[:, 0].values
    y = aligned.iloc[:, 1].values
    denom = np.dot(x, x)
    if denom == 0:
        return np.nan
    return float(np.dot(x, y) / denom)


def _half_life(phi: float) -> float:
    if np.isnan(phi) or phi <= 0 or phi >= 1:
        return np.inf
    return float(-np.log(2.0) / np.log(phi))


def _hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    clean = series.dropna().values
    if len(clean) <= max_lag + 2:
        return np.nan

    lags = np.arange(2, max_lag)
    tau = []
    for lag in lags:
        std = np.std(clean[lag:] - clean[:-lag])
        if std > 0:
            tau.append((lag, np.sqrt(std)))

    if len(tau) < 2:
        return np.nan

    lags_arr = np.log([t[0] for t in tau])
    tau_arr = np.log([t[1] for t in tau])
    slope, _ = np.polyfit(lags_arr, tau_arr, 1)
    return float(slope * 2.0)


def analyze_windows(df: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    """Step 3/4: ADF, AR(1) half-life, and convergence diagnostics."""
    rows: list[WindowDiagnostics] = []

    for w in windows:
        series = df[f"NDEV_{w}"].dropna()
        if len(series) < 50:
            continue

        adf_p = float(adfuller(series, autolag="AIC")[1])
        phi = _ar1_phi(series)
        half_life = _half_life(phi)

        prefix_points = np.linspace(0.3, 1.0, 8)
        hl_prefix = []
        var_prefix = []
        for p in prefix_points:
            n = int(len(series) * p)
            sub = series.iloc[:n]
            hl_prefix.append(_half_life(_ar1_phi(sub)))
            var_prefix.append(float(sub.var()))

        hl_series = pd.Series(hl_prefix).replace([np.inf, -np.inf], np.nan).dropna()
        var_series = pd.Series(var_prefix).dropna()

        convergence_speed = 0.0
        if len(hl_series) >= 3:
            rel = (hl_series.diff().abs() / hl_series.abs().replace(0, np.nan)).fillna(1)
            stable = rel < 0.1
            convergence_speed = float(stable.mean())

        variance_stability = 0.0
        if len(var_series) >= 3 and var_series.mean() > 0:
            variance_stability = float(1.0 / (1.0 + (var_series.std() / var_series.mean())))

        inv_half_life = 0.0 if np.isinf(half_life) or half_life <= 0 else 1.0 / half_life
        score = (0.4 * (1.0 - adf_p)) + (0.3 * inv_half_life) + (0.3 * convergence_speed)

        rows.append(
            WindowDiagnostics(
                window=w,
                adf_p_value=adf_p,
                phi=phi,
                half_life=half_life,
                convergence_speed=convergence_speed,
                variance_stability=variance_stability,
                composite_score=score,
            )
        )

    rank_df = pd.DataFrame([r.__dict__ for r in rows]).sort_values(
        "composite_score", ascending=False
    )
    return rank_df.reset_index(drop=True)


def rank_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4 helper for display-friendly ranking table."""
    out = df.copy()
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["rank", "window", "composite_score", "adf_p_value", "half_life", "convergence_speed", "variance_stability"]]


def detect_regimes(df: pd.DataFrame, atr_window: int = 20, hurst_window: int = 120) -> pd.DataFrame:
    """Step 5: volatility and Hurst-based trend/range filter."""
    out = df.copy()

    prev_close = out["Close"].shift(1)
    tr = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - prev_close).abs(),
            (out["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR"] = tr.rolling(atr_window).mean()
    out["regime_vol"] = out["ATR"] / out["Close"]

    out["hurst"] = (
        out["Close"]
        .rolling(hurst_window)
        .apply(lambda s: _hurst_exponent(pd.Series(s)), raw=False)
    )
    out["is_trending"] = (out["hurst"] > 0.5) | (out["regime_vol"] > out["regime_vol"].rolling(hurst_window).median())
    return out


def backtest_mean_reversion(
    df: pd.DataFrame,
    window: int,
    entry_z: float = 2.0,
    stop_z: float = 3.0,
    transaction_cost: float = 0.001,
) -> BacktestResult:
    """Step 6: vectorized PnL with no-lookahead signal execution."""
    out = df.copy()
    ndev_col = f"NDEV_{window}"
    if ndev_col not in out.columns:
        raise ValueError(f"{ndev_col} missing from dataframe")

    rolling_std = out[ndev_col].rolling(window).std()
    zscore = out[ndev_col] / rolling_std

    long_entry = (zscore < -entry_z) & (~out["is_trending"])
    short_entry = (zscore > entry_z) & (~out["is_trending"])
    mean_exit = zscore.abs() < 0.25
    stop_exit = zscore.abs() > stop_z

    raw_pos = pd.Series(0, index=out.index, dtype=float)
    raw_pos[long_entry] = 1.0
    raw_pos[short_entry] = -1.0
    raw_pos[(mean_exit | stop_exit)] = 0.0
    position = raw_pos.replace(0, np.nan).ffill().fillna(0.0)

    returns = out["Close"].pct_change().fillna(0.0)
    trade_flag = position.diff().abs().fillna(0.0)
    strategy_returns = (position.shift(1).fillna(0.0) * returns) - (trade_flag * transaction_cost)

    equity = (1.0 + strategy_returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0

    ann_factor = np.sqrt(252 * (6.5 * 60 / 5))
    ret_std = strategy_returns.std()
    sharpe = 0.0 if ret_std == 0 else float(strategy_returns.mean() / ret_std * ann_factor)

    trades = pd.DataFrame(
        {
            "position": position,
            "returns": strategy_returns,
            "zscore": zscore,
            "is_trending": out["is_trending"],
        }
    )

    return BacktestResult(
        trades=trades,
        equity_curve=equity,
        returns=strategy_returns,
        sharpe=sharpe,
        max_drawdown=float(drawdown.min()),
        total_return=float(equity.iloc[-1] - 1.0),
    )


def walk_forward_validate(
    df: pd.DataFrame,
    ranked_windows: pd.DataFrame,
    train_fraction: float = 0.7,
    top_k: int = 3,
) -> pd.DataFrame:
    """Step 7: train/test walk-forward on top-ranked windows."""
    if ranked_windows.empty:
        return pd.DataFrame()

    split_idx = int(len(df) * train_fraction)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    rows = []
    for _, row in ranked_windows.head(top_k).iterrows():
        w = int(row["window"])
        train_bt = backtest_mean_reversion(train_df, window=w)
        test_bt = backtest_mean_reversion(test_df, window=w)

        rows.append(
            {
                "window": w,
                "train_sharpe": train_bt.sharpe,
                "test_sharpe": test_bt.sharpe,
                "train_return": train_bt.total_return,
                "test_return": test_bt.total_return,
                "train_max_drawdown": train_bt.max_drawdown,
                "test_max_drawdown": test_bt.max_drawdown,
            }
        )

    return pd.DataFrame(rows).sort_values("test_sharpe", ascending=False).reset_index(drop=True)
