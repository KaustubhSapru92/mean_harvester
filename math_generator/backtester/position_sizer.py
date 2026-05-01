import numpy as np
import pandas as pd


class PositionSizer:
    """
    Step 6 — Stage 2.

    Computes Kelly-criterion position sizes from the signal series and
    applies them to produce a scaled position series for each window.

    Kelly formula:
        f* = (p * b - q) / b

        where:
            p = win rate (fraction of trades that are profitable)
            q = 1 - p   (loss rate)
            b = average win / average loss  (win/loss ratio)

    Half-Kelly is applied (f* / 2) to reduce variance and account for
    estimation error in p and b from limited trade history.

    The resulting fraction is capped at max_position_pct to prevent
    over-leverage regardless of what Kelly produces.

    Position series:
        position_t = kelly_fraction * signal_t

    where signal_t ∈ {-1, 0, +1} from Stage 1.

    Edge cases:
        - Fewer than 10 trades: Kelly not reliable → fallback to
          fixed fraction (max_position_pct / 2)
        - Zero losses (100% win rate): cap at max_position_pct
        - Negative Kelly (edge < 0): fallback to 0 (no position)
    """

    MIN_TRADES_FOR_KELLY = 10
    FALLBACK_FRACTION    = None   # set to max_position_pct / 2 at runtime

    def __init__(
        self,
        max_position_pct: float = 0.25,
    ):
        if not (0 < max_position_pct <= 1.0):
            raise ValueError(
                f"max_position_pct must be in (0, 1], got {max_position_pct}"
            )
        self.max_position_pct = max_position_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_kelly(
        self,
        trade_log: list[dict],
        bar_returns: pd.Series,
        signal: pd.Series,
    ) -> dict:
        """
        Derive Kelly fraction from completed trade history.

        PnL per trade is approximated as:
            trade_pnl = direction * sum(bar_returns[entry_bar:exit_bar])

        This is a first-order approximation — close enough for Kelly
        estimation without requiring price data per trade.

        Parameters
        ----------
        trade_log   : list of trade dicts from SignalGenerator
        bar_returns : Close-to-Close returns Series aligned to signal index
        signal      : signal Series from Stage 1

        Returns
        -------
        dict with keys:
            win_rate, win_loss_ratio, kelly_full, kelly_half,
            kelly_capped, n_trades, n_wins, n_losses, fallback_used
        """
        fallback_fraction = self.max_position_pct / 2.0

        if len(trade_log) < self.MIN_TRADES_FOR_KELLY:
            return {
                "win_rate":      None,
                "win_loss_ratio": None,
                "kelly_full":    None,
                "kelly_half":    None,
                "kelly_capped":  fallback_fraction,
                "n_trades":      len(trade_log),
                "n_wins":        None,
                "n_losses":      None,
                "fallback_used": True,
                "fallback_reason": f"fewer than {self.MIN_TRADES_FOR_KELLY} trades",
            }

        # Compute per-trade PnL from bar returns
        trade_pnls = []
        for trade in trade_log:
            try:
                entry = trade["entry_bar"]
                exit_ = trade["exit_bar"]
                direc = trade["direction"]
                mask  = (bar_returns.index >= entry) & (bar_returns.index <= exit_)
                pnl   = direc * bar_returns[mask].sum()
                trade_pnls.append(float(pnl))
            except Exception:
                continue

        if not trade_pnls:
            return {
                "win_rate":      None,
                "win_loss_ratio": None,
                "kelly_full":    None,
                "kelly_half":    None,
                "kelly_capped":  fallback_fraction,
                "n_trades":      len(trade_log),
                "n_wins":        None,
                "n_losses":      None,
                "fallback_used": True,
                "fallback_reason": "could not compute trade PnLs",
            }

        wins   = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]

        n_wins   = len(wins)
        n_losses = len(losses)
        n_total  = len(trade_pnls)

        p = n_wins / n_total
        q = 1.0 - p

        # Win/loss ratio — handle zero-loss edge case
        if n_losses == 0:
            # All wins — cap at max
            return {
                "win_rate":       round(p, 4),
                "win_loss_ratio": None,
                "kelly_full":     1.0,
                "kelly_half":     self.max_position_pct,
                "kelly_capped":   self.max_position_pct,
                "n_trades":       n_total,
                "n_wins":         n_wins,
                "n_losses":       0,
                "fallback_used":  False,
                "fallback_reason": None,
            }

        avg_win  = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss == 0:
            b = 1.0
        else:
            b = avg_win / avg_loss

        kelly_full = (p * b - q) / b

        # Negative Kelly → no edge → return zero
        if kelly_full <= 0:
            return {
                "win_rate":       round(p, 4),
                "win_loss_ratio": round(b, 4),
                "kelly_full":     round(kelly_full, 6),
                "kelly_half":     0.0,
                "kelly_capped":   0.0,
                "n_trades":       n_total,
                "n_wins":         n_wins,
                "n_losses":       n_losses,
                "fallback_used":  False,
                "fallback_reason": "negative Kelly — no edge detected",
            }

        kelly_half   = kelly_full / 2.0
        kelly_capped = min(kelly_half, self.max_position_pct)

        return {
            "win_rate":       round(p, 4),
            "win_loss_ratio": round(b, 4),
            "kelly_full":     round(kelly_full, 6),
            "kelly_half":     round(kelly_half, 6),
            "kelly_capped":   round(kelly_capped, 6),
            "n_trades":       n_total,
            "n_wins":         n_wins,
            "n_losses":       n_losses,
            "fallback_used":  False,
            "fallback_reason": None,
        }

    def scale_positions(
        self,
        signal: pd.Series,
        kelly_capped: float,
    ) -> pd.Series:
        """
        Apply Kelly fraction to signal to produce scaled position series.

        position_t = kelly_capped * signal_t

        Parameters
        ----------
        signal       : pd.Series[int] from Stage 1
        kelly_capped : capped half-Kelly fraction from compute_kelly()

        Returns
        -------
        pd.Series of float in [-max_position_pct, +max_position_pct]
        """
        position = signal.astype(float) * kelly_capped
        position.name = signal.name.replace("signal_", "position_")
        return position

    def size_all(
        self,
        signal_results: dict[int, dict],
        base_vwap: pd.DataFrame,
    ) -> dict[int, dict]:
        """
        Compute Kelly sizing for all windows and return enriched results.

        Parameters
        ----------
        signal_results : { window: { signal, trade_log } } from Stage 1
        base_vwap      : OHLCV DataFrame — used to compute bar returns

        Returns
        -------
        { window: { signal, trade_log, position, kelly_meta } }
        """
        bar_returns = base_vwap["Close"].pct_change().fillna(0.0)

        sized = {}

        for w, res in signal_results.items():
            signal    = res["signal"]
            trade_log = res["trade_log"]

            kelly_meta   = self.compute_kelly(trade_log, bar_returns, signal)
            kelly_capped = kelly_meta["kelly_capped"]
            position     = self.scale_positions(signal, kelly_capped)

            sized[w] = {
                "signal":     signal,
                "trade_log":  trade_log,
                "position":   position,
                "kelly_meta": kelly_meta,
            }

        return sized

    @staticmethod
    def kelly_report(sized_results: dict[int, dict]) -> str:
        """
        Human-readable Kelly sizing summary per window.
        """
        lines = ["Kelly Position Sizing:"]

        for w, res in sized_results.items():
            km = res["kelly_meta"]
            if km["fallback_used"]:
                lines.append(
                    f"  W={w:>4}  [FALLBACK]  "
                    f"fraction={km['kelly_capped']:.4f}  "
                    f"reason={km['fallback_reason']}"
                )
            elif km["kelly_capped"] == 0.0:
                lines.append(
                    f"  W={w:>4}  [NO EDGE ]  "
                    f"p={km['win_rate']:.3f}  "
                    f"b={km['win_loss_ratio']:.3f}  "
                    f"kelly={km['kelly_full']:.4f}"
                )
            else:
                lines.append(
                    f"  W={w:>4}  [KELLY   ]  "
                    f"p={km['win_rate']:.3f}  "
                    f"b={km['win_loss_ratio']:.3f}  "
                    f"f*={km['kelly_full']:.4f}  "
                    f"half={km['kelly_half']:.4f}  "
                    f"capped={km['kelly_capped']:.4f}  "
                    f"trades={km['n_trades']}"
                )

        return "\n".join(lines)