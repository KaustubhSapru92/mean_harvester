import numpy as np
import pandas as pd


class SignalGenerator:
    """
    Step 6 — Stage 1.

    Generates entry, exit, and stop-loss signals from VNDEV series.

    Rules (evaluated in priority order each bar):
      1. STOP   — if |VNDEV| >= stop_threshold while in a trade → exit immediately
      2. EXIT   — if VNDEV crosses zero while in a trade → exit
      3. TIMEOUT— if bars_held >= half_life_bars while in a trade → exit
      4. ENTRY  — if flat and VNDEV crosses below -entry_threshold → long (+1)
                  if flat and VNDEV crosses above +entry_threshold → short (-1)

    Mean-reversion direction convention:
      VNDEV < -entry_threshold → price below VWAP → expect upward reversion → LONG
      VNDEV > +entry_threshold → price above VWAP → expect downward reversion → SHORT

    Output
    ------
    signal   : pd.Series of int  {-1, 0, +1}  — position direction per bar
    position : pd.Series of int  {-1, 0, +1}  — same as signal here,
               Kelly scaling applied in Stage 2
    trade_log: list of dicts, one entry per completed trade
    """

    def __init__(
        self,
        entry_threshold: float = 2.0,
        stop_threshold: float  = 3.0,
    ):
        if stop_threshold <= entry_threshold:
            raise ValueError(
                f"stop_threshold ({stop_threshold}) must be > "
                f"entry_threshold ({entry_threshold})"
            )
        self.entry_threshold = entry_threshold
        self.stop_threshold  = stop_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        vndev: pd.Series,
        half_life_bars: float,
        window: int,
    ) -> dict:
        """
        Generate signal series and trade log for one VNDEV series.

        Parameters
        ----------
        vndev          : VNDEV series (NaN on high-vol bars)
        half_life_bars : AR(1) half-life from window_scores (bars)
        window         : VWAP window integer (for labelling only)

        Returns
        -------
        dict with keys:
            signal     : pd.Series[int]   — bar-by-bar position direction
            trade_log  : list[dict]        — one row per completed trade
        """
        half_life = int(np.ceil(half_life_bars))
        values    = vndev.values
        index     = vndev.index
        n         = len(values)

        signal    = np.zeros(n, dtype=int)

        # Trade state
        in_trade      = False
        direction     = 0
        entry_idx     = 0
        entry_vndev   = 0.0
        bars_held     = 0

        trade_log = []

        for i in range(1, n):
            v     = values[i]
            v_lag = values[i - 1]

            # Skip NaN bars — carry flat position, don't count as held
            if np.isnan(v) or np.isnan(v_lag):
                signal[i] = 0
                if in_trade:
                    # Close any open trade at NaN boundary cleanly
                    trade_log.append(self._close_trade(
                        window, direction, entry_idx, i,
                        entry_vndev, v_lag, bars_held,
                        "nan_boundary", index
                    ))
                    in_trade  = False
                    direction = 0
                continue

            if in_trade:
                bars_held += 1

                # Priority 1: stop-loss
                if abs(v) >= self.stop_threshold:
                    trade_log.append(self._close_trade(
                        window, direction, entry_idx, i,
                        entry_vndev, v, bars_held,
                        "stop_loss", index
                    ))
                    in_trade  = False
                    direction = 0
                    signal[i] = 0
                    continue

                # Priority 2: zero-cross exit
                zero_cross = (v_lag > 0 and v <= 0) or (v_lag < 0 and v >= 0)
                if zero_cross:
                    trade_log.append(self._close_trade(
                        window, direction, entry_idx, i,
                        entry_vndev, v, bars_held,
                        "reversion", index
                    ))
                    in_trade  = False
                    direction = 0
                    signal[i] = 0
                    continue

                # Priority 3: half-life timeout
                if bars_held >= half_life:
                    trade_log.append(self._close_trade(
                        window, direction, entry_idx, i,
                        entry_vndev, v, bars_held,
                        "timeout", index
                    ))
                    in_trade  = False
                    direction = 0
                    signal[i] = 0
                    continue

                # Still in trade
                signal[i] = direction

            else:
                # Entry logic — check threshold cross from previous bar
                long_entry  = v_lag > -self.entry_threshold and v <= -self.entry_threshold
                short_entry = v_lag < +self.entry_threshold and v >= +self.entry_threshold

                if long_entry:
                    in_trade    = True
                    direction   = +1
                    entry_idx   = i
                    entry_vndev = v
                    bars_held   = 0
                    signal[i]   = +1

                elif short_entry:
                    in_trade    = True
                    direction   = -1
                    entry_idx   = i
                    entry_vndev = v
                    bars_held   = 0
                    signal[i]   = -1

                else:
                    signal[i] = 0

        # Close any trade still open at end of series
        if in_trade:
            trade_log.append(self._close_trade(
                window, direction, entry_idx, n - 1,
                entry_vndev, values[n - 1], bars_held,
                "end_of_series", index
            ))

        return {
            "signal":    pd.Series(signal, index=index, name=f"signal_{window}"),
            "trade_log": trade_log,
        }

    def generate_all(
        self,
        vndev_map: dict[int, pd.Series],
        window_scores: pd.DataFrame,
    ) -> dict[int, dict]:
        """
        Generate signals for all regime_pass windows.

        Parameters
        ----------
        vndev_map     : { window: pd.Series } from VolNormalizer
        window_scores : enriched DataFrame from vwap_pipeline

        Returns
        -------
        { window: { signal, trade_log } }
        """
        results = {}

        for w, vndev in vndev_map.items():
            if w not in window_scores.index:
                continue
            row = window_scores.loc[w]
            if not bool(row.get("regime_pass", False)):
                continue
            if pd.isna(row.get("half_life_bars")):
                continue

            results[w] = self.generate(
                vndev          = vndev,
                half_life_bars = float(row["half_life_bars"]),
                window         = w,
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _close_trade(
        window: int,
        direction: int,
        entry_idx: int,
        exit_idx: int,
        entry_vndev: float,
        exit_vndev: float,
        bars_held: int,
        exit_reason: str,
        index: pd.Index,
    ) -> dict:
        return {
            "window":      window,
            "direction":   direction,
            "entry_bar":   index[entry_idx],
            "exit_bar":    index[exit_idx],
            "entry_vndev": round(float(entry_vndev), 4),
            "exit_vndev":  round(float(exit_vndev),  4),
            "bars_held":   bars_held,
            "exit_reason": exit_reason,
        }

    @staticmethod
    def trade_log_df(results: dict[int, dict]) -> pd.DataFrame:
        """
        Flatten all per-window trade logs into a single DataFrame.
        Useful for cross-window analysis in Stage 4.
        """
        rows = []
        for w, res in results.items():
            rows.extend(res["trade_log"])
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("entry_bar").reset_index(drop=True)

    @staticmethod
    def flag_report(results: dict[int, dict]) -> str:
        """
        Human-readable signal summary per window.
        """
        lines = ["Signal Generation:"]
        for w, res in results.items():
            log    = res["trade_log"]
            n      = len(log)
            stops  = sum(1 for t in log if t["exit_reason"] == "stop_loss")
            rev    = sum(1 for t in log if t["exit_reason"] == "reversion")
            tout   = sum(1 for t in log if t["exit_reason"] == "timeout")
            lines.append(
                f"  W={w:>4}  trades={n:>4}  "
                f"reversion={rev}  timeout={tout}  stop={stops}"
            )
        return "\n".join(lines)