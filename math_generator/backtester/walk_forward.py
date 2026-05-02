import numpy as np
import pandas as pd


class WalkForwardSplitter:
    """
    Step 7 — Stage 1.

    Splits each window's VNDEV series and trade log chronologically
    into train (70%) and test (30%) segments.

    Rules:
        - Split is on bar index, not calendar time, so it respects
          actual signal density regardless of gaps in market hours.
        - The cut point is the last bar index at or before the 70% mark.
        - Train set is used to derive parameters (Kelly fraction).
        - Test set receives frozen parameters — no re-optimisation.
        - No shuffling. Future bars must never appear in train.

    Output per window:
        train_vndev  : pd.Series — VNDEV bars up to cut point
        test_vndev   : pd.Series — VNDEV bars from cut point onward
        train_trades : list[dict] — trades whose entry_bar falls in train
        test_trades  : list[dict] — trades whose entry_bar falls in test
        cut_bar      : timestamp of the split point
        n_train      : number of train bars
        n_test       : number of test bars
    """

    TRAIN_FRACTION = 0.70

    def __init__(self, train_fraction: float = 0.70):
        if not (0 < train_fraction < 1):
            raise ValueError(
                f"train_fraction must be in (0, 1), got {train_fraction}"
            )
        self.train_fraction = train_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_window(
        self,
        vndev: pd.Series,
        trade_log: list[dict],
        window: int,
    ) -> dict:
        """
        Split one window's VNDEV series and trade log.

        Parameters
        ----------
        vndev     : full VNDEV series for this window (NaN on high-vol bars)
        trade_log : list of trade dicts from SignalGenerator for this window
        window    : VWAP window integer (for labelling only)

        Returns
        -------
        dict with keys:
            window, cut_bar, n_train, n_test,
            train_vndev, test_vndev,
            train_trades, test_trades
        """
        # Use only non-NaN bars for the split so high-vol gaps
        # don't distort the 70% cut point
        valid_bars = vndev.dropna()
        n_valid    = len(valid_bars)

        if n_valid < 10:
            return {
                "window":       window,
                "cut_bar":      None,
                "n_train":      0,
                "n_test":       0,
                "train_vndev":  pd.Series(dtype=float),
                "test_vndev":   pd.Series(dtype=float),
                "train_trades": [],
                "test_trades":  [],
                "skip_reason":  f"insufficient valid bars ({n_valid} < 10)",
            }

        cut_idx  = int(np.floor(n_valid * self.train_fraction))
        cut_idx  = max(1, min(cut_idx, n_valid - 1))   # guard edges
        cut_bar  = valid_bars.index[cut_idx]

        # Split VNDEV on the full series (preserves NaN bars in each half)
        train_vndev = vndev[vndev.index < cut_bar]
        test_vndev  = vndev[vndev.index >= cut_bar]

        # Split trade log by entry_bar
        train_trades = [t for t in trade_log if t["entry_bar"] <  cut_bar]
        test_trades  = [t for t in trade_log if t["entry_bar"] >= cut_bar]

        return {
            "window":       window,
            "cut_bar":      cut_bar,
            "n_train":      int((~train_vndev.isna()).sum()),
            "n_test":       int((~test_vndev.isna()).sum()),
            "train_vndev":  train_vndev,
            "test_vndev":   test_vndev,
            "train_trades": train_trades,
            "test_trades":  test_trades,
            "skip_reason":  None,
        }

    def split_all(
        self,
        vndev_map: dict[int, pd.Series],
        signal_results: dict[int, dict],
    ) -> dict[int, dict]:
        """
        Split all windows.

        Parameters
        ----------
        vndev_map      : { window: pd.Series } from VolNormalizer
        signal_results : { window: { signal, trade_log } } from SignalGenerator

        Returns
        -------
        { window: split_dict }
        """
        splits = {}

        for w, vndev in vndev_map.items():
            trade_log = signal_results.get(w, {}).get("trade_log", [])
            splits[w] = self.split_window(vndev, trade_log, w)

        return splits

    @staticmethod
    def flag_report(splits: dict[int, dict]) -> str:
        """
        Human-readable split summary for console output.
        """
        lines = ["Walk-Forward Split:"]

        for w, s in splits.items():
            if s.get("skip_reason"):
                lines.append(
                    f"  W={w:>4}  [SKIP  ]  {s['skip_reason']}"
                )
            else:
                lines.append(
                    f"  W={w:>4}  [SPLIT ]  "
                    f"train={s['n_train']:>4} bars / {len(s['train_trades']):>3} trades  "
                    f"test={s['n_test']:>4} bars / {len(s['test_trades']):>3} trades  "
                    f"cut={s['cut_bar']}"
                )

        return "\n".join(lines)