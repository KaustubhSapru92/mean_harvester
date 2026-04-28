import pandas as pd
from typing import Dict, Optional


class WindowDiagnostics:
    """
    Step 4: Ranks VWAP windows statistically using a composite score.

    Composite = 0.4 * (1 - ADF p-value)
              + 0.3 * (1 / half_life_bars, normalised)
              + 0.3 * convergence_speed

    Stages are additive — each stage enriches the internal state and
    the final ranked DataFrame is produced in Stage 4.
    """

    def __init__(self, adf_results: pd.DataFrame,
                 hl_results: pd.DataFrame,
                 convergence_summary: pd.DataFrame,
                 convergence_detail: dict[int, dict],
                 weights: Dict[str, float]
                 ):
        self.adf_results        = adf_results
        self.hl_results         = hl_results
        self.convergence_summary = convergence_summary
        self.convergence_detail  = convergence_detail
        self.weights: Dict[str, float] = weights or {"stat": 0.4, "hl": 0.3, "conv": 0.3}

        # Built progressively across stages
        self._scores: Optional[pd.DataFrame] = None

    @property
    def scores(self) -> pd.DataFrame:
        if self._scores is None:
            raise RuntimeError("Scores not computed yet.")
        return self._scores.copy()

    # ------------------------------------------------------------------
    # Stage 1 — Eligibility gate
    # ------------------------------------------------------------------

    def build_eligible_set(self) -> pd.DataFrame:
        """
        Applies three hard eligibility criteria and returns a DataFrame
        of eligible windows with their raw metric values attached.

        Eligibility requires ALL of:
          1. is_stationary == True       (passed ADF at p < 0.05)
          2. hl_results.valid == True    (phi in (0,1), half_life_bars not null)
          3. convergence_summary.converged == True

        Ineligible windows are included with eligible=False so the table
        is always complete across all windows.

        Returns
        -------
        DataFrame indexed by window with columns:
            p_value | half_life_bars | converged | eligible
        """
        all_windows = self.adf_results.index.tolist()
        rows = []

        for w in all_windows:
            is_stationary = bool(self.adf_results.loc[w, "is_stationary"])
            hl_valid      = bool(self.hl_results.loc[w, "valid"])
            converged     = bool(self.convergence_summary.loc[w, "converged"])

            eligible = is_stationary and hl_valid and converged

            rows.append({
                "window":         w,
                "p_value":        self.adf_results.loc[w, "p_value"],
                "half_life_bars": self.hl_results.loc[w, "half_life_bars"],
                "converged":      converged,
                "eligible":       eligible,
            })

        self._scores = pd.DataFrame(rows).set_index("window")
        return self._scores

    @staticmethod
    def eligible_report(scores: pd.DataFrame) -> str:
        """
        Human-readable eligibility summary for console output.
        """
        lines = []
        for w, row in scores.iterrows():
            tag = "ELIGIBLE" if row["eligible"] else "EXCLUDED"
            hl = f"{row['half_life_bars']:.1f}" if pd.notna(row["half_life_bars"]) else "—"
            p = f"{row['p_value']:.4f}" if pd.notna(row["p_value"]) else "—"
            lines.append(
                f"  W={w:>4}  [{tag:<8}]  p={p}  hl={hl} bars  converged={row['converged']}"
            )
        return "Eligibility Gate:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Stage 2 — Convergence speed derivation
    # ------------------------------------------------------------------

    def derive_convergence_speed(self) -> pd.DataFrame:
        """
        Derives a continuous convergence_speed score in [0, 1] for each
        eligible window by scanning its slice_fractions trace.

        Method
        ------
        For each eligible window, walk the slice_fractions list from
        convergence_detail and find the earliest fraction at which BOTH
        ADF p-value and half-life are simultaneously stable across all
        remaining slices.

        Stability at slice i means:
          - All p_values from index i onward stay within SPEED_STABLE_BAND
            of the value at index i (relative change < threshold)
          - Same check applied to half_lives

        convergence_speed = 1 - convergence_fraction
          → converges at 20% of history  →  speed = 0.80  (high)
          → converges at 90% of history  →  speed = 0.10  (low)

        Ineligible windows receive convergence_speed = 0.0.

        Returns
        -------
        self._scores updated in-place with column: convergence_speed
        Also returns self._scores for chaining.
        """
        if self._scores is None:
            raise RuntimeError("Call build_eligible_set() before derive_convergence_speed().")

        SPEED_STABLE_BAND = 0.10  # max relative change to be considered stable

        speeds = {}

        for w, row in self._scores.iterrows():
            if not row["eligible"]:
                speeds[w] = 0.0
                continue

            detail = self.convergence_detail.get(w)
            if detail is None:
                speeds[w] = 0.0
                continue

            fractions = detail["slice_fractions"]
            p_values = detail["p_values"]
            half_lives = detail["half_lives"]
            n = len(fractions)

            convergence_fraction = 1.0  # pessimistic default

            for i in range(n):
                p_tail = [v for v in p_values[i:] if v is not None]
                hl_tail = [v for v in half_lives[i:] if v is not None]

                if len(p_tail) < 2 or len(hl_tail) < 2:
                    continue

                # Check relative stability of both tails from index i
                p_base = abs(p_tail[0])
                hl_base = abs(hl_tail[0])

                if p_base == 0 or hl_base == 0:
                    continue

                p_stable = all(
                    abs(p_tail[j] - p_tail[j - 1]) / p_base < SPEED_STABLE_BAND
                    for j in range(1, len(p_tail))
                )
                hl_stable = all(
                    abs(hl_tail[j] - hl_tail[j - 1]) / hl_base < SPEED_STABLE_BAND
                    for j in range(1, len(hl_tail))
                )

                if p_stable and hl_stable:
                    convergence_fraction = fractions[i]
                    break

            speeds[w] = round(1.0 - convergence_fraction, 4)

        self._scores["convergence_speed"] = pd.Series(speeds)
        return self._scores

    @staticmethod
    def convergence_speed_report(scores: pd.DataFrame) -> str:
        """
        Human-readable convergence speed summary for console output.
        """
        lines = []
        for w, row in scores.iterrows():
            if not row["eligible"]:
                lines.append(f"  W={w:>4}  [EXCLUDED]  speed=—")
                continue
            speed = row["convergence_speed"]
            frac = round(1.0 - speed, 4)
            lines.append(
                f"  W={w:>4}  speed={speed:.4f}  "
                f"(stabilised at {frac:.0%} of history)"
            )
        return "Convergence Speed:\n" + "\n".join(lines)

    # Stage 3 — Sub-score computation
    def compute_subscores(self) -> pd.DataFrame:
        """
        Computes three normalised [0, 1] sub-scores for each eligible window.

        Sub-scores
        ----------
        stat_score  = 1 - p_value
            Lower ADF p → higher confidence in stationarity → higher score.

        hl_score    = (1 / half_life_bars), normalised across eligible windows
            Faster mean reversion → higher score.
            Normalised so the fastest-reverting eligible window scores 1.0.
            Ineligible windows score 0.0.

        conv_score  = convergence_speed (already in [0, 1] from Stage 2)
            Higher speed → stabilised earlier in history → higher score.

        Returns
        -------
        self._scores updated in-place with columns:
            stat_score | hl_score | conv_score
        Also returns self._scores for chaining.
        """
        if self._scores is None or "convergence_speed" not in self._scores.columns:
            raise RuntimeError(
                "Call build_eligible_set() and derive_convergence_speed() "
                "before compute_subscores()."
            )

        scores = self._scores

        # --- Stationarity score ---
        scores["stat_score"] = scores["p_value"].apply(
            lambda p: round(1.0 - float(p), 6) if pd.notna(p) else 0.0
        )

        # --- Half-life score (normalised inverse) ---
        eligible_mask = scores["eligible"]
        hl_vals       = scores.loc[eligible_mask, "half_life_bars"]
        valid_hl      = hl_vals.dropna()

        if not valid_hl.empty:
            inv_hl     = 1.0 / valid_hl
            hl_min     = inv_hl.min()
            hl_max     = inv_hl.max()
            hl_range   = hl_max - hl_min

            def _normalise_hl(w):
                if not scores.loc[w, "eligible"]:
                    return 0.0
                hl = scores.loc[w, "half_life_bars"]
                if pd.isna(hl) or hl <= 0:
                    return 0.0
                inv = 1.0 / hl
                if hl_range == 0:
                    return 1.0          # all eligible windows have identical hl
                return round((inv - hl_min) / hl_range, 6)

            scores["hl_score"] = pd.Series(
                {w: _normalise_hl(w) for w in scores.index}
            )
        else:
            scores["hl_score"] = 0.0

        # --- Convergence score (direct from Stage 2) ---
        scores["conv_score"] = scores["convergence_speed"]

        self._scores = scores
        return self._scores

    @staticmethod
    def subscores_report(scores: pd.DataFrame) -> str:
        """
        Human-readable sub-score summary for console output.
        """
        lines = []
        for w, row in scores.iterrows():
            if not row["eligible"]:
                lines.append(f"  W={w:>4}  [EXCLUDED]")
                continue
            lines.append(
                f"  W={w:>4}  "
                f"stat={row['stat_score']:.4f}  "
                f"hl={row['hl_score']:.4f}  "
                f"conv={row['conv_score']:.4f}"
            )
        return "Sub-Scores:\n" + "\n".join(lines)

    # Stage 4 — Composite score and ranking

    def compute_composite(self) -> pd.DataFrame:
        """
        Computes the weighted composite score and assigns ranks.

        Formula
        -------
        composite = 0.4 * stat_score
                  + 0.3 * hl_score
                  + 0.3 * conv_score

        Ineligible windows receive composite = 0.0 and are ranked last.
        Rank 1 = highest composite score.

        Returns
        -------
        self._scores updated in-place with columns:
            composite | rank
        Also returns self._scores for chaining.
        """
        if self._scores is None or "stat_score" not in self._scores.columns:
            raise RuntimeError( "Call compute_subscores() before compute_composite().")

        w = self.weights
        scores = self._scores.copy()

        self._scores["composite"] = (
            w["stat"] * scores["stat_score"]
            + w["hl"]   * scores["hl_score"]
            + w["conv"] * scores["conv_score"]
        ).round(6)

        # Ineligible windows forced to 0.0 — not left at whatever
        # arithmetic produces from null sub-scores
        self._scores.loc[~self._scores["eligible"], "composite"] = 0.0

        # Rank: 1 = best. Eligible windows ranked among themselves first,
        # ineligible windows all share the bottom rank band.
        self._scores["rank"] = (
            self._scores["composite"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        return self._scores

    @staticmethod
    def ranking_report(scores: pd.DataFrame) -> str:
        """
        Human-readable final ranking for console output.
        """
        ranked = scores.sort_values("rank")
        lines  = []

        for w, row in ranked.iterrows():
            if not row["eligible"]:
                lines.append(
                    f"  Rank {int(row['rank']):>2}  W={w:>4}  "
                    f"[EXCLUDED]  composite=0.0000"
                )
            else:
                lines.append(
                    f"  Rank {int(row['rank']):>2}  W={w:>4}  "
                    f"composite={row['composite']:.4f}  "
                    f"(stat={row['stat_score']:.3f}  "
                    f"hl={row['hl_score']:.3f}  "
                    f"conv={row['conv_score']:.3f})"
                )

        return "Final Window Ranking:\n" + "\n".join(lines)