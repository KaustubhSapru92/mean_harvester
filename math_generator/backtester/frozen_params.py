from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenStrategyParams:
    """
    Train-learned strategy settings that are applied unchanged to test data.
    """

    window: int
    half_life_bars: float
    entry_threshold: float
    stop_threshold: float
    kelly_capped: float
    atr_threshold: float
    atr_window: int
    hurst_H: float | None
    vol_norm_window: int
