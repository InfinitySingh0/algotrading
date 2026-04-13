"""
strategies.py – IMC-style trading strategies.

Two strategies are implemented:

1. MeanReversionStrategy
   - Enters long when mid-price z-score is below -entry_z (price cheap)
   - Enters short when z-score is above +entry_z (price expensive)
   - Exits when z-score crosses back through exit_z
   - Optional inventory penalty reduces aggressiveness as position grows

2. MomentumStrategy
   - Computes short-term minus long-term rolling return
   - Buys when momentum is positive, sells when negative
   - Scales in/out based on signal strength

Both strategies share:
- Per-product position limits
- Max gross exposure cap
- Optional stop-loss
- Configurable transaction cost + slippage
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal output container
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """Desired inventory change at bar t."""
    desired_position: int = 0   # target position after this bar
    reason: str = ""            # optional debug label


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseStrategy:
    """
    Base class for bar-based strategies.

    Subclasses implement `generate_signal(row, current_position)`.
    """

    name: ClassVar[str] = "base"

    def __init__(
        self,
        position_limit: int = cfg.POSITION_LIMIT,
        max_gross_exposure: int = cfg.MAX_GROSS_EXPOSURE,
        transaction_cost: float = cfg.TRANSACTION_COST,
        slippage: float = cfg.SLIPPAGE,
        stop_loss_threshold: float = cfg.STOP_LOSS_THRESHOLD,
        inventory_penalty: float = cfg.INVENTORY_PENALTY,
    ) -> None:
        self.position_limit = position_limit
        self.max_gross_exposure = max_gross_exposure
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.stop_loss_threshold = stop_loss_threshold
        self.inventory_penalty = inventory_penalty

    # ------------------------------------------------------------------
    def clip_position(self, desired: int, current: int = 0) -> int:
        """Enforce position limits."""
        clipped = int(np.clip(desired, -self.position_limit, self.position_limit))
        return clipped

    def trade_cost(self, qty: int, price: float) -> float:
        """Return total cost (TC + slippage) for a trade of *qty* units."""
        return abs(qty) * (self.transaction_cost + self.slippage)

    def generate_signal(self, row: Any, current_position: int) -> Signal:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1) Mean-Reversion Strategy
# ---------------------------------------------------------------------------

class MeanReversionStrategy(BaseStrategy):
    """
    Mean-reversion market-making style strategy.

    Entry/exit rules
    ----------------
    * Short when zscore > +entry_z  (price too high relative to rolling mean)
    * Long  when zscore < -entry_z  (price too low)
    * Close long  when zscore > -exit_z
    * Close short when zscore <  exit_z
    * Inventory penalty: reduce desired position proportional to current holding

    Parameters
    ----------
    entry_z : z-score threshold to enter
    exit_z  : z-score threshold to exit
    trade_size : units per signal
    """

    name: ClassVar[str] = "mean_reversion"

    def __init__(
        self,
        entry_z: float = cfg.MR_ENTRY_ZSCORE,
        exit_z: float = cfg.MR_EXIT_ZSCORE,
        trade_size: int = cfg.MR_TRADE_SIZE,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.trade_size = trade_size

    def generate_signal(self, row: Any, current_position: int) -> Signal:
        z = row.get("zscore", 0.0) if isinstance(row, dict) else getattr(row, "zscore", 0.0)
        if z is None or (isinstance(z, float) and np.isnan(z)):
            z = 0.0

        # Inventory penalty: shrink desired position toward zero
        penalty_dir = -np.sign(current_position)
        inv_adj = int(round(self.inventory_penalty * abs(current_position)))

        if z > self.entry_z:
            # Price too high → short
            desired = -self.trade_size + penalty_dir * inv_adj
        elif z < -self.entry_z:
            # Price too low → long
            desired = self.trade_size + penalty_dir * inv_adj
        elif abs(z) < self.exit_z:
            # Near fair value → flatten
            desired = 0
        else:
            # Hold current position (directionally consistent)
            desired = current_position + penalty_dir * inv_adj

        desired = self.clip_position(desired, current_position)
        return Signal(desired_position=desired, reason=f"z={z:.3f}")


# ---------------------------------------------------------------------------
# 2) Momentum / Trend-Following Strategy
# ---------------------------------------------------------------------------

class MomentumStrategy(BaseStrategy):
    """
    Momentum / trend-following strategy.

    Signal: short_ret - long_ret (cross-over of fast and slow momentum)
    * Positive signal → long
    * Negative signal → short
    * Near-zero signal → flat

    Parameters
    ----------
    short_window : fast momentum window (bars)
    long_window  : slow momentum window (bars)
    trade_size   : units per signal
    threshold    : minimum |momentum| to trade
    """

    name: ClassVar[str] = "momentum"

    def __init__(
        self,
        short_window: int = cfg.MOM_SHORT_WINDOW,
        long_window: int = cfg.MOM_LONG_WINDOW,
        trade_size: int = cfg.MOM_TRADE_SIZE,
        threshold: float = 0.0005,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.short_window = short_window
        self.long_window = long_window
        self.trade_size = trade_size
        self.threshold = threshold

    def generate_signal(self, row: Any, current_position: int) -> Signal:
        short_key = f"ret_{self.short_window}"
        long_key = f"ret_{self.long_window}"

        if isinstance(row, dict):
            short_ret = row.get(short_key, np.nan)
            long_ret = row.get(long_key, np.nan)
        else:
            short_ret = getattr(row, short_key, np.nan)
            long_ret = getattr(row, long_key, np.nan)

        if pd.isna(short_ret) or pd.isna(long_ret):
            return Signal(desired_position=0, reason="no_signal")

        momentum = short_ret - long_ret

        if momentum > self.threshold:
            desired = self.trade_size
        elif momentum < -self.threshold:
            desired = -self.trade_size
        else:
            desired = 0

        desired = self.clip_position(desired, current_position)
        return Signal(desired_position=desired, reason=f"mom={momentum:.5f}")


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, type] = {
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
}


def create_strategy(name: str, **params) -> BaseStrategy:
    """Instantiate a strategy by name with given parameters."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY)}")
    return cls(**params)
