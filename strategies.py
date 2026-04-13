"""
strategies.py
=============
Trading strategy implementations for the IMC Trading Challenge pipeline.

Strategies
----------
MeanReversionStrategy  : Fades z-score extremes (market-making / stat-arb style).
MomentumStrategy       : Follows momentum signal with trend filter.

Both share a common interface:
    generate_signals(features_df) -> pd.Series  (float signal in [-1, +1])

Risk controls applied *after* signal generation (in backtest.py):
  - product-level position limits
  - max gross exposure
  - optional stop-loss
  - inventory penalty to discourage runaway positions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MeanReversionParams:
    window: int = 40                   # z-score window
    entry_z: float = 1.5               # enter when |z| > entry_z
    exit_z: float = 0.3                # exit when |z| < exit_z
    position_limit: int = 10           # max abs inventory (units)
    max_gross_exposure: float = 20.0   # max sum(|inventory|) across all products
    stop_loss_pct: float = 0.02        # stop if unrealised loss > 2% of entry
    inventory_penalty: float = 0.001   # linear penalty per unit of inventory


@dataclass
class MomentumParams:
    window_fast: int = 20              # fast momentum window
    window_slow: int = 60              # slow momentum window
    entry_threshold: float = 0.001     # enter when |momentum| > threshold
    position_limit: int = 10
    max_gross_exposure: float = 20.0
    stop_loss_pct: float = 0.03
    inventory_penalty: float = 0.0005
    vol_scale: bool = True             # scale position size by inverse volatility


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseStrategy:
    """Shared helpers for all strategies."""

    name: str = "base"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Return a signal series in [-1, +1] aligned to df.index.
        Positive = long, Negative = short, 0 = flat.
        Must not use future information.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utility: apply hard position limit to a raw signal
    # ------------------------------------------------------------------
    @staticmethod
    def _clip_signal(signal: pd.Series) -> pd.Series:
        return signal.clip(-1.0, 1.0).fillna(0.0)


# ---------------------------------------------------------------------------
# Mean-Reversion Strategy
# ---------------------------------------------------------------------------

class MeanReversionStrategy(BaseStrategy):
    """
    Mean-Reversion Market-Making / Stat-Arb strategy.

    Logic
    -----
    - When z-score > +entry_z  → short (price is above fair value)
    - When z-score < -entry_z  → long  (price is below fair value)
    - Hold until z-score reverts past exit_z
    - An inventory penalty discourages holding large positions

    Why it may work in IMC microstructure
    --------------------------------------
    IMC challenge prices tend to mean-revert around a slowly drifting fair
    value because market makers continuously supply liquidity.  Short-term
    dislocations caused by noise traders or delayed information arrival create
    predictable return opportunities that the z-score detects.
    """

    name = "mean_reversion"

    def __init__(self, params: Optional[MeanReversionParams] = None):
        self.params = params or MeanReversionParams()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        z_col = f"zscore_{p.window}"

        if z_col not in df.columns:
            # Try any available z-score column
            z_cols = [c for c in df.columns if c.startswith("zscore_")]
            if not z_cols:
                return pd.Series(0.0, index=df.index, name="signal")
            z_col = z_cols[0]

        z = df[z_col].fillna(0.0)

        # State-machine signal generation (bar-by-bar).
        # 'current_position' persists across iterations to track the open inventory state.
        signals = np.zeros(len(df))
        current_position = 0.0

        for i in range(len(df)):
            zi = z.iloc[i]

            # Exit condition: z reverted past exit threshold
            if current_position > 0 and zi >= -p.exit_z:
                current_position = 0.0
            elif current_position < 0 and zi <= p.exit_z:
                current_position = 0.0

            # Entry condition
            if current_position == 0.0:
                if zi > p.entry_z:
                    current_position = -1.0   # short: price too high
                elif zi < -p.entry_z:
                    current_position = 1.0    # long: price too low

            # Apply inventory penalty: reduce signal linearly
            adj = current_position * max(0.0, 1.0 - p.inventory_penalty * abs(current_position))
            signals[i] = adj

        return self._clip_signal(pd.Series(signals, index=df.index, name="signal"))


# ---------------------------------------------------------------------------
# Momentum Strategy
# ---------------------------------------------------------------------------

class MomentumStrategy(BaseStrategy):
    """
    Trend-Following / Momentum strategy.

    Logic
    -----
    - Fast momentum (short window) as entry signal
    - Slow momentum (long window) as trend filter
    - Only enter if fast and slow signals agree
    - Optionally scale position size by inverse rolling volatility

    Why it may work in IMC microstructure
    --------------------------------------
    Some IMC products (e.g. commodity-like products) can exhibit short bursts
    of trending behaviour around new equilibrium prices driven by information
    arrival.  Momentum captures these episodes while the dual-window filter
    avoids noise-driven false signals during mean-reverting regimes.
    """

    name = "momentum"

    def __init__(self, params: Optional[MomentumParams] = None):
        self.params = params or MomentumParams()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        fast_col = f"momentum_{p.window_fast}"
        slow_col = f"momentum_{p.window_slow}"

        # Graceful fallback: use any available momentum column
        mom_cols = [c for c in df.columns if c.startswith("momentum_")]
        if fast_col not in df.columns:
            if not mom_cols:
                return pd.Series(0.0, index=df.index, name="signal")
            fast_col = mom_cols[0]
        if slow_col not in df.columns:
            slow_col = fast_col  # degenerate case: use same

        fast = df[fast_col].fillna(0.0)
        slow = df[slow_col].fillna(0.0)

        # Raw signal: sign of fast momentum, filtered by slow
        raw_signal = np.where(
            (fast > p.entry_threshold) & (slow > 0), 1.0,
            np.where(
                (fast < -p.entry_threshold) & (slow < 0), -1.0,
                0.0
            )
        )

        # Volatility scaling
        if p.vol_scale:
            rvol_cols = [c for c in df.columns if c.startswith("rvol_")]
            if rvol_cols:
                rvol = df[rvol_cols[0]].ffill().fillna(1e-6)
                rvol = rvol.replace(0, 1e-6)
                target_vol = 0.01  # target 1% daily vol exposure
                size = (target_vol / rvol).clip(0.0, 1.0)
                raw_signal = raw_signal * size.values

        signal = pd.Series(raw_signal, index=df.index, name="signal")
        return self._clip_signal(signal)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY = {
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
}


def get_strategy(name: str, params=None) -> BaseStrategy:
    """Instantiate a strategy by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY)}")
    cls = STRATEGY_REGISTRY[name]
    if params is not None:
        return cls(params)
    return cls()
