"""
metrics.py – Performance metrics for the IMC trading research pipeline.

All functions accept arrays / Series of per-bar PnL or equity and return
scalar metrics.  No pandas or external dependencies beyond numpy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(
    pnl: pd.Series,
    risk_free: float = 0.0,
    annualise_factor: float = 1.0,
) -> float:
    """
    Compute the Sharpe ratio of a per-bar PnL series.

    Parameters
    ----------
    pnl : per-bar realised PnL (not cumulative)
    risk_free : per-bar risk-free rate (default 0)
    annualise_factor : multiply result by this to annualise (e.g. sqrt(252))
    """
    excess = pnl - risk_free
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((excess.mean() / std) * annualise_factor)


def sortino_ratio(
    pnl: pd.Series,
    risk_free: float = 0.0,
    annualise_factor: float = 1.0,
) -> float:
    """Sortino ratio: use downside deviation instead of full std."""
    excess = pnl - risk_free
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float((excess.mean() / downside_std) * annualise_factor)


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    """
    Maximum drawdown of a cumulative-PnL series.

    Returns a non-positive float (e.g. -150.0 means 150 units peak-to-trough loss).
    """
    roll_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - roll_max
    return float(drawdown.min())


def drawdown_series(cumulative_pnl: pd.Series) -> pd.Series:
    """Return the full drawdown series."""
    roll_max = cumulative_pnl.cummax()
    return cumulative_pnl - roll_max


def hit_rate(pnl: pd.Series) -> float:
    """Fraction of bars with positive PnL."""
    nonzero = pnl[pnl != 0]
    if len(nonzero) == 0:
        return float("nan")
    return float((nonzero > 0).sum() / len(nonzero))


def total_pnl(pnl: pd.Series) -> float:
    return float(pnl.sum())


def turnover(trades: pd.Series) -> float:
    """Mean absolute position change per bar."""
    return float(trades.abs().mean())


def avg_trade_pnl(pnl: pd.Series, trades: pd.Series) -> float:
    """Average PnL on bars where a trade occurred."""
    traded_bars = pnl[trades.abs() > 0]
    if len(traded_bars) == 0:
        return float("nan")
    return float(traded_bars.mean())


# ---------------------------------------------------------------------------
# Summary dict
# ---------------------------------------------------------------------------

def compute_metrics(
    pnl: pd.Series,
    position: pd.Series,
    risk_free: float = 0.0,
) -> dict:
    """
    Compute a full set of performance metrics.

    Parameters
    ----------
    pnl : per-bar realised PnL
    position : per-bar inventory (position size)
    risk_free : per-bar risk-free rate

    Returns
    -------
    dict with keys: total_pnl, sharpe, sortino, max_dd, hit_rate, turnover,
                    avg_trade_pnl, n_trades
    """
    cum = pnl.cumsum()
    trades = position.diff().fillna(position.iloc[0] if len(position) else 0)

    n_trades = int((trades.abs() > 0).sum())

    return {
        "total_pnl": total_pnl(pnl),
        "sharpe": sharpe_ratio(pnl, risk_free=risk_free),
        "sortino": sortino_ratio(pnl, risk_free=risk_free),
        "max_drawdown": max_drawdown(cum),
        "hit_rate": hit_rate(pnl),
        "turnover": turnover(trades),
        "avg_trade_pnl": avg_trade_pnl(pnl, trades),
        "n_trades": n_trades,
    }
