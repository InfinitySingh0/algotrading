"""
backtest.py
===========
Bar-based backtesting engine for the IMC Trading Challenge pipeline.

Key features
------------
- Realistic friction: configurable transaction cost and slippage.
- Risk controls per bar: position limits, gross exposure cap, stop-loss.
- Metrics: realized/unrealized PnL, inventory, turnover, Sharpe, max-drawdown,
  hit rate.
- Time-based train/validation split.
- Grid/random parameter search ranked by risk-adjusted return.

Public API
----------
run_backtest(features_df, strategy, params) -> BacktestResult
parameter_search(features_df, strategy_cls, param_grid, ...) -> pd.DataFrame
"""

from __future__ import annotations

import itertools
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from strategies import BaseStrategy, MeanReversionParams, MomentumParams


# ---------------------------------------------------------------------------
# Friction / execution parameters
# ---------------------------------------------------------------------------

@dataclass
class ExecutionParams:
    transaction_cost_pct: float = 0.0005   # 0.05% of trade value
    slippage_pct: float = 0.0002           # 0.02% adverse slippage
    position_limit: int = 10               # max abs units per product
    max_gross_exposure: float = 20.0       # max sum(|pos|) across portfolio
    stop_loss_pct: float = 0.02            # stop-out if loss > 2% of cost basis


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    product: str
    strategy_name: str
    n_bars: int
    pnl_series: pd.Series              # cumulative realized PnL
    unrealized_series: pd.Series       # unrealized mark-to-market
    inventory_series: pd.Series        # inventory over time
    trade_log: pd.DataFrame            # individual fills
    metrics: Dict[str, float] = field(default_factory=dict)

    def compute_metrics(self) -> None:
        """Compute all performance metrics in-place."""
        pnl = self.pnl_series
        total_pnl = float(pnl.iloc[-1]) if len(pnl) else 0.0
        bar_pnl = pnl.diff().fillna(pnl.iloc[:1] if len(pnl) else pd.Series([0.0]))

        # Sharpe-like: mean / std of bar PnL (no annualisation; dimensionless)
        mean_bp = float(bar_pnl.mean()) if len(bar_pnl) else 0.0
        std_bp = float(bar_pnl.std()) if len(bar_pnl) > 1 else 1e-9
        sharpe = mean_bp / std_bp if std_bp > 0 else 0.0

        # Drawdown
        cum_max = pnl.cummax()
        dd = pnl - cum_max
        max_dd = float(dd.min()) if len(dd) else 0.0

        # Turnover
        trades = self.trade_log
        turnover = float(trades["quantity"].abs().sum()) if len(trades) else 0.0

        # Hit rate: fraction of trades that were profitable (using fill prices)
        if len(trades) > 0 and "pnl" in trades.columns:
            hit_rate = float((trades["pnl"] > 0).mean())
        else:
            hit_rate = float("nan")

        # Number of trades
        n_trades = len(trades)

        self.metrics = {
            "total_pnl": total_pnl,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "turnover": turnover,
            "n_trades": n_trades,
            "hit_rate": hit_rate,
            "mean_bar_pnl": mean_bp,
            "std_bar_pnl": std_bp,
        }


# ---------------------------------------------------------------------------
# Core backtester
# ---------------------------------------------------------------------------

def run_backtest(
    features_df: pd.DataFrame,
    strategy: BaseStrategy,
    exec_params: Optional[ExecutionParams] = None,
    price_col: str = "mid_price",
    product: str = "UNKNOWN",
) -> BacktestResult:
    """
    Run a bar-based backtest.

    Parameters
    ----------
    features_df : Feature DataFrame (single product, sorted by time).
    strategy    : Instantiated strategy.
    exec_params : Friction / risk-control parameters.
    price_col   : Column to use as fill price.
    product     : Product label for output.

    Returns
    -------
    BacktestResult
    """
    ep = exec_params or ExecutionParams()

    # --- Generate signals ---
    signals = strategy.generate_signals(features_df)

    if price_col not in features_df.columns:
        alt_cols = [c for c in ["mid_price", "price", "close"] if c in features_df.columns]
        if not alt_cols:
            warnings.warn(f"Price column '{price_col}' not found; skipping backtest.")
            empty = pd.Series(dtype=float)
            return BacktestResult(product, strategy.name, 0, empty, empty, empty, pd.DataFrame())
        price_col = alt_cols[0]

    prices = features_df[price_col].ffill().bfill()

    n = len(features_df)
    inventory = 0.0
    cash = 0.0              # running cash (negative when long, positive when short)
    cost_basis = 0.0        # weighted-average entry price for stop-loss tracking

    # Total equity at each bar = cash + inventory * price
    total_equity = np.zeros(n)
    unrealized_pnl = np.zeros(n)
    inventory_track = np.zeros(n)

    trade_records: List[dict] = []

    for i in range(n):
        sig = float(signals.iloc[i])
        price = float(prices.iloc[i])
        if np.isnan(price):
            total_equity[i] = total_equity[i - 1] if i > 0 else 0.0
            inventory_track[i] = inventory
            unrealized_pnl[i] = 0.0
            continue

        # --- Target position from signal ---
        target_pos = sig * ep.position_limit

        # --- Gross-exposure cap (portfolio-level; single product here) ---
        if abs(target_pos) > ep.max_gross_exposure:
            target_pos = np.sign(target_pos) * ep.max_gross_exposure

        # --- Stop-loss ---
        if inventory != 0.0 and cost_basis != 0.0:
            unrealized_pct = (price - cost_basis) * np.sign(inventory) / abs(cost_basis)
            if unrealized_pct < -ep.stop_loss_pct:
                target_pos = 0.0   # flatten

        # --- Compute trade ---
        delta = target_pos - inventory
        if abs(delta) >= 1e-9:
            # Execution price with slippage (adverse)
            direction = np.sign(delta)
            fill_price = price * (1.0 + direction * ep.slippage_pct)

            # Transaction cost
            cost = abs(delta) * fill_price * ep.transaction_cost_pct

            old_inv = inventory
            inventory += delta
            cash -= delta * fill_price + cost

            # Update weighted-average cost basis
            if abs(inventory) > 1e-9:
                if np.sign(inventory) == np.sign(old_inv) or old_inv == 0.0:
                    cost_basis = (
                        (cost_basis * abs(old_inv) + fill_price * abs(delta))
                        / abs(inventory)
                    )
                else:
                    cost_basis = fill_price
            else:
                cost_basis = 0.0

            # Per-trade PnL (realised on reducing trades; costs already in cash)
            trade_pnl = 0.0
            if abs(inventory) < abs(old_inv) and np.sign(inventory) == np.sign(old_inv):
                # Price-only attribution; costs are already reflected in cash / equity
                trade_pnl = (fill_price - cost_basis) * (-delta) * np.sign(old_inv)
            trade_records.append({
                "bar": i,
                "price": fill_price,
                "quantity": delta,
                "cost": cost,
                "pnl": trade_pnl,
            })

        # Total equity = cash + mark-to-market of open position
        total_equity[i] = cash + inventory * price
        inventory_track[i] = inventory
        unrealized_pnl[i] = inventory * price

    equity_series = pd.Series(total_equity, index=features_df.index)

    result = BacktestResult(
        product=product,
        strategy_name=strategy.name,
        n_bars=n,
        pnl_series=equity_series,
        unrealized_series=pd.Series(unrealized_pnl, index=features_df.index),
        inventory_series=pd.Series(inventory_track, index=features_df.index),
        trade_log=pd.DataFrame(trade_records),
    )
    result.compute_metrics()
    return result


# ---------------------------------------------------------------------------
# Train / Validation split
# ---------------------------------------------------------------------------

def train_val_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split.  Returns (train_df, val_df).
    The split point is at train_ratio of the sorted rows.
    No shuffling to avoid look-ahead bias.
    """
    n = len(df)
    split = int(n * train_ratio)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parameter search
# ---------------------------------------------------------------------------

def _expand_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _random_sample(
    param_grid: Dict[str, List[Any]],
    n_samples: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    all_combos = _expand_grid(param_grid)
    if len(all_combos) <= n_samples:
        return all_combos
    return rng.sample(all_combos, n_samples)


def parameter_search(
    features_df: pd.DataFrame,
    strategy_cls: Type[BaseStrategy],
    param_grid: Dict[str, List[Any]],
    exec_params: Optional[ExecutionParams] = None,
    train_ratio: float = 0.7,
    search_mode: str = "grid",      # "grid" or "random"
    n_random_samples: int = 30,
    seed: int = 42,
    price_col: str = "mid_price",
    product: str = "UNKNOWN",
) -> pd.DataFrame:
    """
    Search over *param_grid* for a given strategy class.

    Returns
    -------
    DataFrame ranked by Sharpe (descending), one row per parameter set.
    Includes both train and validation metrics.
    """
    train_df, val_df = train_val_split(features_df, train_ratio)

    if search_mode == "random":
        combos = _random_sample(param_grid, n_random_samples, seed=seed)
    else:
        combos = _expand_grid(param_grid)

    rows = []
    for combo in combos:
        # Build strategy params dataclass
        try:
            if strategy_cls.__name__ == "MeanReversionStrategy":
                strat_params = MeanReversionParams(**{
                    k: v for k, v in combo.items()
                    if k in MeanReversionParams.__dataclass_fields__
                })
            elif strategy_cls.__name__ == "MomentumStrategy":
                strat_params = MomentumParams(**{
                    k: v for k, v in combo.items()
                    if k in MomentumParams.__dataclass_fields__
                })
            else:
                strat_params = None
            strat = strategy_cls(strat_params)
        except Exception as exc:
            warnings.warn(f"Failed to build strategy with params {combo}: {exc}")
            continue

        # Train backtest
        train_res = run_backtest(train_df, strat, exec_params, price_col, product)
        # Val backtest
        val_res = run_backtest(val_df, strat, exec_params, price_col, product)

        row = {**combo}
        for k, v in train_res.metrics.items():
            row[f"train_{k}"] = v
        for k, v in val_res.metrics.items():
            row[f"val_{k}"] = v
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    # Rank by validation Sharpe (primary), then by validation PnL (secondary)
    sort_cols = []
    if "val_sharpe" in result_df.columns:
        sort_cols.append("val_sharpe")
    if "val_total_pnl" in result_df.columns:
        sort_cols.append("val_total_pnl")
    if sort_cols:
        result_df.sort_values(sort_cols, ascending=False, inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    return result_df


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def metrics_to_markdown(result: BacktestResult, split: str = "full") -> str:
    """Return a markdown table row for a backtest result."""
    m = result.metrics
    return (
        f"| {result.product} | {result.strategy_name} | {split} "
        f"| {m.get('total_pnl', 0):.2f} "
        f"| {m.get('sharpe', 0):.4f} "
        f"| {m.get('max_drawdown', 0):.2f} "
        f"| {m.get('turnover', 0):.0f} "
        f"| {m.get('n_trades', 0)} "
        f"| {m.get('hit_rate', float('nan')):.2%} |"
    )
