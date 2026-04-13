"""
backtest.py – Bar-based backtesting engine for the IMC trading pipeline.

Features
--------
* Iterates row-by-row over a feature-enriched product dataframe
* Calls strategy.generate_signal() each bar
* Enforces position limits, gross exposure cap, stop-loss
* Tracks realized + unrealized PnL, inventory, turnover, drawdown
* Provides train/validation split and performance summary
* Supports simple grid and random parameter search

Usage
-----
    from backtest import run_backtest, train_val_split, grid_search
    from strategies import MeanReversionStrategy

    strategy = MeanReversionStrategy(entry_z=1.5, exit_z=0.5)
    results = run_backtest(strategy, feature_df)
    print(results["metrics"])
"""

from __future__ import annotations

import itertools
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg
from metrics import compute_metrics
from strategies import BaseStrategy, create_strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Train / Validation split
# ---------------------------------------------------------------------------

def train_val_split(
    df: pd.DataFrame,
    train_ratio: float = cfg.TRAIN_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split: first `train_ratio` of rows → train, rest → validation.
    """
    n = len(df)
    split = int(n * train_ratio)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


# ---------------------------------------------------------------------------
# Core backtester
# ---------------------------------------------------------------------------

def run_backtest(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    risk_free: float = cfg.RISK_FREE_RATE,
    label: str = "",
) -> Dict[str, Any]:
    """
    Run a bar-based backtest of *strategy* on feature dataframe *df*.

    Returns a dict with:
      - "pnl_series": per-bar realized PnL Series
      - "position_series": per-bar inventory Series
      - "cumulative_pnl": cumulative PnL Series
      - "metrics": dict of performance metrics
      - "trade_log": list of trade records
    """
    if df.empty:
        return {
            "pnl_series": pd.Series(dtype=float),
            "position_series": pd.Series(dtype=float),
            "cumulative_pnl": pd.Series(dtype=float),
            "metrics": {},
            "trade_log": [],
        }

    price_col = "mid_price" if "mid_price" in df.columns else "price"

    position = 0
    cost_basis: Optional[float] = None  # volume-weighted average entry price per unit

    pnl_list: List[float] = []
    position_list: List[int] = []
    trade_log: List[Dict] = []

    # Pre-extract to plain Python structures for fast iteration
    prices_arr = df[price_col].to_numpy(dtype=float)
    # Convert to list of dicts once; much faster than repeated df.iloc calls
    records: List[Dict] = df.to_dict("records")
    n_bars = len(df)
    prev_price: float = float("nan")

    for bar_i in range(n_bars):
        row_dict = records[bar_i]
        price = prices_arr[bar_i]

        if np.isnan(price):
            pnl_list.append(0.0)
            position_list.append(position)
            prev_price = price
            continue

        # ----------------------------------------------------------------
        # Stop-loss check before generating new signal
        # ----------------------------------------------------------------
        if cost_basis is not None and position != 0:
            unrealized_pnl = (price - cost_basis) * position
            if unrealized_pnl < -strategy.stop_loss_threshold:
                # Flatten immediately
                cost = strategy.trade_cost(-position, price)
                bar_pnl = (price - cost_basis) * position - cost
                trade_log.append({
                    "bar": bar_i, "qty": -position,
                    "price": price, "reason": "stop_loss",
                    "pnl": bar_pnl,
                })
                position = 0
                cost_basis = None
                pnl_list.append(bar_pnl)
                position_list.append(position)
                prev_price = price
                continue

        # ----------------------------------------------------------------
        # Generate signal
        # ----------------------------------------------------------------
        signal = strategy.generate_signal(row_dict, position)
        desired = signal.desired_position

        # ----------------------------------------------------------------
        # Gross exposure cap
        # ----------------------------------------------------------------
        desired = int(np.clip(desired, -strategy.max_gross_exposure, strategy.max_gross_exposure))

        # ----------------------------------------------------------------
        # Execute trade
        # ----------------------------------------------------------------
        qty = desired - position  # delta
        bar_pnl = 0.0

        if qty != 0:
            cost = strategy.trade_cost(qty, price)

            if position != 0 and np.sign(qty) != np.sign(position):
                # Partial or full unwind
                close_qty = min(abs(qty), abs(position)) * np.sign(qty)
                if cost_basis is not None:
                    close_pnl = (price - cost_basis) * (-close_qty)
                else:
                    close_pnl = 0.0
                bar_pnl += close_pnl

            position += qty
            if position == 0:
                cost_basis = None
            elif cost_basis is None:
                cost_basis = price
            else:
                # Update average cost basis
                prev_abs = abs(position - qty)
                new_abs = abs(qty)
                total_abs = abs(position)
                if total_abs > 0:
                    cost_basis = (cost_basis * prev_abs + price * new_abs) / total_abs

            bar_pnl -= cost  # deduct transaction cost + slippage

            trade_log.append({
                "bar": bar_i, "qty": qty, "price": price,
                "reason": signal.reason, "pnl": bar_pnl,
            })

        # Mark-to-market change for existing position (no trade this bar)
        else:
            if position != 0 and not np.isnan(prev_price):
                bar_pnl = (price - prev_price) * position

        pnl_list.append(bar_pnl)
        position_list.append(position)
        prev_price = price

    pnl_series = pd.Series(pnl_list, index=df.index, name="pnl")
    pos_series = pd.Series(position_list, index=df.index, name="position")
    cum_pnl = pnl_series.cumsum()

    metrics = compute_metrics(pnl_series, pos_series, risk_free=risk_free)

    return {
        "pnl_series": pnl_series,
        "position_series": pos_series,
        "cumulative_pnl": cum_pnl,
        "metrics": metrics,
        "trade_log": trade_log,
        "label": label,
    }


# ---------------------------------------------------------------------------
# Convenience: run on both train and validation sets
# ---------------------------------------------------------------------------

def run_train_val(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    train_ratio: float = cfg.TRAIN_RATIO,
) -> Dict[str, Any]:
    """
    Run backtest on train and validation splits, return combined result.
    """
    train_df, val_df = train_val_split(df, train_ratio)
    train_result = run_backtest(strategy, train_df, label="train")
    val_result = run_backtest(strategy, val_df, label="validation")
    return {
        "train": train_result,
        "validation": val_result,
        "n_train": len(train_df),
        "n_val": len(val_df),
    }


# ---------------------------------------------------------------------------
# Parameter search
# ---------------------------------------------------------------------------

def grid_search(
    strategy_name: str,
    param_grid: Dict[str, List[Any]],
    df: pd.DataFrame,
    train_ratio: float = cfg.TRAIN_RATIO,
    rank_metric: str = "sharpe",
    fixed_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Exhaustive grid search over param_grid.

    Parameters
    ----------
    strategy_name : "mean_reversion" or "momentum"
    param_grid    : {param_name: [value1, value2, ...], ...}
    df            : feature-enriched single-product dataframe
    train_ratio   : fraction for training
    rank_metric   : metric to rank by (higher is better)
    fixed_params  : parameters held constant across all runs

    Returns
    -------
    DataFrame with one row per parameter set, sorted by val rank_metric desc
    """
    fixed_params = fixed_params or {}
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    logger.info("Grid search: %d combinations for %s", len(combos), strategy_name)
    return _run_search(strategy_name, keys, combos, df, train_ratio, rank_metric, fixed_params)


def random_search(
    strategy_name: str,
    param_grid: Dict[str, List[Any]],
    df: pd.DataFrame,
    n_iter: int = cfg.PARAM_SEARCH_N_RANDOM,
    train_ratio: float = cfg.TRAIN_RATIO,
    rank_metric: str = "sharpe",
    seed: int = cfg.PARAM_SEARCH_RANDOM_SEED,
    fixed_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Random search: sample *n_iter* parameter combinations.
    """
    fixed_params = fixed_params or {}
    rng = random.Random(seed)
    keys = list(param_grid.keys())
    all_combos = list(itertools.product(*[param_grid[k] for k in keys]))
    n_sample = min(n_iter, len(all_combos))
    combos = rng.sample(all_combos, n_sample)
    logger.info("Random search: %d combinations for %s", len(combos), strategy_name)
    return _run_search(strategy_name, keys, combos, df, train_ratio, rank_metric, fixed_params)


def _run_search(
    strategy_name: str,
    keys: List[str],
    combos: List[tuple],
    df: pd.DataFrame,
    train_ratio: float,
    rank_metric: str,
    fixed_params: Dict,
) -> pd.DataFrame:
    """Internal helper shared by grid_search and random_search."""
    train_df, val_df = train_val_split(df, train_ratio)

    records = []
    for combo in combos:
        params = dict(zip(keys, combo))
        params.update(fixed_params)
        try:
            strat = create_strategy(strategy_name, **params)
            train_res = run_backtest(strat, train_df, label="train")
            val_res = run_backtest(strat, val_df, label="val")

            row = {**params}
            for k, v in train_res["metrics"].items():
                row[f"train_{k}"] = round(v, 4) if isinstance(v, float) else v
            for k, v in val_res["metrics"].items():
                row[f"val_{k}"] = round(v, 4) if isinstance(v, float) else v
            records.append(row)
        except Exception as exc:
            logger.warning("Param combo %s failed: %s", params, exc)

    if not records:
        return pd.DataFrame()

    results_df = pd.DataFrame(records)
    val_col = f"val_{rank_metric}"
    if val_col in results_df.columns:
        results_df.sort_values(val_col, ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    return results_df
