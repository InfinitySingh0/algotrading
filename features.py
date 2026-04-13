"""
features.py – Leakage-safe feature engineering for IMC trading data.

All features are computed with strict anti-look-ahead semantics:
  * Rolling statistics use only past/current data (.shift(1) where the
    feature value itself would otherwise include the current bar).
  * No future prices are ever accessed.

Usage
-----
    from features import build_features
    feat_df = build_features(product_df)   # product_df from data_loader.get_product_df
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

import config as cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_price(df: pd.DataFrame) -> str:
    """Return mid_price column name, raising if absent."""
    if "mid_price" in df.columns:
        return "mid_price"
    # Derive mid from bid/ask level 1
    if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
        df = df.copy()
        df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
        return "mid_price"
    for fallback in ("price", "close", "last"):
        if fallback in df.columns:
            return fallback
    raise KeyError("No usable price column found in dataframe.")


# ---------------------------------------------------------------------------
# Core feature builders
# ---------------------------------------------------------------------------

def add_mid_price(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'mid_price' column exists (derive if needed)."""
    df = df.copy()
    if "mid_price" not in df.columns:
        if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
            df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
        elif "price" in df.columns:
            df["mid_price"] = df["price"]
    return df


def add_returns(df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
    """Add log-return and rolling-return features."""
    df = df.copy()
    if "mid_price" not in df.columns:
        df = add_mid_price(df)

    price = df["mid_price"].replace(0, np.nan)

    # Log returns (1-bar)
    df["log_ret_1"] = np.log(price / price.shift(1))

    windows = windows or cfg.ROLLING_WINDOWS
    for w in windows:
        # Rolling sum of log-returns (leakage-safe: uses past w bars)
        df[f"ret_{w}"] = df["log_ret_1"].rolling(w, min_periods=1).sum()

    return df


def add_momentum(df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
    """Add price momentum: (price_t - price_{t-w}) / price_{t-w}."""
    df = df.copy()
    if "mid_price" not in df.columns:
        df = add_mid_price(df)

    price = df["mid_price"].replace(0, np.nan)

    windows = windows or cfg.MOMENTUM_WINDOWS
    for w in windows:
        lagged = price.shift(w)
        df[f"mom_{w}"] = (price - lagged) / lagged

    return df


def add_zscore(df: pd.DataFrame, window: int = cfg.ZSCORE_WINDOW) -> pd.DataFrame:
    """Add rolling mean-reversion z-score of mid_price."""
    df = df.copy()
    if "mid_price" not in df.columns:
        df = add_mid_price(df)

    price = df["mid_price"].replace(0, np.nan)

    roll_mean = price.rolling(window, min_periods=2).mean()
    roll_std = price.rolling(window, min_periods=2).std()

    # Shift so that z-score at bar t uses mean/std computed up to bar t-1
    roll_mean_lag = roll_mean.shift(1)
    roll_std_lag = roll_std.shift(1)

    with np.errstate(divide="ignore", invalid="ignore"):
        df["zscore"] = np.where(
            roll_std_lag > 0,
            (price - roll_mean_lag) / roll_std_lag,
            0.0,
        )

    return df


def add_rolling_volatility(df: pd.DataFrame, window: int = cfg.VOL_WINDOW) -> pd.DataFrame:
    """Add rolling realised volatility (std of log-returns)."""
    df = df.copy()
    if "log_ret_1" not in df.columns:
        df = add_returns(df)

    df["roll_vol"] = df["log_ret_1"].rolling(window, min_periods=2).std()
    return df


def add_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add bid-ask spread features when level-1 book columns exist."""
    df = df.copy()
    if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
        df["spread"] = df["ask_price_1"] - df["bid_price_1"]
        df["rel_spread"] = df["spread"] / df["mid_price"].replace(0, np.nan)

        # Rolling mean spread (lagged to avoid look-ahead)
        df["roll_spread_20"] = df["spread"].rolling(20, min_periods=1).mean().shift(1)
    return df


def add_volume_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add order-flow / volume imbalance features.

    Imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    Cumulative order flow delta over short window.
    """
    df = df.copy()
    bid_v = None
    ask_v = None

    if "bid_volume_1" in df.columns and "ask_volume_1" in df.columns:
        bid_v = df["bid_volume_1"].fillna(0)
        ask_v = df["ask_volume_1"].fillna(0)
    elif "quantity" in df.columns:
        # Trades file – no direct imbalance, use signed qty proxy
        df["signed_qty"] = df["quantity"].fillna(0)
        df["cumflow_10"] = df["signed_qty"].rolling(10, min_periods=1).sum().shift(1)
        return df

    if bid_v is not None and ask_v is not None:
        total = bid_v + ask_v
        with np.errstate(divide="ignore", invalid="ignore"):
            df["vol_imbalance"] = np.where(
                total > 0, (bid_v - ask_v) / total, 0.0
            )
        # Cumulative imbalance over short window (lagged)
        df["cumflow_10"] = df["vol_imbalance"].rolling(10, min_periods=1).sum().shift(1)

    return df


def add_cross_product_features(
    prices_df: pd.DataFrame,
    products: List[str],
) -> pd.DataFrame:
    """
    Add cross-product relative value features.

    Computes ratio and z-score of each product pair.
    Returns a combined dataframe indexed by (timestamp, product) with added columns.
    Requires prices_df to have 'product' and 'mid_price' and 'timestamp' columns.
    """
    if len(products) < 2:
        return prices_df

    df = prices_df.copy()
    if "mid_price" not in df.columns:
        df = add_mid_price(df)

    # Pivot to wide: index=timestamp, columns=product
    ts_col = "timestamp"
    prod_col = "product"
    if ts_col not in df.columns or prod_col not in df.columns:
        return df

    wide = df.pivot_table(index=ts_col, columns=prod_col, values="mid_price", aggfunc="last")

    for i, p1 in enumerate(products):
        for p2 in products[i + 1:]:
            if p1 not in wide.columns or p2 not in wide.columns:
                continue
            col_name = f"rel_{p1}_{p2}"
            ratio = wide[p1] / wide[p2].replace(0, np.nan)
            roll_mean = ratio.rolling(20, min_periods=2).mean().shift(1)
            roll_std = ratio.rolling(20, min_periods=2).std().shift(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                zscore = np.where(roll_std > 0, (ratio - roll_mean) / roll_std, 0.0)
            ratio_series = ratio.rename(col_name)
            zscore_series = pd.Series(zscore, index=wide.index, name=f"z_{col_name}")

            # Merge back on timestamp
            df = df.merge(
                pd.concat([ratio_series, zscore_series], axis=1).reset_index(),
                on=ts_col,
                how="left",
            )

    return df


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    rolling_windows: Optional[List[int]] = None,
    momentum_windows: Optional[List[int]] = None,
    zscore_window: int = cfg.ZSCORE_WINDOW,
    vol_window: int = cfg.VOL_WINDOW,
) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a single-product dataframe.

    Parameters
    ----------
    df : product-level price dataframe from data_loader.get_product_df
    rolling_windows : windows for rolling returns (default from config)
    momentum_windows : windows for momentum (default from config)
    zscore_window : window for mean-reversion z-score
    vol_window : window for rolling volatility

    Returns
    -------
    df with added feature columns (no in-place modification of input)
    """
    df = add_mid_price(df)
    df = add_returns(df, windows=rolling_windows)
    df = add_momentum(df, windows=momentum_windows)
    df = add_zscore(df, window=zscore_window)
    df = add_rolling_volatility(df, window=vol_window)
    df = add_spread_features(df)
    df = add_volume_imbalance(df)

    # Drop rows where mid_price is NaN (typically first few bars)
    feature_cols = [c for c in df.columns if c not in ("day", "timestamp", "product", "_source_file")]
    df = df.dropna(subset=["mid_price"]).reset_index(drop=True)

    return df
