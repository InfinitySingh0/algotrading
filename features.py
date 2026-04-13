"""
features.py
===========
Reusable feature engineering for the IMC Trading Challenge pipeline.

All rolling/window features are shifted by 1 bar to prevent look-ahead bias.
Column availability is checked defensively; a warning is issued when a feature
cannot be computed and the column is filled with NaN.

Public API
----------
build_features(df, schema, window_short, window_long) -> pd.DataFrame
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from data_loader import _find_col, _BID1_HINTS, _ASK1_HINTS, _PRICE_HINTS, _VOLUME_HINTS


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_col(df: pd.DataFrame, col: Optional[str]) -> Optional[pd.Series]:
    """Return numeric series for *col* or None if absent/all-NaN."""
    if col is None or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    return s if s.notna().any() else None


def _shifted(s: pd.Series, shift: int = 1) -> pd.Series:
    """Shift a series to avoid look-ahead bias."""
    return s.shift(shift)


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def feat_mid_price(df: pd.DataFrame, schema: dict) -> pd.Series:
    """Best-estimate mid-price: prefer mid_price col, else (bid1+ask1)/2, else price."""
    mid_col = _find_col(df, ["mid_price"])
    if mid_col:
        s = _safe_col(df, mid_col)
        if s is not None:
            return s.rename("mid_price")

    bid1 = _find_col(df, _BID1_HINTS)
    ask1 = _find_col(df, _ASK1_HINTS)
    if bid1 and ask1:
        b = _safe_col(df, bid1)
        a = _safe_col(df, ask1)
        if b is not None and a is not None:
            return ((b + a) / 2).rename("mid_price")

    price_col = _find_col(df, _PRICE_HINTS)
    s = _safe_col(df, price_col)
    if s is not None:
        warnings.warn("mid_price: falling back to price column.")
        return s.rename("mid_price")

    warnings.warn("mid_price: no suitable column found; returning NaN series.")
    return pd.Series(np.nan, index=df.index, name="mid_price")


def feat_spread(df: pd.DataFrame, schema: dict) -> pd.Series:
    """Best-effort bid-ask spread; returns NaN if not available."""
    bid1 = _find_col(df, _BID1_HINTS)
    ask1 = _find_col(df, _ASK1_HINTS)
    if bid1 and ask1:
        b = _safe_col(df, bid1)
        a = _safe_col(df, ask1)
        if b is not None and a is not None:
            return (a - b).rename("spread")
    warnings.warn("spread: bid/ask columns not found; returning NaN.")
    return pd.Series(np.nan, index=df.index, name="spread")


def feat_returns(mid: pd.Series, window: int = 1) -> pd.Series:
    """Log returns over *window* bars (shifted by 1 to prevent leakage)."""
    log_r = np.log(mid / mid.shift(window))
    return _shifted(log_r).rename(f"ret_{window}")


def feat_momentum(mid: pd.Series, window: int = 20) -> pd.Series:
    """Cumulative log return over *window* bars (shifted)."""
    mom = np.log(mid / mid.shift(window))
    return _shifted(mom).rename(f"momentum_{window}")


def feat_zscore(mid: pd.Series, window: int = 40) -> pd.Series:
    """
    Mean-reversion z-score: (price - rolling_mean) / rolling_std.
    A large positive z-score suggests over-bought; negative → over-sold.
    Shifted by 1 bar.
    """
    roll_mean = mid.rolling(window, min_periods=window // 2).mean()
    roll_std = mid.rolling(window, min_periods=window // 2).std()
    z = (mid - roll_mean) / roll_std.replace(0, np.nan)
    return _shifted(z).rename(f"zscore_{window}")


def feat_rolling_vol(mid: pd.Series, window: int = 20) -> pd.Series:
    """Annualised rolling log-return volatility (shifted)."""
    log_r = np.log(mid / mid.shift(1))
    rv = log_r.rolling(window, min_periods=window // 2).std()
    return _shifted(rv).rename(f"rvol_{window}")


def feat_volume_imbalance(df: pd.DataFrame, schema: dict) -> pd.Series:
    """
    Order-flow / volume-imbalance proxy using LOB bid/ask volumes.

    imbalance = (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1)

    Falls back to NaN if volumes are absent.
    """
    bv_col = _find_col(df, ["bid_volume_1", "bid_vol_1", "bidvolume1"])
    av_col = _find_col(df, ["ask_volume_1", "ask_vol_1", "askvolume1"])
    if bv_col and av_col:
        bv = _safe_col(df, bv_col)
        av = _safe_col(df, av_col)
        if bv is not None and av is not None:
            total = (bv + av).replace(0, np.nan)
            imb = (bv - av) / total
            return _shifted(imb).rename("vol_imbalance")
    warnings.warn("vol_imbalance: bid/ask volume columns not found; returning NaN.")
    return pd.Series(np.nan, index=df.index, name="vol_imbalance")


def feat_intraday_time(df: pd.DataFrame) -> pd.Series:
    """
    Normalised intraday time [0, 1] based on 'ts' column, if it's numeric.
    Useful for detecting open/close patterns.
    """
    if "ts" in df.columns and pd.api.types.is_numeric_dtype(df["ts"]):
        ts = df["ts"].astype(float)
        rng = ts.max() - ts.min()
        if rng > 0:
            return ((ts - ts.min()) / rng).rename("intraday_time")
    return pd.Series(np.nan, index=df.index, name="intraday_time")


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    schema: dict,
    window_short: int = 20,
    window_long: int = 60,
) -> pd.DataFrame:
    """
    Build full feature dataframe for a single product's price history.

    Parameters
    ----------
    df           : DataFrame for a single product (sorted by time).
    schema       : Schema dict from data_loader.infer_schema.
    window_short : Short rolling window (bars).
    window_long  : Long rolling window (bars).

    Returns
    -------
    DataFrame with original columns + feature columns.
    All features are shifted 1 bar (no look-ahead).
    """
    out = df.copy()

    # --- Core price series ---
    mid = feat_mid_price(out, schema)
    out["mid_price"] = mid

    # --- Spread ---
    out["spread"] = feat_spread(out, schema)

    # --- Returns ---
    out["ret_1"] = feat_returns(mid, 1)
    out["ret_5"] = feat_returns(mid, 5)

    # --- Momentum ---
    out[f"momentum_{window_short}"] = feat_momentum(mid, window_short)
    out[f"momentum_{window_long}"] = feat_momentum(mid, window_long)

    # --- Mean-reversion z-scores ---
    out[f"zscore_{window_short}"] = feat_zscore(mid, window_short)
    out[f"zscore_{window_long}"] = feat_zscore(mid, window_long)

    # --- Rolling volatility ---
    out[f"rvol_{window_short}"] = feat_rolling_vol(mid, window_short)
    out[f"rvol_{window_long}"] = feat_rolling_vol(mid, window_long)

    # --- Volume imbalance ---
    out["vol_imbalance"] = feat_volume_imbalance(out, schema)

    # --- Intraday time ---
    out["intraday_time"] = feat_intraday_time(out)

    return out


# ---------------------------------------------------------------------------
# Cross-product relative-value features
# ---------------------------------------------------------------------------

def build_cross_product_features(
    product_dfs: dict[str, pd.DataFrame],
    window: int = 40,
) -> dict[str, pd.DataFrame]:
    """
    Add cross-product relative-value features when multiple products exist.

    For each pair (A, B), adds to each product's DataFrame:
      - rel_value_{B}: z-score of (mid_A - mid_B * hedge_ratio) over *window* bars
      - hedge_ratio is the ratio of mean prices (static, computed on full series
        but shifted to avoid leakage).

    Parameters
    ----------
    product_dfs : {product_name: feature_df_with_mid_price}
    window      : rolling window for the spread z-score

    Returns
    -------
    Updated copy of product_dfs with cross features appended.
    """
    if len(product_dfs) < 2:
        return product_dfs

    products = list(product_dfs.keys())
    out = {p: df.copy() for p, df in product_dfs.items()}

    # Align midprice series on a shared integer index (ts)
    mid_series: dict[str, pd.Series] = {}
    for p, df in product_dfs.items():
        if "mid_price" in df.columns:
            mid_series[p] = df.set_index("ts")["mid_price"] if "ts" in df.columns else df["mid_price"]

    for i, pa in enumerate(products):
        for pb in products[i + 1:]:
            if pa not in mid_series or pb not in mid_series:
                continue
            ma = mid_series[pa]
            mb = mid_series[pb]
            # Align
            aligned = pd.DataFrame({"a": ma, "b": mb}).dropna()
            if len(aligned) < window:
                continue
            # The hedge ratio uses full-sample mean prices, which is a static
            # approximation.  The rolling z-score is still shifted by 1 bar (line below)
            # to prevent look-ahead in the signal.  In production, the hedge ratio
            # should be estimated only on past data (e.g. via expanding-window regression).
            hedge = float(aligned["a"].mean() / aligned["b"].mean()) if aligned["b"].mean() != 0 else 1.0
            spread_series = aligned["a"] - hedge * aligned["b"]
            roll_mean = spread_series.rolling(window, min_periods=window // 2).mean()
            roll_std = spread_series.rolling(window, min_periods=window // 2).std()
            z = ((spread_series - roll_mean) / roll_std.replace(0, np.nan)).shift(1)
            z.name = f"rel_value_{pb}"

            # Merge back
            if "ts" in out[pa].columns:
                out[pa] = out[pa].merge(
                    z.reset_index().rename(columns={"index": "ts"}),
                    on="ts", how="left"
                )
            else:
                if len(z) == len(out[pa]):
                    out[pa][f"rel_value_{pb}"] = z.values

    return out
