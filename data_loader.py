"""
data_loader.py – Discover, load and summarise IMC-style CSV market data.

Supports two file patterns:
  * prices_*.csv  – order-book snapshots  (semicolon-separated, day column present)
  * trades_*.csv  – public trade tape      (semicolon-separated, no day column)

Usage
-----
    from data_loader import load_all, summarise

    prices_df, trades_df = load_all(".")       # discover from current directory
    print(summarise(prices_df, trades_df))
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv_robust(path: Path) -> Optional[pd.DataFrame]:
    """Read a semicolon-delimited CSV; skip bad lines and warn on errors."""
    try:
        df = pd.read_csv(
            path,
            sep=";",
            on_bad_lines="warn",
            engine="python",
        )
        logger.info("Loaded %s  shape=%s", path.name, df.shape)
        return df
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _infer_price_col(df: pd.DataFrame) -> Optional[str]:
    """Return the best price column available."""
    for candidate in ("mid_price", "price", "close", "last"):
        if candidate in df.columns:
            return candidate
    return None


def _infer_volume_col(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("quantity", "volume", "bid_volume_1", "ask_volume_1"):
        if candidate in df.columns:
            return candidate
    return None


def _infer_product_col(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("product", "symbol", "instrument", "ticker"):
        if candidate in df.columns:
            return candidate
    return None


def _infer_timestamp_col(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("timestamp", "time", "datetime", "date"):
        if candidate in df.columns:
            return candidate
    return None

# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def discover_files(data_dir: str = ".") -> Dict[str, List[Path]]:
    """Return dict with keys 'prices' and 'trades' listing discovered CSVs."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    prices = sorted(root.glob("prices*.csv"))
    trades = sorted(root.glob("trades*.csv"))

    # Fall back: any csv if none found with expected prefixes
    if not prices and not trades:
        all_csv = sorted(root.glob("*.csv"))
        logger.warning(
            "No prices_*.csv / trades_*.csv found; treating all CSVs as prices: %s",
            [p.name for p in all_csv],
        )
        prices = all_csv

    logger.info(
        "Discovered %d price file(s) and %d trade file(s).",
        len(prices), len(trades),
    )
    return {"prices": prices, "trades": trades}


def load_prices(paths: List[Path]) -> pd.DataFrame:
    """Load and concatenate all price snapshot CSVs."""
    frames: List[pd.DataFrame] = []
    for p in paths:
        df = _read_csv_robust(p)
        if df is None:
            continue
        df["_source_file"] = p.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Coerce numeric columns
    numeric_hints = [c for c in combined.columns if any(
        kw in c for kw in ("price", "volume", "pnl", "profit", "mid")
    )]
    for col in numeric_hints:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Sort by day then timestamp if available
    sort_cols = [c for c in ("day", "timestamp") if c in combined.columns]
    if sort_cols:
        combined.sort_values(sort_cols, inplace=True)
        combined.reset_index(drop=True, inplace=True)

    return combined


def load_trades(paths: List[Path]) -> pd.DataFrame:
    """Load and concatenate all trade tape CSVs."""
    frames: List[pd.DataFrame] = []
    for p in paths:
        df = _read_csv_robust(p)
        if df is None:
            continue
        df["_source_file"] = p.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    for col in ("price", "quantity"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    ts_col = _infer_timestamp_col(combined)
    if ts_col:
        combined.sort_values(ts_col, inplace=True)
        combined.reset_index(drop=True, inplace=True)

    return combined


def load_all(data_dir: str = ".") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Discover and load all price and trade CSVs from *data_dir*."""
    files = discover_files(data_dir)
    prices_df = load_prices(files["prices"])
    trades_df = load_trades(files["trades"])
    return prices_df, trades_df


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------

def infer_schema(df: pd.DataFrame, name: str = "dataset") -> Dict:
    """Return a dict describing inferred schema of a dataframe."""
    if df.empty:
        return {"name": name, "empty": True}

    schema = {
        "name": name,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "timestamp_col": _infer_timestamp_col(df),
        "product_col": _infer_product_col(df),
        "price_col": _infer_price_col(df),
        "volume_col": _infer_volume_col(df),
    }
    return schema


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _spread_proxy(df: pd.DataFrame) -> Optional[pd.Series]:
    """Compute best-level spread per row if bid/ask columns exist."""
    if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
        spread = df["ask_price_1"] - df["bid_price_1"]
        return spread.where(spread >= 0)
    return None


def data_quality_report(df: pd.DataFrame, label: str = "dataset") -> str:
    """Return a human-readable string summarising data quality."""
    if df.empty:
        return f"## {label}\n\nDataset is empty.\n"

    lines: List[str] = [f"## {label}", ""]

    # Shape
    lines.append(f"- Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        lines.append(f"- Missing values in {len(missing)} column(s):")
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            lines.append(f"    - `{col}`: {cnt:,} ({pct:.1f}%)")
    else:
        lines.append("- No missing values")

    # Duplicates
    dup_count = df.duplicated().sum()
    lines.append(f"- Duplicate rows: {dup_count:,}")

    # Timestamp range
    ts_col = _infer_timestamp_col(df)
    if ts_col and df[ts_col].notna().any():
        lines.append(
            f"- Timestamp range: {df[ts_col].min()} → {df[ts_col].max()}"
        )

    day_col = "day" if "day" in df.columns else None
    if day_col:
        lines.append(f"- Day values: {sorted(df[day_col].dropna().unique().tolist())}")

    # Products
    prod_col = _infer_product_col(df)
    if prod_col:
        products = df[prod_col].dropna().unique().tolist()
        lines.append(f"- Products: {products}")

    # Per-product stats
    price_col = _infer_price_col(df)
    if prod_col and price_col:
        lines.append("")
        lines.append("### Per-product price statistics")
        lines.append("")
        stats = (
            df.groupby(prod_col)[price_col]
            .agg(["count", "mean", "std", "min", "max"])
            .round(4)
        )
        lines.append(stats.to_markdown())

    # Spread
    spread = _spread_proxy(df)
    if spread is not None and prod_col:
        lines.append("")
        lines.append("### Per-product spread proxy (bid-ask level 1)")
        lines.append("")
        spread_stats = (
            df.assign(_spread=spread)
            .groupby(prod_col)["_spread"]
            .agg(["mean", "std", "min", "max"])
            .round(4)
        )
        lines.append(spread_stats.to_markdown())

    # Returns and volatility
    if prod_col and price_col:
        lines.append("")
        lines.append("### Per-product return statistics")
        lines.append("")
        ret_rows = []
        for product, grp in df.groupby(prod_col):
            prices = grp[price_col].dropna()
            if len(prices) < 2:
                continue
            rets = prices.pct_change().dropna()
            vol = rets.std()
            # simple outlier count: |ret| > 3 * std
            outliers = (rets.abs() > 3 * vol).sum() if vol > 0 else 0
            ret_rows.append({
                "product": product,
                "n_bars": len(prices),
                "mean_ret": round(rets.mean(), 6),
                "vol": round(vol, 6),
                "skew": round(float(rets.skew()), 4),
                "outliers_3σ": int(outliers),
            })
        if ret_rows:
            ret_df = pd.DataFrame(ret_rows).set_index("product")
            lines.append(ret_df.to_markdown())

    # Volume / liquidity proxy
    vol_col = _infer_volume_col(df)
    if prod_col and vol_col:
        lines.append("")
        lines.append("### Per-product volume / liquidity proxy")
        lines.append("")
        liq = (
            df.groupby(prod_col)[vol_col]
            .agg(["sum", "mean", "max"])
            .round(2)
        )
        lines.append(liq.to_markdown())

    lines.append("")
    return "\n".join(lines)


def get_products(prices_df: pd.DataFrame) -> List[str]:
    """Return sorted list of products in the prices dataframe."""
    prod_col = _infer_product_col(prices_df)
    if prod_col is None or prices_df.empty:
        return []
    return sorted(prices_df[prod_col].dropna().unique().tolist())


def get_product_df(prices_df: pd.DataFrame, product: str) -> pd.DataFrame:
    """Return rows for a single product, sorted by (day, timestamp)."""
    prod_col = _infer_product_col(prices_df)
    if prod_col is None:
        return pd.DataFrame()
    mask = prices_df[prod_col] == product
    sub = prices_df[mask].copy()
    sort_cols = [c for c in ("day", "timestamp") if c in sub.columns]
    if sort_cols:
        sub.sort_values(sort_cols, inplace=True)
    sub.reset_index(drop=True, inplace=True)
    return sub
