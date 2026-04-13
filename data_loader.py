"""
data_loader.py
==============
Discover, load, and profile CSV market-data files for the IMC Trading
Challenge research pipeline.

Supported schemas (auto-detected):
  - prices_*.csv  : semicolon-separated LOB snapshots with bid/ask levels
  - trades_*.csv  : semicolon-separated executed trades

Fallback behaviour is documented via warnings when expected columns are absent.
"""

from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column-name heuristics
# ---------------------------------------------------------------------------

# Candidate column names for common fields (case-insensitive match)
_PRICE_HINTS = ["mid_price", "price", "close", "last", "bid_price_1", "ask_price_1"]
_VOLUME_HINTS = ["quantity", "volume", "size", "vol"]
_TIMESTAMP_HINTS = ["timestamp", "time", "ts", "date", "datetime"]
_PRODUCT_HINTS = ["product", "symbol", "instrument", "ticker"]
_BID1_HINTS = ["bid_price_1", "bid1", "bid_price"]
_ASK1_HINTS = ["ask_price_1", "ask1", "ask_price"]

# Threshold multiplier for IQR-based outlier detection
_OUTLIER_IQR_MULTIPLIER: float = 3.0


def _find_col(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    """Return the first column whose lower-cased name matches any hint."""
    lower = {c.lower(): c for c in df.columns}
    for h in hints:
        if h.lower() in lower:
            return lower[h.lower()]
    return None


# ---------------------------------------------------------------------------
# CSV detection helpers
# ---------------------------------------------------------------------------

def _detect_separator(path: str) -> str:
    """Peek at the first line and guess the field delimiter."""
    with open(path, "r", errors="replace") as fh:
        first = fh.readline()
    if first.count(";") > first.count(","):
        return ";"
    return ","


def _is_prices_file(path: str) -> bool:
    return "price" in os.path.basename(path).lower()


def _is_trades_file(path: str) -> bool:
    return "trade" in os.path.basename(path).lower()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a single CSV with auto-detected separator.
    Malformed rows are silently dropped (on_bad_lines='warn').
    """
    sep = _detect_separator(path)
    try:
        df = pd.read_csv(
            path,
            sep=sep,
            on_bad_lines="warn",
            low_memory=False,
        )
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Failed to read {path}: {exc}")
        return pd.DataFrame()

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Try to coerce object columns to numeric."""
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0.5 * len(df):
            df[col] = converted
    return df


def _parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a 'ts' column representing an orderable time index."""
    ts_col = _find_col(df, _TIMESTAMP_HINTS)
    if ts_col is None:
        warnings.warn("No timestamp column found; using row index as ts.")
        df["ts"] = np.arange(len(df))
        return df

    raw = df[ts_col]
    # Already numeric?
    if pd.api.types.is_numeric_dtype(raw):
        df["ts"] = raw.astype(float)
        return df

    # Try parsing as datetime
    parsed = pd.to_datetime(raw, errors="coerce", infer_datetime_format=True)
    if parsed.notna().mean() > 0.5:
        df["ts"] = parsed
        return df

    # Fallback: try numeric coercion
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().mean() > 0.5:
        df["ts"] = num
        return df

    warnings.warn(f"Could not parse timestamp column '{ts_col}'; using row index.")
    df["ts"] = np.arange(len(df))
    return df


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------

def infer_schema(df: pd.DataFrame, path: str) -> dict:
    """Return a dict describing the inferred schema of a loaded dataframe."""
    schema: dict = {
        "path": path,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "file_type": "prices" if _is_prices_file(path) else
                     ("trades" if _is_trades_file(path) else "unknown"),
        "price_col": _find_col(df, _PRICE_HINTS),
        "volume_col": _find_col(df, _VOLUME_HINTS),
        "timestamp_col": _find_col(df, _TIMESTAMP_HINTS),
        "product_col": _find_col(df, _PRODUCT_HINTS),
        "bid1_col": _find_col(df, _BID1_HINTS),
        "ask1_col": _find_col(df, _ASK1_HINTS),
    }
    return schema


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_stats(df: pd.DataFrame, schema: dict) -> dict:
    """Compute data-quality and descriptive statistics for one dataframe."""
    stats: dict = {
        "n_rows": len(df),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "missing_by_col": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
    }

    ts_col = schema.get("timestamp_col")
    if ts_col and ts_col in df.columns:
        ts = pd.to_numeric(df[ts_col], errors="coerce").dropna()
        stats["ts_min"] = float(ts.min()) if len(ts) else None
        stats["ts_max"] = float(ts.max()) if len(ts) else None
    else:
        stats["ts_min"] = None
        stats["ts_max"] = None

    price_col = schema.get("price_col")
    if price_col and price_col in df.columns:
        prices = pd.to_numeric(df[price_col], errors="coerce").dropna()
        if len(prices):
            q1, q3 = prices.quantile(0.25), prices.quantile(0.75)
            iqr = q3 - q1
            n_outliers = int(((prices < q1 - _OUTLIER_IQR_MULTIPLIER * iqr) | (prices > q3 + _OUTLIER_IQR_MULTIPLIER * iqr)).sum())
            stats["price_stats"] = {
                "mean": float(prices.mean()),
                "std": float(prices.std()),
                "min": float(prices.min()),
                "max": float(prices.max()),
                "n_outliers_iqr3": n_outliers,
            }

    bid1 = schema.get("bid1_col")
    ask1 = schema.get("ask1_col")
    if bid1 and ask1 and bid1 in df.columns and ask1 in df.columns:
        b = pd.to_numeric(df[bid1], errors="coerce")
        a = pd.to_numeric(df[ask1], errors="coerce")
        spread = (a - b).dropna()
        if len(spread):
            stats["spread_proxy"] = {
                "mean": float(spread.mean()),
                "median": float(spread.median()),
                "std": float(spread.std()),
            }
    else:
        price_col2 = schema.get("price_col")
        if price_col2 and price_col2 in df.columns:
            prices2 = pd.to_numeric(df[price_col2], errors="coerce").dropna()
            if len(prices2) > 1:
                rets = prices2.pct_change().dropna()
                stats["spread_proxy"] = {
                    "fallback": "2 * median(|return|) * price used as spread proxy",
                    "estimate": float(2 * rets.abs().median() * prices2.mean()),
                }

    return stats


# ---------------------------------------------------------------------------
# Per-product aggregation
# ---------------------------------------------------------------------------

def per_product_stats(
    df: pd.DataFrame, schema: dict
) -> Dict[str, dict]:
    """
    Split the dataframe by product and compute per-product statistics.
    Returns {product_name: stats_dict}.
    """
    product_col = schema.get("product_col")
    price_col = schema.get("price_col")

    if product_col is None or product_col not in df.columns:
        warnings.warn("No product column found; treating whole file as one product.")
        return {"_ALL_": _product_stats_single(df, schema)}

    results = {}
    for prod, grp in df.groupby(product_col):
        results[str(prod)] = _product_stats_single(grp.reset_index(drop=True), schema)
    return results


def _product_stats_single(df: pd.DataFrame, schema: dict) -> dict:
    """Compute stats for a single-product slice."""
    price_col = schema.get("price_col")
    volume_col = schema.get("volume_col")
    bid1 = schema.get("bid1_col")
    ask1 = schema.get("ask1_col")

    s: dict = {}

    if price_col and price_col in df.columns:
        prices = pd.to_numeric(df[price_col], errors="coerce").dropna()
        rets = prices.pct_change().dropna()
        s["n_bars"] = int(len(prices))
        s["price_mean"] = float(prices.mean())
        s["price_std"] = float(prices.std())
        s["volatility_daily"] = float(rets.std()) if len(rets) else 0.0
        s["return_mean"] = float(rets.mean()) if len(rets) else 0.0
        s["return_std"] = float(rets.std()) if len(rets) else 0.0

    if volume_col and volume_col in df.columns:
        vols = pd.to_numeric(df[volume_col], errors="coerce").dropna()
        s["volume_mean"] = float(vols.mean()) if len(vols) else 0.0
        s["volume_total"] = float(vols.sum()) if len(vols) else 0.0

    if bid1 and ask1 and bid1 in df.columns and ask1 in df.columns:
        b = pd.to_numeric(df[bid1], errors="coerce")
        a = pd.to_numeric(df[ask1], errors="coerce")
        spread = (a - b).dropna()
        s["spread_mean"] = float(spread.mean()) if len(spread) else None
        s["spread_median"] = float(spread.median()) if len(spread) else None
        # Liquidity proxy: 1 / mean_spread (larger = more liquid)
        if s.get("spread_mean") and s["spread_mean"] > 0:
            s["liquidity_proxy"] = float(1.0 / s["spread_mean"])

    return s


# ---------------------------------------------------------------------------
# Main discovery / loading function
# ---------------------------------------------------------------------------

def discover_and_load(
    data_dir: str = ".",
    recursive: bool = True,
    exclude_dirs: Optional[List[str]] = None,
) -> Tuple[List[pd.DataFrame], List[dict], List[dict]]:
    """
    Discover all CSV files under *data_dir*, load them, infer schemas,
    and compute summary statistics.

    Parameters
    ----------
    data_dir     : Root directory to search.
    recursive    : Whether to recurse into sub-directories.
    exclude_dirs : Sub-directory names to exclude (e.g. ["outputs"]).

    Returns
    -------
    dataframes : list[DataFrame]
    schemas    : list[dict]
    stats      : list[dict]
    """
    if exclude_dirs is None:
        exclude_dirs = ["outputs", ".git", "__pycache__", "node_modules"]

    pattern = os.path.join(data_dir, "**/*.csv") if recursive else os.path.join(data_dir, "*.csv")
    paths = sorted(glob.glob(pattern, recursive=recursive))

    # Also check the root level (non-recursive match)
    if not paths:
        paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    # Filter out excluded directories
    def _is_excluded(p: str) -> bool:
        parts = Path(p).parts
        return any(ex in parts for ex in exclude_dirs)

    paths = [p for p in paths if not _is_excluded(p)]

    if not paths:
        warnings.warn(f"No CSV files found under '{data_dir}'.")
        return [], [], []

    dataframes, schemas, all_stats = [], [], []
    for p in paths:
        df = load_csv(p)
        if df.empty:
            continue
        df = _coerce_numeric(df)
        df = _parse_timestamp(df)
        df.sort_values("ts", inplace=True, ignore_index=True)

        schema = infer_schema(df, p)
        stats = summary_stats(df, schema)
        stats["per_product"] = per_product_stats(df, schema)

        dataframes.append(df)
        schemas.append(schema)
        all_stats.append(stats)

    return dataframes, schemas, all_stats


# ---------------------------------------------------------------------------
# Pretty printer (used in run_research.py)
# ---------------------------------------------------------------------------

def print_data_dictionary(schemas: List[dict], stats: List[dict]) -> None:
    """Print a human-readable data dictionary to stdout."""
    for sch, st in zip(schemas, stats):
        print(f"\n{'='*70}")
        print(f"FILE : {sch['path']}")
        print(f"TYPE : {sch['file_type']}  |  rows={st['n_rows']}  "
              f"duplicates={st['n_duplicate_rows']}  "
              f"missing={st['total_missing']}")
        print(f"COLS : {sch['columns']}")
        if st.get("ts_min") is not None:
            print(f"TIME : {st['ts_min']} → {st['ts_max']}")
        if "price_stats" in st:
            ps = st["price_stats"]
            print(f"PRICE: mean={ps['mean']:.2f}  std={ps['std']:.2f}  "
                  f"[{ps['min']:.2f}, {ps['max']:.2f}]  "
                  f"outliers={ps['n_outliers_iqr3']}")
        if "spread_proxy" in st:
            sp = st["spread_proxy"]
            if "mean" in sp:
                print(f"SPREAD: mean={sp['mean']:.4f}  median={sp['median']:.4f}")
            elif "estimate" in sp:
                print(f"SPREAD(fallback): {sp['fallback']} → {sp['estimate']:.4f}")
        print("\nPer-product stats:")
        for prod, ps2 in st.get("per_product", {}).items():
            pm = ps2.get('price_mean')
            pm_str = f"{pm:.2f}" if isinstance(pm, float) else str(pm)
            print(f"  [{prod}]  bars={ps2.get('n_bars', '?')}  "
                  f"price_mean={pm_str}  "
                  f"vol_daily={ps2.get('volatility_daily', 0):.6f}  "
                  f"spread_mean={ps2.get('spread_mean', '?')}")
