"""
run_research.py
===============
End-to-end CLI entry point for the IMC Trading Challenge research pipeline.

Usage
-----
    python run_research.py --data-dir .
    python run_research.py --data-dir data --out-dir outputs --search random

Pipeline steps
--------------
1. Discover & load CSV data
2. Print data dictionary + summary stats
3. Feature engineering per product
4. EDA charts (saved to out-dir/figures)
5. Backtest (mean-reversion + momentum) on train / validation splits
6. Parameter search
7. Write markdown report
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Suppress noisy warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from data_loader import discover_and_load, print_data_dictionary
from features import build_features, build_cross_product_features
from strategies import MeanReversionStrategy, MomentumStrategy, MeanReversionParams, MomentumParams
from backtest import (
    run_backtest,
    train_val_split,
    parameter_search,
    ExecutionParams,
    metrics_to_markdown,
)


# ---------------------------------------------------------------------------
# EDA / plotting
# ---------------------------------------------------------------------------

def _save_eda_charts(
    df: pd.DataFrame,
    product: str,
    schema: dict,
    out_dir: str,
) -> List[str]:
    """
    Generate and save EDA charts for one product.
    Returns list of saved file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not available; skipping charts.")
        return []

    figs_dir = os.path.join(out_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)
    saved = []

    if "mid_price" not in df.columns:
        return saved

    price = df["mid_price"].dropna()
    ts = df["ts"] if "ts" in df.columns else pd.Series(range(len(df)))

    # --- 1. Mid-price time series ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ts[:len(price)], price.values, linewidth=0.8)
    ax.set_title(f"{product} – Mid-price over time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = os.path.join(figs_dir, f"{product}_midprice.png")
    fig.savefig(path1, dpi=100)
    plt.close(fig)
    saved.append(path1)

    # --- 2. Returns distribution ---
    log_ret = np.log(price / price.shift(1)).dropna()
    if len(log_ret) > 10:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(log_ret.values, bins=min(60, len(log_ret) // 5 + 1), color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_title(f"{product} – Log-returns distribution")
        ax.set_xlabel("Log-return")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path2 = os.path.join(figs_dir, f"{product}_returns_dist.png")
        fig.savefig(path2, dpi=100)
        plt.close(fig)
        saved.append(path2)

    # --- 3. Autocorrelation (lag plot) ---
    if len(log_ret) > 20:
        max_lag = min(40, len(log_ret) // 4)
        lags = range(1, max_lag + 1)
        # Filter out NaN ACF values (can occur at high lags with insufficient data)
        acf_pairs = [
            (lag, float(log_ret.autocorr(lag=lag)))
            for lag in lags
        ]
        acf_pairs = [(lag, v) for lag, v in acf_pairs if not np.isnan(v)]
        if acf_pairs:
            plot_lags, acf_vals = zip(*acf_pairs)
            ci = 1.96 / np.sqrt(len(log_ret))

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(list(plot_lags), list(acf_vals), color="steelblue", alpha=0.7)
            ax.axhline(ci, color="red", linestyle="--", linewidth=0.8, label="95% CI")
            ax.axhline(-ci, color="red", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title(f"{product} – Return autocorrelation")
            ax.set_xlabel("Lag (bars)")
            ax.set_ylabel("ACF")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path3 = os.path.join(figs_dir, f"{product}_acf.png")
            fig.savefig(path3, dpi=100)
            plt.close(fig)
            saved.append(path3)

    # --- 4. Intraday pattern (if ts is numeric and has meaningful range) ---
    if "ts" in df.columns and pd.api.types.is_numeric_dtype(df["ts"]):
        ts_vals = df["ts"].astype(float)
        rng = ts_vals.max() - ts_vals.min()
        if rng > 0 and "ret_1" in df.columns:
            # Bin by normalised intraday time
            n_bins = 20
            df2 = df[["ts", "ret_1"]].dropna()
            df2 = df2.copy()
            df2["bin"] = pd.cut(
                (df2["ts"] - df2["ts"].min()) / rng,
                bins=n_bins, labels=False
            )
            intraday = df2.groupby("bin")["ret_1"].mean()
            if len(intraday) > 3:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.bar(intraday.index, intraday.values, color="darkorange", alpha=0.7)
                ax.axhline(0, color="black", linewidth=0.5)
                ax.set_title(f"{product} – Mean return by intraday bin")
                ax.set_xlabel("Intraday bin (0=open, 19=close)")
                ax.set_ylabel("Mean log-return")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                path4 = os.path.join(figs_dir, f"{product}_intraday.png")
                fig.savefig(path4, dpi=100)
                plt.close(fig)
                saved.append(path4)

    return saved


def _save_pnl_chart(result, out_dir: str) -> str:
    """Save cumulative PnL chart for a backtest result."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    figs_dir = os.path.join(out_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(result.pnl_series.values, linewidth=0.8, label="Realized PnL")
    axes[0].fill_between(
        range(len(result.unrealized_series)),
        result.unrealized_series.values,
        alpha=0.2, label="Unrealized"
    )
    axes[0].set_title(f"{result.product} – {result.strategy_name} – Cumulative PnL")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result.inventory_series.values, linewidth=0.8, color="darkorange")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title("Inventory")
    axes[1].set_xlabel("Bar")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(figs_dir, f"{result.product}_{result.strategy_name}_pnl.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    schemas: list,
    all_stats: list,
    backtest_results: list,
    param_search_results: dict,
    out_dir: str,
) -> str:
    """
    Assemble a markdown report and write it to {out_dir}/report.md.
    Returns the path.
    """
    lines = []
    lines.append("# IMC Trading Challenge – Research Pipeline Report\n")
    lines.append(f"*Generated automatically by `run_research.py`*\n")
    lines.append("")

    # --- Dataset summary ---
    lines.append("## 1. Dataset Summary\n")
    lines.append("| File | Type | Rows | Products | Timestamp Range | Missing |")
    lines.append("|------|------|------|----------|-----------------|---------|")
    for sch, st in zip(schemas, all_stats):
        products = list(st.get("per_product", {}).keys())
        ts_range = f"{st.get('ts_min', '?')} → {st.get('ts_max', '?')}"
        lines.append(
            f"| {os.path.basename(sch['path'])} "
            f"| {sch['file_type']} "
            f"| {st['n_rows']} "
            f"| {', '.join(products[:5])} "
            f"| {ts_range} "
            f"| {st['total_missing']} |"
        )
    lines.append("")

    # Per-product stats
    lines.append("### Per-product statistics\n")
    lines.append("| File | Product | Bars | Price Mean | Daily Vol | Spread Mean | Liquidity Proxy |")
    lines.append("|------|---------|------|-----------|-----------|-------------|-----------------|")
    for sch, st in zip(schemas, all_stats):
        for prod, ps in st.get("per_product", {}).items():
            lines.append(
                f"| {os.path.basename(sch['path'])} "
                f"| {prod} "
                f"| {ps.get('n_bars', '?')} "
                f"| {ps.get('price_mean', 0):.2f} "
                f"| {ps.get('volatility_daily', 0):.6f} "
                f"| {ps.get('spread_mean', '?')} "
                f"| {ps.get('liquidity_proxy', '?')} |"
            )
    lines.append("")

    # --- Strategy descriptions ---
    lines.append("## 2. Strategy Descriptions\n")
    lines.append("""### 2.1 Mean-Reversion Strategy

**Hypothesis:** IMC challenge prices mean-revert around a slowly drifting fair value,
driven by continuous market-maker quoting and noise-trader flow.

**Signal generation:**
- Compute rolling z-score: `z = (price − μ_w) / σ_w` with window `w`.
- Enter **short** when `z > entry_z` (price above fair value).
- Enter **long**  when `z < -entry_z` (price below fair value).
- Exit when `|z| < exit_z`.

**Risk controls:** position limit, gross-exposure cap, stop-loss on unrealized loss,
inventory penalty to discourage runaway positions.

**Why it works:** Empirically, bid-ask spreads in IMC-style LOB data are wide relative
to tick size, and prices oscillate within them.  The z-score identifies the regime
where price has drifted unusually far, creating a high-probability reversion trade.

---

### 2.2 Momentum / Trend-Following Strategy

**Hypothesis:** Short bursts of momentum exist when information arrives and the market
is discovering a new equilibrium.

**Signal generation:**
- Fast momentum = `log(price_t / price_{t−fast})`, filtered by slow momentum sign.
- Only enter if both fast and slow momentum agree in direction.
- Scale position size by inverse rolling volatility (volatility targeting).

**Risk controls:** Same as above.

**Why it works:** When new fundamental information (e.g., a reference-price update) hits
the market, the initial move tends to be under-reacted to and continuation is observed
for a few bars before mean-reversion re-asserts.

---
""")

    # --- Backtest results ---
    lines.append("## 3. Backtest Results\n")
    lines.append("| Product | Strategy | Split | Total PnL | Sharpe | Max DD | Turnover | Trades | Hit Rate |")
    lines.append("|---------|----------|-------|-----------|--------|--------|----------|--------|----------|")
    for product, split, result in backtest_results:
        lines.append(metrics_to_markdown(result, split))
    lines.append("")

    # --- Parameter search ---
    lines.append("## 4. Parameter Search Results\n")
    for strat_name, search_df in param_search_results.items():
        lines.append(f"### {strat_name}\n")
        if search_df is None or len(search_df) == 0:
            lines.append("_No results._\n")
            continue
        # Show top 10
        top = search_df.head(10)
        # Select key columns
        show_cols = [c for c in top.columns if not c.startswith("train_")]
        try:
            lines.append(top[show_cols].to_markdown(index=False))
        except ImportError:
            header = "| " + " | ".join(str(c) for c in show_cols) + " |"
            sep = "|" + "|".join(["---"] * len(show_cols)) + "|"
            lines.append(header)
            lines.append(sep)
            for _, row in top[show_cols].iterrows():
                lines.append("| " + " | ".join(str(v) for v in row) + " |")
        lines.append("")

    # --- Limitations ---
    lines.append("## 5. Limitations and Failure Modes\n")
    lines.append("""- **Data size:** With only 2 days of data (~20k bars per day per product), all estimates
  are noisy.  Wider parameter search on more data is recommended.
- **Slippage model:** A simple percentage-of-price model is used.  Real fill uncertainty
  can be much higher at the top of book.
- **Parameter overfitting:** Grid search on limited data can overfit.  Out-of-sample
  Sharpe should be treated as an upper bound.
- **Single-product backtesting:** Cross-product features are computed but the backtest
  engine handles each product independently.  Portfolio-level correlation and netting
  are not modelled.
- **No tick-level simulation:** The bar-based engine assumes fills at the midprice.
  In reality, limit orders may not fill and market orders pay the spread.
- **Mean-reversion failure:** If the market enters a persistent trending regime
  (e.g., around a major news event), the mean-reversion strategy will accumulate losses.
- **Momentum failure:** In stable, tight-spread markets the momentum signal will trade
  noise and destroy value through transaction costs.

## 6. Assumptions

- CSV files use semicolon delimiters as found in the IMC round-0 data.
- `mid_price` is used as the fill price; actual fills would use bid for sells, ask for buys.
- Transaction cost: 0.05% of trade value.  Slippage: 0.02%.
- Position limit: 10 units per product.
- Train/validation split: 70% / 30% (time-ordered).
""")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))
    return report_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data_dir: str = ".",
    out_dir: str = "outputs",
    train_ratio: float = 0.7,
    search_mode: str = "grid",
    n_random: int = 30,
    seed: int = 42,
    skip_plots: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ---- Step 1: Load data ----
    print("\n[1/7] Discovering and loading CSV files...")
    dataframes, schemas, all_stats = discover_and_load(data_dir)
    if not dataframes:
        print(f"ERROR: No CSV files found in '{data_dir}'. Exiting.")
        sys.exit(1)
    print(f"  Loaded {len(dataframes)} file(s).")
    print_data_dictionary(schemas, all_stats)

    # ---- Step 2: Separate prices vs trades ----
    prices_dfs = [(df, sch, st) for df, sch, st in zip(dataframes, schemas, all_stats)
                  if sch["file_type"] == "prices"]
    trades_dfs = [(df, sch, st) for df, sch, st in zip(dataframes, schemas, all_stats)
                  if sch["file_type"] == "trades"]
    print(f"\n  Prices files: {len(prices_dfs)}, Trades files: {len(trades_dfs)}")

    # ---- Step 3: Feature engineering ----
    print("\n[2/7] Building features...")

    # Combine all prices into per-product DataFrames
    product_frames: Dict[str, pd.DataFrame] = {}
    for df, sch, _st in prices_dfs:
        prod_col = sch.get("product_col")
        if prod_col and prod_col in df.columns:
            for prod, grp in df.groupby(prod_col):
                prod = str(prod)
                feat_df = build_features(grp.reset_index(drop=True), sch)
                if prod in product_frames:
                    product_frames[prod] = pd.concat(
                        [product_frames[prod], feat_df], ignore_index=True
                    ).sort_values("ts").reset_index(drop=True)
                else:
                    product_frames[prod] = feat_df
        else:
            feat_df = build_features(df, sch)
            key = os.path.basename(sch["path"])
            product_frames[key] = feat_df

    if not product_frames:
        print("  [WARNING] No price data found; using first available dataframe.")
        df0, sch0, _ = dataframes[0], schemas[0], all_stats[0]
        feat_df = build_features(df0, sch0)
        product_frames["UNKNOWN"] = feat_df

    print(f"  Products: {list(product_frames.keys())}")

    # Cross-product features
    product_frames = build_cross_product_features(product_frames)

    # ---- Step 4: EDA charts ----
    print("\n[3/7] Generating EDA charts...")
    all_eda_paths = []
    if not skip_plots:
        for prod, feat_df in product_frames.items():
            sch_for_prod = schemas[0] if schemas else {}
            paths = _save_eda_charts(feat_df, prod, sch_for_prod, out_dir)
            all_eda_paths.extend(paths)
            if paths:
                print(f"  Saved {len(paths)} chart(s) for {prod}")
    else:
        print("  (skipped)")

    # ---- Step 5: Backtesting ----
    print("\n[4/7] Running backtests...")
    exec_p = ExecutionParams(
        transaction_cost_pct=0.0005,
        slippage_pct=0.0002,
        position_limit=10,
        max_gross_exposure=20.0,
        stop_loss_pct=0.02,
    )

    backtest_results = []  # [(product, split_label, BacktestResult)]

    for prod, feat_df in product_frames.items():
        if len(feat_df) < 50:
            print(f"  [SKIP] {prod}: too few bars ({len(feat_df)})")
            continue

        train_df, val_df = train_val_split(feat_df, train_ratio)
        print(f"\n  [{prod}] train={len(train_df)} bars, val={len(val_df)} bars")

        for strat_cls, strat_name in [
            (MeanReversionStrategy, "mean_reversion"),
            (MomentumStrategy, "momentum"),
        ]:
            strat = strat_cls()
            for split_label, split_df in [("train", train_df), ("val", val_df)]:
                result = run_backtest(split_df, strat, exec_p, price_col="mid_price", product=prod)
                print(f"    {strat_name} [{split_label}]: "
                      f"PnL={result.metrics.get('total_pnl', 0):.2f}  "
                      f"Sharpe={result.metrics.get('sharpe', 0):.4f}  "
                      f"DD={result.metrics.get('max_drawdown', 0):.2f}")
                backtest_results.append((prod, split_label, result))
                if not skip_plots:
                    _save_pnl_chart(result, out_dir)

    # ---- Step 6: Parameter search ----
    print("\n[5/7] Running parameter search...")
    param_search_results: Dict[str, pd.DataFrame] = {}

    mr_grid = {
        "window": [20, 40, 60],
        "entry_z": [1.0, 1.5, 2.0],
        "exit_z": [0.2, 0.5],
    }
    mom_grid = {
        "window_fast": [10, 20, 40],
        "window_slow": [40, 60, 80],
        "entry_threshold": [0.0005, 0.001, 0.002],
    }

    for prod, feat_df in product_frames.items():
        if len(feat_df) < 100:
            continue
        print(f"  Searching mean_reversion params for {prod}...")
        mr_search = parameter_search(
            feat_df, MeanReversionStrategy, mr_grid,
            exec_params=exec_p, train_ratio=train_ratio,
            search_mode=search_mode, n_random_samples=n_random, seed=seed,
            price_col="mid_price", product=prod,
        )
        param_search_results.setdefault("mean_reversion", [])
        if len(mr_search):
            param_search_results["mean_reversion"].append(mr_search)

        print(f"  Searching momentum params for {prod}...")
        mom_search = parameter_search(
            feat_df, MomentumStrategy, mom_grid,
            exec_params=exec_p, train_ratio=train_ratio,
            search_mode=search_mode, n_random_samples=n_random, seed=seed,
            price_col="mid_price", product=prod,
        )
        param_search_results.setdefault("momentum", [])
        if len(mom_search):
            param_search_results["momentum"].append(mom_search)

    # Combine per-product search results
    combined_search: Dict[str, Optional[pd.DataFrame]] = {}
    for strat_name, dfs in param_search_results.items():
        if dfs:
            combined_search[strat_name] = pd.concat(dfs, ignore_index=True).sort_values(
                "val_sharpe", ascending=False
            ).reset_index(drop=True)
        else:
            combined_search[strat_name] = None

    # Save to CSV
    for strat_name, df in combined_search.items():
        if df is not None and len(df):
            csv_path = os.path.join(out_dir, f"param_search_{strat_name}.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Saved {strat_name} search results to {csv_path}")

    # ---- Step 7: Write report ----
    print("\n[6/7] Writing report...")
    report_path = build_report(schemas, all_stats, backtest_results, combined_search, out_dir)
    print(f"  Report written to {report_path}")

    print("\n[7/7] Done!")
    print(f"  Outputs saved to '{out_dir}/'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IMC Trading Challenge – Research & Backtesting Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default=".",
        help="Directory to search for CSV files (recursive)."
    )
    parser.add_argument(
        "--out-dir", default="outputs",
        help="Output directory for figures and reports."
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7,
        help="Fraction of data used for training (0-1)."
    )
    parser.add_argument(
        "--search", choices=["grid", "random"], default="grid",
        help="Parameter search mode."
    )
    parser.add_argument(
        "--n-random", type=int, default=30,
        help="Number of random samples (only for --search random)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating matplotlib figures."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        search_mode=args.search,
        n_random=args.n_random,
        seed=args.seed,
        skip_plots=args.no_plots,
    )
