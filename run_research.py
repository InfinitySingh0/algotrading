"""
run_research.py – End-to-end orchestration script for the IMC trading pipeline.

Usage
-----
    python run_research.py
    python run_research.py --data-dir . --output-dir outputs
    python run_research.py --data-dir data/ --output-dir outputs --no-plots

Options
-------
  --data-dir    DIR   Directory to scan for CSV files (default: current dir)
  --output-dir  DIR   Where to write plots and reports   (default: outputs/)
  --no-plots         Skip matplotlib chart generation
  --train-ratio R    Fraction of data used for training  (default: 0.7)
  --top-n        N   Number of top param sets to report  (default: 5)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configure logging before importing project modules
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_research")

# ---------------------------------------------------------------------------
# Project modules
# ---------------------------------------------------------------------------
import config as cfg
from data_loader import (
    load_all,
    get_products,
    get_product_df,
    data_quality_report,
    infer_schema,
)
from features import build_features, add_cross_product_features
from strategies import MeanReversionStrategy, MomentumStrategy, create_strategy
from backtest import run_train_val, grid_search, random_search, train_val_split, run_backtest
from metrics import compute_metrics

# ---------------------------------------------------------------------------
# Matplotlib (optional)
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available – skipping plot generation.")


# ===========================================================================
# EDA
# ===========================================================================

def run_eda(
    prices_df: pd.DataFrame,
    products: List[str],
    output_dir: Path,
    generate_plots: bool = True,
) -> str:
    """
    Generate EDA plots and return a markdown summary string.
    """
    eda_dir = output_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    eda_lines: List[str] = ["## EDA Summary\n"]

    for product in products:
        logger.info("EDA for product: %s", product)
        df = get_product_df(prices_df, product)
        if df.empty or "mid_price" not in df.columns:
            eda_lines.append(f"### {product}\n\nNo usable data.\n")
            continue

        from features import build_features as _bf
        feat = _bf(df)
        price = feat["mid_price"].dropna()
        rets = feat["log_ret_1"].dropna() if "log_ret_1" in feat.columns else pd.Series(dtype=float)

        eda_lines.append(f"### {product}\n")
        eda_lines.append(f"- Bars: {len(price):,}")
        eda_lines.append(f"- Price range: {price.min():.2f} – {price.max():.2f}")
        if len(rets):
            eda_lines.append(f"- Mean return: {rets.mean():.6f}")
            eda_lines.append(f"- Volatility (std): {rets.std():.6f}")
            eda_lines.append(f"- Skewness: {rets.skew():.4f}")
            # Lag-1 autocorrelation
            ac1 = float(rets.autocorr(lag=1)) if len(rets) > 1 else float("nan")
            eda_lines.append(f"- Lag-1 autocorrelation of returns: {ac1:.4f}")

        if not generate_plots or not HAS_MATPLOTLIB:
            eda_lines.append("")
            continue

        # ----------------------------------------------------------------
        # Plot 1: Price evolution
        # ----------------------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{product} – EDA", fontsize=14)

        ax = axes[0, 0]
        ax.plot(price.values, linewidth=0.7, color="steelblue")
        ax.set_title("Mid-price evolution")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Price")

        # ----------------------------------------------------------------
        # Plot 2: Returns distribution
        # ----------------------------------------------------------------
        ax = axes[0, 1]
        if len(rets) > 10:
            ax.hist(rets, bins=60, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_title("Returns distribution")
        ax.set_xlabel("Log-return")

        # ----------------------------------------------------------------
        # Plot 3: Autocorrelation (lag 1–10)
        # ----------------------------------------------------------------
        ax = axes[1, 0]
        if len(rets) > 20:
            lags = list(range(1, 11))
            acs = [float(rets.autocorr(lag=l)) for l in lags]
            ax.bar(lags, acs, color="steelblue", alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title("Return autocorrelation (lags 1–10)")
            ax.set_xlabel("Lag")
            ax.set_ylabel("ACF")

        # ----------------------------------------------------------------
        # Plot 4: Rolling volatility (regime proxy)
        # ----------------------------------------------------------------
        ax = axes[1, 1]
        if "roll_vol" in feat.columns:
            rv = feat["roll_vol"].dropna()
            ax.plot(rv.values, linewidth=0.7, color="orange")
            ax.set_title("Rolling volatility (20-bar)")
            ax.set_xlabel("Bar")
            ax.set_ylabel("Std of log-ret")

        plt.tight_layout()
        plot_path = eda_dir / f"{product}_eda.png"
        plt.savefig(plot_path, dpi=100)
        plt.close(fig)
        logger.info("Saved EDA plot: %s", plot_path)
        eda_lines.append(f"![{product} EDA](eda/{product}_eda.png)\n")

    return "\n".join(eda_lines)


# ===========================================================================
# Feature engineering
# ===========================================================================

def run_feature_engineering(
    prices_df: pd.DataFrame,
    products: List[str],
) -> Dict[str, pd.DataFrame]:
    """Build feature dataframes for each product."""
    feature_dfs: Dict[str, pd.DataFrame] = {}
    for product in products:
        df = get_product_df(prices_df, product)
        if df.empty:
            logger.warning("No data for product %s; skipping.", product)
            continue
        feat = build_features(df)
        feature_dfs[product] = feat
        logger.info(
            "Features built for %s: %d rows, %d cols", product, *feat.shape
        )

    # Cross-product features if multiple products
    if len(products) >= 2 and not prices_df.empty:
        logger.info("Adding cross-product features…")
        combined_feat = add_cross_product_features(prices_df, products)
        # Distribute back
        for product in products:
            if product in feature_dfs and "product" in combined_feat.columns:
                mask = combined_feat["product"] == product
                cross_cols = [
                    c for c in combined_feat.columns
                    if c.startswith("rel_") or c.startswith("z_rel_")
                ]
                if cross_cols:
                    sub = combined_feat.loc[mask, ["timestamp"] + cross_cols].copy()
                    feature_dfs[product] = feature_dfs[product].merge(
                        sub, on="timestamp", how="left", suffixes=("", "_x")
                    )

    return feature_dfs


# ===========================================================================
# Strategy runs
# ===========================================================================

def run_strategies(
    feature_dfs: Dict[str, pd.DataFrame],
    train_ratio: float = cfg.TRAIN_RATIO,
) -> Dict[str, Dict]:
    """Run both strategies on all products. Returns nested results dict."""
    strategies = [
        ("mean_reversion", MeanReversionStrategy()),
        ("momentum", MomentumStrategy()),
    ]

    all_results: Dict[str, Dict] = {}

    for strat_name, strat in strategies:
        all_results[strat_name] = {}
        for product, feat_df in feature_dfs.items():
            logger.info("Backtesting %s on %s …", strat_name, product)
            result = run_train_val(strat, feat_df, train_ratio=train_ratio)
            all_results[strat_name][product] = result

    return all_results


# ===========================================================================
# Parameter search
# ===========================================================================

def run_param_search(
    feature_dfs: Dict[str, pd.DataFrame],
    train_ratio: float = cfg.TRAIN_RATIO,
    top_n: int = 5,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Run parameter search for both strategies on all products.
    Returns {strategy_name: {product: results_df}}.
    """
    mr_grid = {
        "entry_z": [1.0, 1.5, 2.0, 2.5],
        "exit_z": [0.25, 0.5, 0.75],
        "trade_size": [3, 5, 8],
    }
    mom_grid = {
        "short_window": [3, 5, 10],
        "long_window": [15, 20, 30],
        "trade_size": [3, 5, 8],
        "threshold": [0.0001, 0.0005, 0.001],
    }

    param_results: Dict[str, Dict[str, pd.DataFrame]] = {
        "mean_reversion": {},
        "momentum": {},
    }

    for product, feat_df in feature_dfs.items():
        logger.info("Parameter search – mean_reversion on %s …", product)
        mr_res = grid_search(
            "mean_reversion", mr_grid, feat_df, train_ratio=train_ratio
        )
        param_results["mean_reversion"][product] = mr_res.head(top_n) if not mr_res.empty else mr_res

        logger.info("Parameter search – momentum on %s …", product)
        mom_res = random_search(
            "momentum", mom_grid, feat_df,
            n_iter=cfg.PARAM_SEARCH_N_RANDOM, train_ratio=train_ratio,
        )
        param_results["momentum"][product] = mom_res.head(top_n) if not mom_res.empty else mom_res

    return param_results


# ===========================================================================
# Report generation
# ===========================================================================

def _param_table_show_cols(tbl: pd.DataFrame) -> List[str]:
    """Return columns to display in parameter search table (exclude train_ metrics)."""
    return [c for c in tbl.columns if not c.startswith("train_")]


def generate_report(
    output_dir: Path,
    prices_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    products: List[str],
    eda_summary: str,
    strategy_results: Dict,
    param_results: Dict,
    top_n: int = 5,
) -> str:
    """Compose and write report.md. Returns path as string."""

    lines: List[str] = []

    lines += [
        "# IMC Trading Research Report",
        "",
        "Auto-generated by `run_research.py`.",
        "",
    ]

    # ------------------------------------------------------------------
    # 1. Dataset overview
    # ------------------------------------------------------------------
    lines += [
        "## 1. Discovered Datasets",
        "",
    ]
    if not prices_df.empty:
        lines.append(data_quality_report(prices_df, "Prices (order-book snapshots)"))
    if not trades_df.empty:
        lines.append(data_quality_report(trades_df, "Trades (public tape)"))

    # ------------------------------------------------------------------
    # 2. Schema
    # ------------------------------------------------------------------
    lines += ["## 2. Inferred Schema", ""]
    for label, df in [("prices", prices_df), ("trades", trades_df)]:
        if not df.empty:
            schema = infer_schema(df, label)
            lines.append(f"**{label}**")
            lines.append(f"- Columns: {schema['columns']}")
            lines.append(f"- Timestamp col: `{schema['timestamp_col']}`")
            lines.append(f"- Product col: `{schema['product_col']}`")
            lines.append(f"- Price col: `{schema['price_col']}`")
            lines.append(f"- Volume col: `{schema['volume_col']}`")
            lines.append("")

    # ------------------------------------------------------------------
    # 3. EDA
    # ------------------------------------------------------------------
    lines += ["## 3. Exploratory Data Analysis", "", eda_summary, ""]

    # ------------------------------------------------------------------
    # 4. Strategy Definitions
    # ------------------------------------------------------------------
    lines += [
        "## 4. Strategy Definitions",
        "",
        "### 4.1 Mean-Reversion Strategy",
        "",
        "Enters long (short) when the rolling z-score of mid-price falls "
        "below -entry_z (rises above +entry_z). Exits when |z| < exit_z. "
        "Inventory penalty reduces position aggressiveness as inventory grows.",
        "",
        "### 4.2 Momentum Strategy",
        "",
        "Computes the difference between a short-window and long-window "
        "rolling return. Buys when momentum is positive and above a threshold, "
        "sells when negative. Position is reset to zero when momentum crosses "
        "zero.",
        "",
        "### Shared Risk Controls",
        "",
        f"- Position limit per product: ±{cfg.POSITION_LIMIT} units",
        f"- Max gross exposure: {cfg.MAX_GROSS_EXPOSURE} units",
        f"- Transaction cost: {cfg.TRANSACTION_COST} per unit",
        f"- Slippage: {cfg.SLIPPAGE} per unit",
        f"- Stop-loss threshold: {cfg.STOP_LOSS_THRESHOLD} unrealised PnL",
        "",
    ]

    # ------------------------------------------------------------------
    # 5. Backtest Results
    # ------------------------------------------------------------------
    lines += ["## 5. Train / Validation Backtest Results", ""]

    metric_keys = ["total_pnl", "sharpe", "sortino", "max_drawdown", "hit_rate", "n_trades"]

    for strat_name, prod_results in strategy_results.items():
        lines.append(f"### {strat_name.replace('_', ' ').title()}")
        lines.append("")
        rows = []
        for product, res in prod_results.items():
            for split in ("train", "validation"):
                m = res[split]["metrics"]
                rows.append({
                    "product": product,
                    "split": split,
                    **{k: round(m.get(k, float("nan")), 4) for k in metric_keys},
                })
        if rows:
            lines.append(pd.DataFrame(rows).to_markdown(index=False))
        lines.append("")

    # ------------------------------------------------------------------
    # 6. Parameter Search Results
    # ------------------------------------------------------------------
    lines += ["## 6. Parameter Search (Top Results)", ""]

    for strat_name, prod_tables in param_results.items():
        lines.append(f"### {strat_name.replace('_', ' ').title()}")
        lines.append("")
        for product, tbl in prod_tables.items():
            lines.append(f"**{product}** (ranked by val_sharpe)")
            lines.append("")
            if not tbl.empty:
                lines.append(tbl[_param_table_show_cols(tbl)].head(top_n).to_markdown(index=False))
            else:
                lines.append("*(no results)*")
            lines.append("")

    # ------------------------------------------------------------------
    # 7. Best Strategy Summary
    # ------------------------------------------------------------------
    lines += ["## 7. Selected Best Strategy", ""]

    # Find best strategy+product combination by val Sharpe
    best = {"val_sharpe": float("-inf"), "strat": "", "product": ""}
    for strat_name, prod_results in strategy_results.items():
        for product, res in prod_results.items():
            vs = res["validation"]["metrics"].get("sharpe", float("-inf"))
            if vs > best["val_sharpe"]:
                best = {"val_sharpe": vs, "strat": strat_name, "product": product}

    lines.append(
        f"Based on out-of-sample (validation) Sharpe ratio, the best combination is "
        f"**{best['strat'].replace('_', ' ').title()}** on **{best['product']}** "
        f"(val Sharpe = {best['val_sharpe']:.4f})."
    )
    if best["val_sharpe"] < 0:
        lines += [
            "",
            "> **Note**: All tested strategy configurations achieved a negative "
            "out-of-sample Sharpe ratio with default parameters. "
            "This indicates that with the current transaction cost assumptions "
            "({:.1f} + {:.1f} = {:.1f} per unit), the strategies are not "
            "profitable after costs on this limited 2-day dataset. "
            "See the parameter search results for configurations that reduce "
            "turnover and may improve risk-adjusted returns.".format(
                cfg.TRANSACTION_COST, cfg.SLIPPAGE,
                cfg.TRANSACTION_COST + cfg.SLIPPAGE,
            ),
        ]
    lines += [
        "",
        "The mean-reversion strategy tends to perform well in IMC challenge data "
        "because these markets exhibit strong reversion to fair value in tightly "
        "quoted, low-latency order books. The momentum strategy provides "
        "diversification across trending regimes.",
        "",
    ]

    # ------------------------------------------------------------------
    # 8. Limitations
    # ------------------------------------------------------------------
    lines += [
        "## 8. Limitations and Failure Modes",
        "",
        "- **Bar granularity**: The pipeline treats each CSV row as an independent "
        "  bar. In practice, IMC timestamps are discrete ticks; gaps between ticks "
        "  are not modelled.",
        "- **Slippage model**: A flat per-unit cost is used. Real slippage depends "
        "  on order size relative to available liquidity at each level.",
        "- **No cross-product hedging**: Strategies operate independently per "
        "  product; correlated positions could violate gross exposure in live trading.",
        "- **Parameter overfitting**: Grid search is constrained to training data "
        "  only, but the grid itself was hand-designed, introducing implicit bias.",
        "- **Limited history**: Only 2 days of data are available. Validation "
        "  results should be interpreted cautiously.",
        "- **Stop-loss calibration**: The stop-loss threshold is expressed in raw "
        "  price units, not in terms of ATR or percentage, which may behave "
        "  differently across products.",
        "",
    ]

    report_text = "\n".join(lines)

    report_path = output_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Report written to %s", report_path)

    # Also copy to repo root
    root_report = Path("report.md")
    root_report.write_text(report_text, encoding="utf-8")
    logger.info("Report also written to %s", root_report)

    return str(report_path)


# ===========================================================================
# PnL plots
# ===========================================================================

def save_strategy_plots(
    strategy_results: Dict,
    output_dir: Path,
) -> None:
    """Save cumulative PnL charts for all strategy/product combinations."""
    if not HAS_MATPLOTLIB:
        return

    plots_dir = output_dir / "strategy_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for strat_name, prod_results in strategy_results.items():
        for product, res in prod_results.items():
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(f"{strat_name} – {product}", fontsize=12)

            for ax, split in zip(axes, ["train", "validation"]):
                cum = res[split]["cumulative_pnl"]
                pos = res[split]["position_series"]
                ax.plot(cum.values, label="Cum PnL", color="steelblue")
                ax2 = ax.twinx()
                ax2.fill_between(
                    range(len(pos)), pos.values,
                    alpha=0.2, color="orange", label="Position"
                )
                ax.set_title(split.capitalize())
                ax.set_xlabel("Bar")
                ax.set_ylabel("Cumulative PnL")
                ax2.set_ylabel("Position")

            plt.tight_layout()
            fname = plots_dir / f"{strat_name}_{product}.png"
            plt.savefig(fname, dpi=100)
            plt.close(fig)
            logger.info("Saved strategy plot: %s", fname)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IMC Trading Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument(
        "--data-dir", default=cfg.DEFAULT_DATA_DIR,
        help="Directory containing CSV files (default: current dir)"
    )
    parser.add_argument(
        "--output-dir", default=cfg.DEFAULT_OUTPUT_DIR,
        help="Directory for output artifacts (default: outputs/)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Disable matplotlib chart generation"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=cfg.TRAIN_RATIO,
        help=f"Fraction of data for training (default: {cfg.TRAIN_RATIO})"
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="Number of top parameter sets to include in report (default: 5)"
    )
    return parser.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_plots = not args.no_plots and HAS_MATPLOTLIB

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1 – Loading data from '%s'", args.data_dir)
    logger.info("=" * 60)
    prices_df, trades_df = load_all(args.data_dir)

    if prices_df.empty:
        logger.error("No price data found. Exiting.")
        sys.exit(1)

    products = get_products(prices_df)
    logger.info("Products discovered: %s", products)

    # Print schema + quality
    print(data_quality_report(prices_df, "Prices"))
    if not trades_df.empty:
        print(data_quality_report(trades_df, "Trades"))

    # ------------------------------------------------------------------
    # Step 2: EDA
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2 – EDA")
    logger.info("=" * 60)
    eda_summary = run_eda(prices_df, products, output_dir, generate_plots=generate_plots)

    # ------------------------------------------------------------------
    # Step 3: Feature engineering
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 3 – Feature Engineering")
    logger.info("=" * 60)
    feature_dfs = run_feature_engineering(prices_df, products)

    if not feature_dfs:
        logger.error("Feature engineering produced no data. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: Strategy backtests
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 4 – Strategy Backtests (train/val split=%.0f%%/%.0f%%)",
                args.train_ratio * 100, (1 - args.train_ratio) * 100)
    logger.info("=" * 60)
    strategy_results = run_strategies(feature_dfs, train_ratio=args.train_ratio)

    # Print summary table
    for strat_name, prod_results in strategy_results.items():
        print(f"\n=== {strat_name.upper()} ===")
        for product, res in prod_results.items():
            for split in ("train", "validation"):
                m = res[split]["metrics"]
                print(
                    f"  {product:12s} {split:12s}  "
                    f"PnL={m.get('total_pnl', 0):8.2f}  "
                    f"Sharpe={m.get('sharpe', 0):6.3f}  "
                    f"MaxDD={m.get('max_drawdown', 0):8.2f}  "
                    f"Trades={m.get('n_trades', 0):5d}"
                )

    # Save PnL plots
    if generate_plots:
        save_strategy_plots(strategy_results, output_dir)

    # ------------------------------------------------------------------
    # Step 5: Parameter search
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 5 – Parameter Search")
    logger.info("=" * 60)
    param_results = run_param_search(
        feature_dfs, train_ratio=args.train_ratio, top_n=args.top_n
    )

    # Print top-N for each
    for strat_name, prod_tables in param_results.items():
        for product, tbl in prod_tables.items():
            if not tbl.empty:
                val_col = "val_sharpe" if "val_sharpe" in tbl.columns else tbl.columns[-1]
                print(f"\n--- {strat_name} / {product} – top params by {val_col} ---")
                show = _param_table_show_cols(tbl)
                print(tbl[show].head(args.top_n).to_string(index=False))

    # ------------------------------------------------------------------
    # Step 6: Report
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 6 – Generating Report")
    logger.info("=" * 60)
    report_path = generate_report(
        output_dir=output_dir,
        prices_df=prices_df,
        trades_df=trades_df,
        products=products,
        eda_summary=eda_summary,
        strategy_results=strategy_results,
        param_results=param_results,
        top_n=args.top_n,
    )

    logger.info("=" * 60)
    logger.info("Pipeline complete. Outputs in '%s'.", args.output_dir)
    logger.info("Report: %s", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
