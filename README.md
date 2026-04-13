# IMC Trading Challenge – Algorithmic Trading Research Pipeline

A complete Python-based **research and backtesting pipeline** for analysing IMC Trading Challenge style LOB (Limit Order Book) data.

---

## Project Structure

```
algotrading/
├── data_loader.py      # CSV discovery, schema inference, summary stats
├── features.py         # Feature engineering (momentum, z-score, vol, etc.)
├── strategies.py       # Mean-reversion + Momentum strategies with risk controls
├── backtest.py         # Bar-based backtester, parameter search, metrics
├── run_research.py     # End-to-end CLI entry point
├── outputs/            # Generated artefacts (report, figures, CSV results)
│   ├── report.md
│   ├── figures/
│   └── param_search_*.csv
└── *.csv               # IMC round-0 price and trade data files
```

---

## Setup

### Requirements

- Python 3.9+
- `pandas`, `numpy`, `matplotlib`, `tabulate`

### Install dependencies

```bash
pip install pandas numpy matplotlib tabulate
```

> `scipy` and `seaborn` are optional (not required by default).

---

## Running the Pipeline

### Quick start (data files in current directory)

```bash
python run_research.py --data-dir .
```

### With a dedicated data folder

```bash
mkdir -p data
cp *.csv data/
python run_research.py --data-dir data --out-dir outputs
```

### Skip plot generation (faster)

```bash
python run_research.py --data-dir . --no-plots
```

### Random parameter search instead of grid search

```bash
python run_research.py --data-dir . --search random --n-random 50
```

### Full options

```
usage: run_research.py [-h] [--data-dir DATA_DIR] [--out-dir OUT_DIR]
                       [--train-ratio TRAIN_RATIO] [--search {grid,random}]
                       [--n-random N_RANDOM] [--seed SEED] [--no-plots]

options:
  --data-dir      Directory containing CSV files (default: .)
  --out-dir       Output directory for report and figures (default: outputs)
  --train-ratio   Fraction of data for training, rest for validation (default: 0.7)
  --search        Parameter search mode: grid or random (default: grid)
  --n-random      Number of random samples for random search (default: 30)
  --seed          Random seed for reproducibility (default: 42)
  --no-plots      Skip matplotlib figure generation
```

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| **1 Data Loading** | Auto-discover all CSV files; infer schema; handle malformed rows |
| **2 Feature Engineering** | Build rolling features per product; cross-product relative-value |
| **3 EDA Charts** | Mid-price series, return distribution, ACF, intraday patterns |
| **4 Backtesting** | Mean-reversion + Momentum on train/validation split |
| **5 Parameter Search** | Grid/random search; rank by validation Sharpe |
| **6 Report** | Markdown report at `outputs/report.md` |

---

## Generated Outputs

After running, `outputs/` will contain:

```
outputs/
├── report.md                        # Full markdown research report
├── param_search_mean_reversion.csv  # Parameter search results
├── param_search_momentum.csv
└── figures/
    ├── EMERALDS_midprice.png
    ├── EMERALDS_returns_dist.png
    ├── EMERALDS_acf.png
    ├── EMERALDS_intraday.png
    ├── EMERALDS_mean_reversion_pnl.png
    ├── EMERALDS_momentum_pnl.png
    └── ... (same for TOMATOES)
```

---

## CSV Data Format

The pipeline auto-detects two schema types:

### Prices files (`prices_*.csv`)
Semicolon-delimited LOB snapshots:
```
day;timestamp;product;bid_price_1;bid_volume_1;...;ask_price_1;ask_volume_1;...;mid_price;profit_and_loss
```

### Trades files (`trades_*.csv`)
Semicolon-delimited executed trades:
```
timestamp;buyer;seller;symbol;currency;price;quantity
```

### Adapting to new CSV formats

The loader uses heuristic column matching (case-insensitive).  If your CSV uses different column names:
- **Price**: any of `mid_price`, `price`, `close`, `last`, `bid_price_1`
- **Volume**: any of `quantity`, `volume`, `size`, `vol`
- **Timestamp**: any of `timestamp`, `time`, `ts`, `date`, `datetime`
- **Product**: any of `product`, `symbol`, `instrument`, `ticker`

If a required column is absent, the loader falls back gracefully and logs a warning.

---

## Assumptions

1. **Fill price**: Strategies fill at the `mid_price` column (or estimated midprice).
   Real execution would fill at ask (for buys) or bid (for sells), costing half the spread.
2. **Transaction cost**: 0.05% of trade value per fill.
3. **Slippage**: 0.02% adverse price impact per fill.
4. **Position limit**: 10 units per product (configurable in `ExecutionParams`).
5. **Train/validation split**: 70%/30% time-ordered (no data leakage).
6. **Missing LOB levels**: When `bid_price_3` etc. are empty, they are treated as NaN and excluded.

---

## Strategies

### Mean-Reversion
Fades z-score extremes.  Suitable for the tight-spread, low-volatility EMERALDS product.

### Momentum
Follows dual-window momentum signal with volatility scaling.  Suitable for the higher-volatility TOMATOES product.

See `outputs/report.md` for detailed strategy descriptions, metrics, and parameter search results.

---

## Known Limitations

- Bar-based backtester assumes fill at midprice — real fills include spread cost.
- Only 2 days of data; parameter estimates are noisy.
- No portfolio-level PnL aggregation across products.
- Cross-product features are engineered but not used in the single-product backtester.

---

## Module Reference

| Module | Key exports |
|--------|-------------|
| `data_loader` | `discover_and_load()`, `infer_schema()`, `summary_stats()`, `per_product_stats()` |
| `features` | `build_features()`, `build_cross_product_features()` |
| `strategies` | `MeanReversionStrategy`, `MomentumStrategy`, `get_strategy()` |
| `backtest` | `run_backtest()`, `train_val_split()`, `parameter_search()`, `ExecutionParams` |
| `run_research` | `run_pipeline()` (CLI wrapper) |
