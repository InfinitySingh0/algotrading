# IMC Trading Research Pipeline

A complete Python-based research and backtesting pipeline for IMC-style trading challenge data.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (CSV files in current directory)
python run_research.py

# 3. Custom paths
python run_research.py --data-dir data/ --output-dir outputs/

# 4. Skip plot generation (faster, headless environments)
python run_research.py --no-plots
```

Outputs are written to `outputs/` (or `--output-dir`):
- `outputs/eda/`              – per-product EDA plots (PNG)
- `outputs/strategy_plots/`   – cumulative PnL charts per strategy + product
- `outputs/report.md`         – full research report
- `report.md`                 – copy in repo root

---

## Project Structure

| File | Purpose |
|------|---------|
| `config.py`        | All configurable parameters (position limits, costs, windows) |
| `data_loader.py`   | CSV discovery, loading, schema inference, quality report |
| `features.py`      | Leakage-safe feature engineering (returns, momentum, z-score, vol, OFI) |
| `metrics.py`       | Performance metrics (Sharpe, Sortino, max drawdown, hit rate) |
| `strategies.py`    | MeanReversionStrategy + MomentumStrategy with risk controls |
| `backtest.py`      | Bar-based backtester, train/val split, grid/random param search |
| `run_research.py`  | Orchestration script – runs the full pipeline end-to-end |
| `requirements.txt` | Python package dependencies |

---

## CLI Options

```
python run_research.py [OPTIONS]

Options:
  --data-dir   DIR    Directory containing CSV files (default: current dir)
  --output-dir DIR    Where to write plots and reports (default: outputs/)
  --no-plots          Skip matplotlib chart generation
  --train-ratio R     Fraction of data used for training (default: 0.7)
  --top-n       N     Number of top param sets in report (default: 5)
```

---

## Supported CSV Schemas

The pipeline auto-detects two formats used in the IMC Trading Challenge:

### prices\_\*.csv (order-book snapshots)
Semicolon-separated. Expected columns:

| Column | Description |
|--------|-------------|
| `day` | Trading day (integer, e.g. -1, -2) |
| `timestamp` | Tick timestamp within day |
| `product` | Product/symbol name |
| `bid_price_1` / `bid_volume_1` | Best bid |
| `ask_price_1` / `ask_volume_1` | Best ask |
| `bid_price_2/3`, `ask_price_2/3` | Deeper levels (optional) |
| `mid_price` | Pre-computed mid-price |
| `profit_and_loss` | Cumulative PnL (informational) |

### trades\_\*.csv (public trade tape)
Semicolon-separated. Expected columns:

| Column | Description |
|--------|-------------|
| `timestamp` | Trade timestamp |
| `buyer` / `seller` | Counterparty names (may be empty) |
| `symbol` | Product name |
| `currency` | Venue currency |
| `price` | Trade price |
| `quantity` | Trade size |

---

## Adapting to New CSV Schemas

1. **Different separator**: Change `sep=";"` in `data_loader._read_csv_robust`.
2. **Different column names**: Update `PRICES_COLS` / `TRADES_COLS` in `config.py`, or extend the `_infer_*` helpers in `data_loader.py`.
3. **Different file naming**: Change the glob patterns in `data_loader.discover_files`.
4. **Multiple products per row**: The pipeline assumes one product per row; reshape with `pd.melt` before passing to `data_loader.load_prices`.

---

## Assumptions

- Data files are semicolon-delimited CSVs.
- `mid_price` is the primary price signal; derived from `(bid_price_1 + ask_price_1) / 2` if absent.
- Each row represents one bar (tick/snapshot) for one product.
- Train/validation split is strictly time-based to avoid look-ahead bias.
- Transaction costs are modelled as flat per-unit fees (configurable in `config.py`).
- Position limits and stop-losses are enforced at every bar.

---

## Strategies

### Mean Reversion
- **Premise**: Mid-price reverts to its rolling mean. Deviations beyond `entry_z` standard deviations are faded.
- **Entry**: `zscore > entry_z` → short; `zscore < -entry_z` → long.
- **Exit**: `|zscore| < exit_z` → flatten.
- **Why it works for IMC**: IMC challenge order books are tightly quoted around fair value; large dislocations are temporary.

### Momentum
- **Premise**: Short-term price trend persists. Positive momentum → long, negative → short.
- **Signal**: `ret_short - ret_long`; trade when |signal| > threshold.
- **Why it works**: Microstructure momentum (short-lived trend-following) can capture brief directional moves between mean-reversion episodes.

---

## Outputs

After running `python run_research.py` you will find:

```
outputs/
├── eda/
│   ├── EMERALDS_eda.png
│   └── TOMATOES_eda.png
├── strategy_plots/
│   ├── mean_reversion_EMERALDS.png
│   ├── mean_reversion_TOMATOES.png
│   ├── momentum_EMERALDS.png
│   └── momentum_TOMATOES.png
└── report.md
report.md   (copy at repo root)
```
