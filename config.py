"""
config.py – Central configuration for the IMC-style trading research pipeline.

All tuneable knobs live here so that the rest of the code stays free of
magic numbers.  Import this module everywhere instead of hard-coding values.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR: str = "."          # root dir; prices_*.csv / trades_*.csv live here
DEFAULT_OUTPUT_DIR: str = "outputs"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# Column names expected / inferred from the IMC-format CSVs
PRICES_COLS = {
    "day": "day",
    "timestamp": "timestamp",
    "product": "product",
    "bid_price_1": "bid_price_1",
    "bid_volume_1": "bid_volume_1",
    "ask_price_1": "ask_price_1",
    "ask_volume_1": "ask_volume_1",
    "mid_price": "mid_price",
    "pnl": "profit_and_loss",
}

TRADES_COLS = {
    "timestamp": "timestamp",
    "buyer": "buyer",
    "seller": "seller",
    "symbol": "symbol",
    "price": "price",
    "quantity": "quantity",
}

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
ROLLING_WINDOWS = [5, 10, 20, 50]   # bar counts for rolling calculations
MOMENTUM_WINDOWS = [5, 20]           # windows used for momentum features
ZSCORE_WINDOW = 20                   # look-back for mean-reversion z-score
VOL_WINDOW = 20                      # rolling volatility window
AUTOCORR_LAGS = list(range(1, 11))  # lags for autocorrelation analysis

# ---------------------------------------------------------------------------
# Strategy defaults
# ---------------------------------------------------------------------------
POSITION_LIMIT: int = 20            # max units long or short per product
MAX_GROSS_EXPOSURE: int = 40        # abs(long) + abs(short) across all products
TRANSACTION_COST: float = 2.0       # cost per unit traded (price ticks)
SLIPPAGE: float = 1.0               # additional slippage per unit (price ticks)
STOP_LOSS_THRESHOLD: float = 50.0   # unrealised loss per product triggers flat
INVENTORY_PENALTY: float = 0.1      # penalty term per unit of inventory

# Mean-reversion strategy defaults
MR_ENTRY_ZSCORE: float = 1.5        # enter when |z| > this
MR_EXIT_ZSCORE: float = 0.5         # exit when |z| < this
MR_TRADE_SIZE: int = 5              # units per signal

# Momentum strategy defaults
MOM_SHORT_WINDOW: int = 5
MOM_LONG_WINDOW: int = 20
MOM_TRADE_SIZE: int = 5

# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = 0.7            # fraction of data used for training
RISK_FREE_RATE: float = 0.0         # per-bar risk-free rate for Sharpe

# ---------------------------------------------------------------------------
# Parameter search
# ---------------------------------------------------------------------------
PARAM_SEARCH_RANDOM_SEED: int = 42
PARAM_SEARCH_N_RANDOM: int = 30     # number of random parameter sets to try
