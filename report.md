# IMC Trading Challenge – Research Pipeline Report

> **Note:** This file is a placeholder.  Run `python run_research.py --data-dir .` to regenerate
> it with live results.  The generated report will appear at `outputs/report.md`.

---

## Overview

This report is produced automatically by the research pipeline.  It covers:

1. **Dataset Summary** – schema, row counts, missing values, per-product statistics
2. **Strategy Descriptions** – mean-reversion and momentum with risk controls
3. **Backtest Results** – train vs validation PnL, Sharpe, drawdown, hit rate
4. **Parameter Search** – top configurations ranked by validation Sharpe
5. **Limitations** – known failure modes and modelling gaps

To view the full generated report, run:

```bash
python run_research.py --data-dir .
cat outputs/report.md
```
