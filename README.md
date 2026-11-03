# Multi-Factor Portfolio Weight Optimization

This repository presents the code structure and research workflow of a portfolio optimization project I worked on during my internship at GFH Asset Management. The project builds on the strategy team's multi-factor stock-selection model and focuses on optimizing the portfolio weights assigned to the model's stock-selection results.

## Method Overview

Mean-variance optimization is the main framework. I explored several formulations with different risk and return objectives:

- Incorporating a downside variance penalty into the objective to balance return and downside risk.
- Minimizing portfolio variance subject to a minimum return requirement.
- Maximizing expected return subject to a maximum portfolio variance constraint.
- Testing tracking-error minimization against benchmark indices such as the CSI 500 and CSI 300.

The research involved multiple rounds of backtesting and iteration, including changes to the backtest window, optimization settings, and hyperparameters. The repository focuses on the project's code organization and optimization approach rather than on a single parameter setting or backtest result.

## Project Structure

```
.
├── src/                    # 从 notebooks 中提取并去重的核心逻辑，附单元测试
│   ├── optimizer.py        # 均值-方差优化器 / 类夏普比率优化器
│   ├── signals.py          # 滚动窗口、动量信号与协方差构建
│   ├── metrics.py          # 换手率、最大回撤与夏普比率
│   └── backtest.py         # 逐日调仓回测引擎（新股与存量持仓的权重分配逻辑）
├── notebooks/               # 实习期间的原始研究代码（已清除 notebook 输出）
│   ├── csi300_basic.ipynb          # 沪深300组合优化 - 基础版
│   ├── csi300_advanced.ipynb       # 沪深300组合优化 - 改进版（新股权重分配逻辑的来源）
│   ├── csi300_actual_holdings.ipynb # 实际持仓归因分析
│   ├── csi500_sharpe.ipynb         # 中证500 - 类夏普比率优化
│   └── csi500_variance_limit.ipynb # 中证500 - 方差限制下的收益最大化
├── scripts/                 # 独立运行脚本
│   ├── etl_csv_to_mysql.py  # 将因子数据从 CSV 批量导入 MySQL
│   ├── build_cov_demo.py    # 协方差构建与组合优化的独立演示
│   └── ml_price_forecast.py # Transformer 价格预测实验（探索性项目，未接入上述优化流程）
├── data/csv/                 # Barra 风格因子暴露及美股价格/成交量数据
├── tests/                    # src/ 的单元测试（见下文）
├── licenses/mosek.lic        # 已过期的 MOSEK 许可证文件（见下文；已在 .gitignore 中排除）
└── requirements.txt
```

### Relationship Between `src/` and `notebooks/`

The five files in `notebooks/` are the original research notebooks from my internship. They connected to the company’s internal MySQL database and analyzed actual holdings, documenting the evolution of the research approach, including signal construction, optimization modeling, and risk controls. For data confidentiality, and because the stored results were not the final versions, all notebook outputs (charts, tables, and numerical results) have been cleared; only the code remains.

`src/` is a reusable version reorganized from that research. It consolidates optimizer and backtest logic that had been repeated with small variations across notebooks into parameterized modules with clear responsibilities and unit tests. The notebooks record the original research process, while `src/` presents the core logic in a more reusable form. The code quality is best represented by `src/` and its tests, rather than by any single historical backtest result.

## Environment and Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Unit tests for `src/`** (run independently without a database):

```bash
pytest tests/ -v
```

The 23 tests cover optimizer constraints (weights sum to 1, non-negativity, variance limits, per-stock caps, and benchmark deviation bands), signal construction, performance metrics, and end-to-end invariants of the backtest engine on synthetic data (for example, daily weights sum to 1 and turnover stays within `[0, 1]`). The core backtest logic in `src/backtest.py` was also checked against the original notebook implementation through a day-by-day numerical comparison on synthetic data (see the module's top-level comments).

**About the full backtest workflow:** Running the notebooks requires access to the company's internal MySQL database (holdings, market data, and index constituents) and a valid MOSEK academic license (`licenses/mosek.lic` expired on 2026-06-03). Neither is currently available, so the notebooks are retained as a historical record of the methodology without outputs. The runnable and verifiable parts are the unit tests in `src/`.

## Data Notes

The factor files in `data/csv/` (`*_Std.csv`, standardized Barra-style factor exposures) and the price/volume files (such as `close.csv` and `volume.csv`) are a separate dataset covering U.S. tickers, used by the ETL workflow in `scripts/etl_csv_to_mysql.py`. They are distinct from the A-share market, holdings, and index-constituent data used by the CSI 300/500 strategies in `notebooks/`.
