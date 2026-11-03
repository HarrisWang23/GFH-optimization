"""Backtest performance metrics, identical across every notebook that redefined them."""
from __future__ import annotations

import pandas as pd


def turnover(prev_w: pd.Series, curr_w: pd.Series) -> float:
    """0.5 * L1 turnover between two consecutive rebalances' weight vectors."""
    idx = prev_w.index.union(curr_w.index)
    pw = prev_w.reindex(idx, fill_value=0.0)
    cw = curr_w.reindex(idx, fill_value=0.0)
    return 0.5 * (cw - pw).abs().sum()


def max_drawdown(cumulative_return: pd.Series) -> float:
    """Max drawdown of a cumulative-return series (e.g. ``(1 + returns).cumprod() - 1``)."""
    series = cumulative_return + 1
    cum_max = series.cummax()
    drawdown = (cum_max - series) / cum_max
    return drawdown.max()


def sharpe_ratio(total_return: float, volatility: float, risk_free: float = 0.03) -> float:
    """Non-annualized Sharpe ratio, matching the ``(total_return - rf) / volatility`` used
    across the notebooks (``rf`` varied per notebook: 0.03 in csi300_advanced, 0.05 in
    csi300_basic — pass explicitly rather than relying on the default)."""
    return (total_return - risk_free) / volatility
