"""Signal construction: the rolling lookback window, momentum mu, and covariance Sigma
used to feed src/optimizer.py at each rebalance date. Ported from the identical block
repeated in notebooks/csi300_advanced.ipynb, notebooks/csi300_basic.ipynb and
src/_legacy_backtest_tracking_error.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def lookback_window(all_days: pd.DatetimeIndex, target_day, lookback_days: int) -> pd.DatetimeIndex:
    """Trailing ``lookback_days`` trading days ending on (and including) ``target_day``.

    Count-based, not calendar-date-based — this matches csi300_advanced.ipynb /
    src/_legacy_backtest_tracking_error.py. (src/_legacy_backtest_draft.py used a calendar-date
    subtraction instead, ``all_days > target_day - lookback_days``, which is not equivalent
    once weekends/holidays are involved; that variant was never the one carried through to a
    completed backtest, so it is not reproduced here.)
    """
    return all_days[all_days <= target_day][-lookback_days:]


def momentum_mu(price_wide: pd.DataFrame, window_days: pd.DatetimeIndex, tickers, clip: tuple[float, float] | None = None) -> pd.Series:
    """1-day momentum signal: the most recent realized daily return as of ``window_days[-1]``.

    Uses ``window_days[-2]`` / ``window_days[-3]`` (i.e. the day *before* the last day in the
    window, over the day before that) rather than the window's own last two days — the original
    notebooks deliberately stop one day short of the rebalance date itself to avoid look-ahead.
    """
    prev1_day = window_days[-2]
    prev2_day = window_days[-3]
    price_t_1 = price_wide.loc[prev1_day, tickers]
    price_t_2 = price_wide.loc[prev2_day, tickers]
    mu = (price_t_1 / price_t_2 - 1.0)
    if clip is not None:
        mu = mu.clip(lower=clip[0], upper=clip[1])
    return mu


def covariance(returns_df: pd.DataFrame) -> np.ndarray:
    return returns_df.cov().values


def window_returns(price_wide: pd.DataFrame, window_days: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    return price_wide.loc[window_days, tickers].pct_change().dropna()
