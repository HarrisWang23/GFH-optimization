"""Walk-forward backtest engine for the age-split index-enhancement strategy.

This is a faithful port of the completed, results-bearing loop in
``notebooks/csi300_advanced.ipynb`` (cell 13) — the only one of the several backtest drafts
(``src/_legacy_backtest_draft.py``, ``src/_legacy_backtest_tracking_error.py``) that was actually
carried through to a full run with reported Sharpe/drawdown numbers. It is *not* a rewrite: every
branch below is the same branch that ran, including one non-obvious quirk documented at the call
site — see the NOTE below.

Strategy per rebalance day:
  - Split today's holdings into newly-entered names (``age == 1``) and the rest.
  - Optimize only the age==1 sleeve (mean-variance, risk-limited) and rescale it to
    ``age1_sleeve_weight`` (20% originally) of the book.
  - The remaining ``1 - age1_sleeve_weight`` (80%) tracks the benchmark/actual position weight
    (``df_long``) as-is, for *every* held name (age==1 names get both a sliver of the optimized
    sleeve and their share of the 80% benchmark sleeve).
  - If there are no age==1 names that day, the whole book just follows the benchmark weight.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import cvxpy as cp

from . import signals
from .optimizer import solve_mean_variance
from .metrics import turnover as turnover_fn


@dataclass
class BacktestResult:
    returns: pd.Series
    weights_history: dict[pd.Timestamp, pd.Series] = field(default_factory=dict)
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    rebalance_days: list = field(default_factory=list)


def run_age_split_backtest(
    df_price: pd.DataFrame,
    df_holdings: pd.DataFrame,
    df_long: pd.DataFrame,
    lookback_days: int = 120,
    holding_days: int = 1,
    age1_sleeve_weight: float = 0.20,
    risk_limit: float = 0.005,
    mu_clip: tuple[float, float] | None = None,
    exclude_dates=(pd.Timestamp("2025-01-01"),),
    solver=cp.MOSEK,
) -> BacktestResult:
    """``df_price``: long ``[dt, ticker, close]``. ``df_holdings``: long ``[dt, ticker, age]``.
    ``df_long``: long ``[dt, ticker, weight]`` (the benchmark/actual-position weight each name
    should track for its 80% sleeve).
    """
    df_price_wide = df_price.pivot(index="dt", columns="ticker", values="close")
    all_days = df_price_wide.index.sort_values()
    for d in exclude_dates:
        all_days = all_days[all_days != pd.Timestamp(d)]

    df_long = df_long.copy()
    df_long["dt"] = pd.to_datetime(df_long["dt"]).dt.normalize()

    target_days = all_days[lookback_days::holding_days]

    rebalance_days = []
    weights_history: dict[pd.Timestamp, pd.Series] = {}
    turnover_series = {}
    portfolio_returns = []
    prev_weights = None

    for target_day in target_days:
        rebalance_days.append(target_day)

        tickers_today = df_holdings.loc[df_holdings["dt"] == target_day, "ticker"].unique().tolist()
        tickers_age1 = df_holdings.loc[
            (df_holdings["dt"] == target_day) & (df_holdings["age"] == 1), "ticker"
        ].unique().tolist()

        window_days = signals.lookback_window(all_days, target_day, lookback_days)
        returns_df = signals.window_returns(df_price_wide, window_days, tickers_today)
        mu = signals.momentum_mu(df_price_wide, window_days, tickers_today, clip=mu_clip)

        target_day_norm = pd.Timestamp(target_day).normalize()
        base_today = (
            df_long.loc[df_long["dt"] == target_day_norm, ["ticker", "weight"]]
            .set_index("ticker")["weight"]
            .reindex(tickers_today, fill_value=0.0)
        )

        if len(tickers_age1) > 0:
            sigma_age1 = signals.covariance(returns_df[tickers_age1])
            mu_age1 = mu.loc[tickers_age1]

            # NOTE (kernel quirk, preserved intentionally): the original notebook called
            # optimization_from_mu_sigma(mu_age1, sigma_age1, prev_weights, base_today.loc[...])
            # with the benchmark weight passed *positionally* into the `turnover_cap` slot, not
            # the `Base_today` slot the code otherwise clearly intends. Because the turnover-cap
            # constraint block was itself commented out, the net effect actually run was: no
            # base-weight deviation band was ever applied to the age==1 sleeve, and w_prev was
            # passed but never used either. That is what produced the notebook's reported
            # results, so it's reproduced here as-is rather than "fixed" to what looks intended.
            weights_age1 = solve_mean_variance(
                mu_age1,
                sigma_age1,
                risk_limit=risk_limit,
                tickers=tickers_age1,
                solver=solver,
            )
            weights_age1 = weights_age1 / weights_age1.sum() * age1_sleeve_weight

            weights_all = pd.Series(0.0, index=tickers_today)
            weights_all.loc[tickers_age1] = weights_age1
            weights = weights_all + (1 - age1_sleeve_weight) * base_today.reindex(
                tickers_today, fill_value=0.0
            )
        else:
            weights = base_today.reindex(tickers_today, fill_value=0.0)

        next_days = all_days[all_days >= target_day][: holding_days + 1]
        if len(next_days) < holding_days:
            continue

        future_returns = df_price_wide.loc[next_days, tickers_today].pct_change().dropna()
        portfolio_ret = future_returns @ weights
        portfolio_returns.append(portfolio_ret)

        if prev_weights is not None:
            turnover_series[target_day] = turnover_fn(prev_weights, weights)

        weights_history[target_day] = weights
        prev_weights = weights

    return BacktestResult(
        returns=pd.concat(portfolio_returns) if portfolio_returns else pd.Series(dtype=float),
        weights_history=weights_history,
        turnover=pd.Series(turnover_series),
        rebalance_days=rebalance_days,
    )
