"""Mean-variance portfolio optimizers, shared by the backtest engine and every notebook.

Consolidated from the near-duplicate ``optimization_from_mu_sigma`` definitions that used
to live independently in ``notebooks/csi300_basic.ipynb``, ``notebooks/csi300_advanced.ipynb``,
``notebooks/csi500_variance_limit.ipynb`` and ``src/_legacy_backtest_*.py``. Each notebook solved
the same QP with a different subset of constraints turned on; every optional constraint below
defaults to *off* so that passing only ``mu``/``Sigma``/``risk_limit`` reproduces the plainest
variant, and each notebook's exact original behavior is recovered by supplying the matching
optional arguments (see call sites in ``src/backtest.py``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp


def _as_index(mu, tickers):
    if tickers is not None:
        return pd.Index(tickers)
    if isinstance(mu, pd.Series):
        return mu.index
    return pd.RangeIndex(len(mu))


def solve_mean_variance(
    mu,
    Sigma,
    risk_limit: float,
    tickers=None,
    upper_cap: float | None = None,
    base_weights: pd.Series | np.ndarray | None = None,
    band: float = 0.03,
    w_prev: pd.Series | np.ndarray | None = None,
    turnover_cap: float | None = None,
    solver=cp.MOSEK,
) -> pd.Series:
    """Maximize expected return under a variance cap (the strategy's core QP).

        maximize    mu @ w
        subject to  sum(w) == 1
                    w >= 0
                    w @ Sigma @ w <= risk_limit
                    w <= upper_cap                      [optional; csi300_basic uses 0.06]
                    base_weights - band <= w <= base_weights + band
                                                         [optional; csi300_advanced uses band=0.03]
                    w_prev - turnover_cap <= w <= w_prev + turnover_cap
                                                         [optional; present but never enabled in
                                                          any original notebook]

    Returns a ``pd.Series`` of weights indexed by ``tickers`` (or ``mu.index`` if ``mu`` is
    already a Series). Returns a Series of NaN, matching the original notebooks' behavior, if
    the solver fails to find a feasible/optimal point.
    """
    index = _as_index(mu, tickers)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)

    w = cp.Variable(n)
    constraints = [
        cp.sum(w) == 1,
        cp.quad_form(w, Sigma) <= risk_limit,
        w >= 0,
    ]

    if upper_cap is not None:
        constraints.append(w <= upper_cap)

    if base_weights is not None:
        base = pd.Series(base_weights).reindex(index, fill_value=0.0).values if isinstance(
            base_weights, pd.Series
        ) else np.asarray(base_weights, dtype=float)
        if not np.allclose(base, 0):
            constraints += [w - base <= band, w - base >= -band]

    if w_prev is not None and turnover_cap is not None:
        prev = pd.Series(w_prev).reindex(index, fill_value=0.0).values if isinstance(
            w_prev, pd.Series
        ) else np.asarray(w_prev, dtype=float)
        constraints += [w - prev <= turnover_cap, prev - w <= turnover_cap]

    prob = cp.Problem(cp.Maximize(mu @ w), constraints)
    prob.solve(solver=solver)

    if w.value is None:
        return pd.Series([np.nan] * n, index=index)
    return pd.Series(w.value, index=index)


def solve_max_sharpe(
    mu,
    Sigma,
    tickers=None,
    risk_free_annual: float = 0.02,
    trading_days: int = 252,
    risk_budget: float = 1.0,
    solver=cp.MOSEK,
) -> pd.Series:
    """Maximize a Sharpe-like ratio (ported from ``notebooks/csi500_sharpe.ipynb``).

        maximize    t
        subject to  (mu - r_f) @ x >= t
                    x @ Sigma @ x <= risk_budget
                    sum(x) == 1
                    x >= 0
    """
    index = _as_index(mu, tickers)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)
    r_f = risk_free_annual / trading_days
    excess_mu = mu - r_f

    x = cp.Variable(n)
    t = cp.Variable()
    constraints = [
        excess_mu @ x >= t,
        cp.quad_form(x, Sigma) <= risk_budget,
        cp.sum(x) == 1,
        x >= 0,
    ]
    prob = cp.Problem(cp.Maximize(t), constraints)
    prob.solve(solver=solver)

    if x.value is None:
        return pd.Series([np.nan] * n, index=index)
    return pd.Series(x.value, index=index)
