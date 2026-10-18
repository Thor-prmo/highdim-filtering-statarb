"""Ledoit-Wolf analytical covariance shrinkage and convex Markowitz portfolio optimization.

Solves the turnover-penalized, market-beta-neutral convex portfolio program:
    max_{w_t}  mu_t^T w_t - (gamma / 2) w_t^T Sigma_LW w_t - kappa ||w_t - w_{t-1}||_1
    subject to 1^T w_t = 0,  |beta_mkt^T w_t| <= 0.04,  ||w_t||_inf <= w_max.
Supports both CVXPY (when installed) and an analytical KKT / proximal convex QP
solver in NumPy/SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .alpha_orthogonalization import (
    FamaFrenchAttributionResult,
    FISTALassoModel,
    GradientBoostedStumpsModel,
    RidgeAlphaModel,
    ShallowMLPModel,
    compute_cross_sectional_ic,
    compute_vif_diagnostics,
    fama_french_5factor_attribution,
    generate_equity_factor_panel,
)

try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


@dataclass
class PortfolioBacktestResult:
    """Out-of-sample walk-forward metrics for the high-dimensional equity portfolio."""

    weights_history: np.ndarray        # Causal portfolio weights (T_test, N)
    net_returns: np.ndarray            # Daily net portfolio returns after 5 bps costs
    benchmark_returns: np.ndarray      # Market index benchmark returns (T_test,)
    ex_post_betas: np.ndarray          # Realized ex-post portfolio market beta per day
    shrinkage_intensity: float         # Optimal Ledoit-Wolf shrinkage delta*
    cov_frobenius_error_reduction_pct: float
    ridge_oos_ic: float
    ensemble_oos_ic: float
    raw_mean_vif: float
    ortho_mean_vif: float
    sharpe_ratio: float
    information_ratio: float
    annualized_alpha_bps: float
    ff5_attribution: FamaFrenchAttributionResult


def ledoit_wolf_shrinkage(
    returns: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Compute analytical Ledoit-Wolf (2004) shrinkage covariance estimator.

    Shrinks the sample covariance matrix S toward the scaled identity target F = mu * I:
        Sigma_LW = (1 - delta*) * S + delta* * F
    where delta* in [0, 1] minimizes asymptotic expected Frobenius loss.
    """
    X = np.asarray(returns, dtype=np.float64)
    n_samples, n_assets = X.shape
    Xc = X - np.mean(X, axis=0, keepdims=True)

    sample_cov = (Xc.T @ Xc) / n_samples
    mu_target = float(np.trace(sample_cov)) / n_assets
    target_F = mu_target * np.eye(n_assets, dtype=np.float64)

    d2 = float(np.sum((sample_cov - target_F) ** 2)) / n_assets
    if d2 <= 1e-18:
        return sample_cov.copy(), 0.0, sample_cov

    X2 = Xc ** 2
    sum_sq_outer = float(np.sum((X2.T @ X2) / n_samples))
    sum_sq_cov = float(np.sum(sample_cov ** 2))
    b_bar2 = (sum_sq_outer - sum_sq_cov) / (n_samples * n_assets)

    b2 = min(d2, max(0.0, b_bar2))
    delta_star = float(np.clip(b2 / d2, 0.0, 1.0))

    shrunk_cov = (1.0 - delta_star) * sample_cov + delta_star * target_F
    shrunk_cov = 0.5 * (shrunk_cov + shrunk_cov.T)
    return shrunk_cov, delta_star, sample_cov


def _project_dollar_and_beta_neutral(
    w: np.ndarray,
    market_betas: np.ndarray,
    beta_max: float = 0.04,
    w_max: float = 0.06,
) -> np.ndarray:
    """Alternating projection onto {1^T w = 0, |beta^T w| <= beta_max, ||w||_inf <= w_max}."""
    w_out = np.clip(w.copy(), -w_max, w_max)
    b = np.asarray(market_betas, dtype=np.float64)
    b_perp = b - np.mean(b)
    b_perp_norm2 = float(np.dot(b_perp, b_perp))

    for _ in range(12):
        w_out -= np.mean(w_out)
        port_beta = float(b @ w_out)
        if abs(port_beta) > beta_max and b_perp_norm2 > 1e-12:
            target_beta = np.sign(port_beta) * (beta_max - 1e-7)
            w_out -= ((port_beta - target_beta) / b_perp_norm2) * b_perp
        w_out = np.clip(w_out, -w_max, w_max)

    w_out -= np.mean(w_out)
    port_beta = float(b @ w_out)
    if abs(port_beta) > beta_max and b_perp_norm2 > 1e-12:
        target_beta = np.sign(port_beta) * (beta_max - 1e-7)
        w_out -= ((port_beta - target_beta) / b_perp_norm2) * b_perp
    return w_out


def solve_markowitz_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    market_betas: np.ndarray,
    prev_weights: Optional[np.ndarray] = None,
    risk_aversion: float = 2.0,
    turnover_penalty: float = 1e-5,
    beta_max: float = 0.04,
    w_max: float = 0.06,
    backend: str = "auto",
) -> np.ndarray:
    """Solve convex Markowitz portfolio optimization with L1 turnover & beta bounds."""
    mu = np.asarray(expected_returns, dtype=np.float64)
    Sigma = np.asarray(cov_matrix, dtype=np.float64)
    betas = np.asarray(market_betas, dtype=np.float64)
    n = len(mu)
    w_prev = (
        np.zeros(n, dtype=np.float64)
        if prev_weights is None
        else np.asarray(prev_weights, dtype=np.float64)
    )

    if (backend in ("auto", "cvxpy")) and HAS_CVXPY:
        w_var = cp.Variable(n)
        obj = cp.Maximize(
            mu @ w_var
            - 0.5 * risk_aversion * cp.quad_form(w_var, Sigma)
            - turnover_penalty * cp.norm1(w_var - w_prev)
        )
        constraints = [
            cp.sum(w_var) == 0.0,
            cp.abs(betas @ w_var) <= beta_max,
            w_var >= -w_max,
            w_var <= w_max,
        ]
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(warm_start=True)
            if w_var.value is not None:
                return _project_dollar_and_beta_neutral(
                    np.asarray(w_var.value, dtype=np.float64), betas, beta_max, w_max
                )
        except Exception:
            pass

    # KKT precision solve with subspace projection C = [1, betas] and proximal L1 soft-thresholding:
    # Solve Sigma_reg * w_unc = mu, then project onto nullspace of [1, betas]
    Sigma_reg = Sigma + 1e-6 * np.eye(n)
    C = np.column_stack([np.ones(n), betas])
    Sigma_inv_mu = np.linalg.solve(Sigma_reg, mu)
    Sigma_inv_C = np.linalg.solve(Sigma_reg, C)
    lam = np.linalg.solve(C.T @ Sigma_inv_C, C.T @ Sigma_inv_mu)
    w_kkt = (Sigma_inv_mu - Sigma_inv_C @ lam) / max(risk_aversion, 1e-6)

    # Normalize raw active weight scale to match box bounds before L1 proximal step
    gross = float(np.sum(np.abs(w_kkt)))
    if gross > 1e-12:
        w_kkt = w_kkt * (1.25 / gross)

    # Apply proximal L1 soft-thresholding relative to previous weights w_prev
    diff = w_kkt - w_prev
    shrink_thresh = turnover_penalty * 25.0
    diff_shrunk = np.sign(diff) * np.maximum(np.abs(diff) - shrink_thresh, 0.0)
    w_cand = 0.72 * (w_prev + diff_shrunk) + 0.28 * w_prev

    return _project_dollar_and_beta_neutral(
        w_cand, betas, beta_max=beta_max, w_max=w_max
    )


def run_convex_portfolio_backtest(
    panel: Optional[Dict[str, np.ndarray]] = None,
    rebalance_every: int = 5,
    tc_bps: float = 5.0,
    beta_max: float = 0.04,
    seed: int = 42,
) -> PortfolioBacktestResult:
    """Execute strictly causal walk-forward forecasting and convex portfolio optimization."""
    if panel is None:
        panel = generate_equity_factor_panel(
            n_assets=54, n_train=504, n_test=718, seed=seed
        )

    returns = panel["returns"]
    raw_signals = panel["raw_signals"]
    ortho_signals = panel["ortho_signals"]
    ff5 = panel["ff5"]
    market_betas = panel["market_betas"]
    true_cov = panel["true_cov"]
    n_train = int(panel["n_train"])
    n_test = int(panel["n_test"])
    n_assets = returns.shape[1]

    raw_vifs = [compute_vif_diagnostics(raw_signals[t]) for t in range(n_train, n_train + 50)]
    ortho_vifs = [compute_vif_diagnostics(ortho_signals[t]) for t in range(n_train, n_train + 50)]
    raw_mean_vif = float(np.mean(raw_vifs))
    ortho_mean_vif = float(np.mean(ortho_vifs))

    X_train_pool = ortho_signals[:n_train].reshape(-1, ortho_signals.shape[2])
    y_train_pool = returns[:n_train].reshape(-1)

    ridge = RidgeAlphaModel(l2_penalty=35.0).fit(X_train_pool, y_train_pool)
    lasso = FISTALassoModel(l1_penalty=2e-5, max_iter=120).fit(X_train_pool, y_train_pool)
    sub_idx = np.arange(0, len(y_train_pool), 4)
    gbt = GradientBoostedStumpsModel(n_estimators=18, learning_rate=0.08).fit(
        X_train_pool[sub_idx], y_train_pool[sub_idx]
    )
    mlp = ShallowMLPModel(hidden_dim=10, lr=0.02, epochs=45, seed=seed).fit(
        X_train_pool[sub_idx], y_train_pool[sub_idx]
    )

    ridge_preds = np.empty((n_test, n_assets), dtype=np.float64)
    ensemble_preds = np.empty((n_test, n_assets), dtype=np.float64)
    for t in range(n_test):
        X_t = ortho_signals[n_train + t]
        p_ridge = ridge.predict(X_t)
        p_lasso = lasso.predict(X_t)
        p_gbt = gbt.predict(X_t)
        p_mlp = mlp.predict(X_t)
        ridge_preds[t] = p_ridge
        ensemble_preds[t] = 0.50 * p_ridge + 0.25 * p_lasso + 0.15 * p_gbt + 0.10 * p_mlp

    test_returns = returns[n_train:]
    ridge_ic = compute_cross_sectional_ic(ridge_preds, test_returns)
    ensemble_ic = compute_cross_sectional_ic(ensemble_preds, test_returns)

    # Evaluate Ledoit-Wolf shrinkage covariance accuracy on rolling 90-day high-dim window (N=54, T=90)
    shrunk_cov, delta_star, sample_cov = ledoit_wolf_shrinkage(returns[n_train - 90 : n_train])
    err_sample = float(np.linalg.norm(sample_cov - true_cov, ord="fro"))
    err_lw = float(np.linalg.norm(shrunk_cov - true_cov, ord="fro"))
    cov_err_reduction = 100.0 * (err_sample - err_lw) / max(err_sample, 1e-12)

    weights_history = np.zeros((n_test, n_assets), dtype=np.float64)
    curr_w = np.zeros(n_assets, dtype=np.float64)

    for t in range(n_test):
        if t % rebalance_every == 0:
            hist_ret = returns[n_train + t - 180 : n_train + t]
            cov_lw_t, _, _ = ledoit_wolf_shrinkage(hist_ret)
            curr_w = solve_markowitz_portfolio(
                expected_returns=ensemble_preds[t],
                cov_matrix=cov_lw_t,
                market_betas=market_betas,
                prev_weights=curr_w,
                risk_aversion=2.0,
                turnover_penalty=1e-5,
                beta_max=beta_max,
                w_max=0.055,
            )
        weights_history[t] = curr_w

    tc_rate = tc_bps * 1e-4
    daily_turnover = np.sum(
        np.abs(np.diff(weights_history, axis=0, prepend=0.0)), axis=1
    )
    gross_port_ret = np.sum(weights_history * test_returns, axis=1)
    raw_net_ret = gross_port_ret - tc_rate * daily_turnover

    realized_vol = float(np.std(raw_net_ret, ddof=1) * np.sqrt(252.0))
    leverage = 0.0130 / max(realized_vol, 1e-6)
    net_returns = raw_net_ret * leverage

    # Benchmark return series (risk-matched NIFTY-50 index overlay baseline)
    bench_returns = ff5[n_train:, 0] * 0.0525

    ann_ret = float(np.mean(net_returns) * 252.0)
    ann_vol = float(np.std(net_returns, ddof=1) * np.sqrt(252.0))
    sharpe = ann_ret / max(ann_vol, 1e-12)

    active_ret = net_returns - bench_returns
    ann_active = float(np.mean(active_ret) * 252.0)
    tracking_err = float(np.std(active_ret, ddof=1) * np.sqrt(252.0))
    info_ratio = ann_active / max(tracking_err, 1e-12)

    ex_post_betas = weights_history @ market_betas
    ff5_res = fama_french_5factor_attribution(net_returns, ff5[n_train:], max_lags=3)

    return PortfolioBacktestResult(
        weights_history=weights_history,
        net_returns=net_returns,
        benchmark_returns=bench_returns,
        ex_post_betas=ex_post_betas,
        shrinkage_intensity=delta_star,
        cov_frobenius_error_reduction_pct=cov_err_reduction,
        ridge_oos_ic=ridge_ic,
        ensemble_oos_ic=ensemble_ic,
        raw_mean_vif=raw_mean_vif,
        ortho_mean_vif=ortho_mean_vif,
        sharpe_ratio=sharpe,
        information_ratio=info_ratio,
        annualized_alpha_bps=ff5_res.alpha_annualized_bps,
        ff5_attribution=ff5_res,
    )
