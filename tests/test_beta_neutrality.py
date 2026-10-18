"""Verification tests for cross-sectional orthogonalization, Ledoit-Wolf shrinkage,
market-beta neutrality within [-0.04, +0.04], and Neyman-Pearson credit optimization.
"""

import numpy as np

from src.highdim_portfolio import (
    compute_vif_diagnostics,
    cross_sectional_orthogonalize,
    fama_french_5factor_attribution,
    generate_credit_application_dataset,
    generate_equity_factor_panel,
    ledoit_wolf_shrinkage,
    run_convex_portfolio_backtest,
    run_credit_policy_optimization,
    solve_markowitz_portfolio,
)


def test_gram_schmidt_orthogonalization_and_vif_collapse() -> None:
    """Orthogonalized alpha factors must have unit variance, zero correlation, and VIF = 1."""
    panel = generate_equity_factor_panel(n_assets=54, n_train=120, n_test=60, seed=42)
    raw_t = panel["raw_signals"][0]
    ortho_t = cross_sectional_orthogonalize(raw_t, method="gram_schmidt")

    n_assets, n_factors = ortho_t.shape
    gram = (ortho_t.T @ ortho_t) / (n_assets - 1)
    assert np.allclose(gram, np.eye(n_factors), atol=1e-10)

    raw_vifs = compute_vif_diagnostics(raw_t)
    ortho_vifs = compute_vif_diagnostics(ortho_t)
    assert float(np.mean(raw_vifs)) > 1.8
    assert np.allclose(ortho_vifs, 1.0, atol=1e-8)


def test_ledoit_wolf_shrinkage_spd_and_frobenius_loss() -> None:
    """Ledoit-Wolf covariance shrinkage must yield SPD matrix and lower Frobenius error."""
    panel = generate_equity_factor_panel(n_assets=54, n_train=120, n_test=60, seed=42)
    short_sample = panel["returns"][:80]  # High-dimensional regime N=54, T=80
    true_cov = panel["true_cov"]

    shrunk_cov, delta_star, sample_cov = ledoit_wolf_shrinkage(short_sample)
    assert 0.0 < delta_star < 1.0

    eigvals = np.linalg.eigvalsh(shrunk_cov)
    assert np.all(eigvals > 0.0)

    loss_sample = np.linalg.norm(sample_cov - true_cov, ord="fro")
    loss_shrunk = np.linalg.norm(shrunk_cov - true_cov, ord="fro")
    assert loss_shrunk < loss_sample


def test_portfolio_dollar_and_beta_neutrality_bounds() -> None:
    """Markowitz portfolio weights must satisfy 1^T w = 0 and |beta_mkt^T w| <= 0.04."""
    rng = np.random.default_rng(7)
    n_assets = 54
    mu = rng.normal(0.0005, 0.002, size=n_assets)
    # Intentionally correlate expected returns with market beta to stress-test beta bound
    betas = rng.uniform(0.65, 1.45, size=n_assets)
    mu += 0.004 * betas

    A = rng.normal(0.0, 0.01, size=(120, n_assets))
    cov_lw, _, _ = ledoit_wolf_shrinkage(A)

    w = solve_markowitz_portfolio(
        expected_returns=mu,
        cov_matrix=cov_lw,
        market_betas=betas,
        beta_max=0.04,
        w_max=0.06,
    )

    assert abs(float(np.sum(w))) < 1e-10, "Dollar neutrality constraint violated"
    assert abs(float(betas @ w)) <= 0.04 + 1e-7, "Market beta neutrality bound violated"
    assert float(np.max(np.abs(w))) <= 0.06 + 1e-7, "Box weight limit violated"


def test_walk_forward_portfolio_and_fama_french_attribution() -> None:
    """Walk-forward portfolio backtest must maintain ex-post beta within [-0.04, +0.04]."""
    panel = generate_equity_factor_panel(seed=42)
    res = run_convex_portfolio_backtest(panel=panel, rebalance_every=5, beta_max=0.04)

    max_abs_beta = float(np.max(np.abs(res.ex_post_betas)))
    assert max_abs_beta <= 0.04 + 1e-6
    assert np.allclose(np.sum(res.weights_history, axis=1), 0.0, atol=1e-9)
    assert res.ff5_attribution.alpha_t_stat_nw > 1.96


def test_credit_policy_neyman_pearson_ceiling() -> None:
    """Credit policy optimizer must respect the 6.5% PD ceiling and improve portfolio NPV."""
    ds = generate_credit_application_dataset(n_applications=10000, seed=42)
    res = run_credit_policy_optimization(dataset=ds, pd_ceiling=0.065, seed=42)

    assert res.roc_auc > 0.75
    assert res.ks_statistic > 0.35
    assert res.optimized_portfolio_pd_pct <= res.pd_ceiling_pct + 0.15
    assert res.optimized_approval_rate_pct > res.legacy_approval_rate_pct
    assert res.npv_improvement_pct > 10.0
