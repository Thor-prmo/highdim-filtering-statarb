"""Strict causality and zero-lookahead verification tests for state-space filters.

Verifies that both the online Hamilton (1989) Markov-switching filter and the
linear Gaussian state-space Kalman cointegration filter satisfy strict temporal
causality: perturbing or corrupting future observations Y_{t+1:T} produces zero
change (within machine epsilon 1e-14) in filtered state posteriors at time <= t.
"""

import numpy as np
import pytest

from src.state_space_filtering import (
    GaussianHMM3State,
    HamiltonOnlineFilter,
    KalmanOUStatArb,
    causal_majority_vote_smoother,
    estimate_ou_parameters,
    generate_cointegrated_pair_data,
    generate_multi_asset_regime_data,
    run_kalman_statarb_backtest,
    run_regimeshift_backtest,
)


def test_online_hamilton_strict_causality() -> None:
    """Filtered state probabilities P(S_t | Y_{1:t}) must be invariant to Y_{t+1:T}."""
    data = generate_multi_asset_regime_data(n_train=400, n_test=300, seed=42)
    hmm = GaussianHMM3State(n_states=3, max_iter=60, tol=1e-6, random_state=42)
    hmm.fit(data["returns_train"])

    h_filter = HamiltonOnlineFilter(hmm)
    obs_clean = data["returns_test"].copy()
    post_clean = h_filter.filter_online(obs_clean)

    # rows must sum to 1 and lie in [0, 1]
    assert np.all(post_clean >= 0.0)
    assert np.allclose(np.sum(post_clean, axis=1), 1.0, atol=1e-12)

    for cutoff_t in (25, 110, 220):
        obs_mutated = obs_clean.copy()
        # Inject extreme counterfactual shocks exclusively after cutoff_t
        obs_mutated[cutoff_t + 1 :] = obs_mutated[cutoff_t + 1 :] * (-8.5) + 0.35
        post_mutated = h_filter.filter_online(obs_mutated)

        assert np.allclose(
            post_clean[: cutoff_t + 1],
            post_mutated[: cutoff_t + 1],
            atol=1e-14,
        ), f"Lookahead leakage detected in Hamilton filter at cutoff t={cutoff_t}"


def test_causal_majority_vote_smoother_invariance_and_turnover() -> None:
    """15-day causal smoother must not leak future states and must reduce turnover."""
    data = generate_multi_asset_regime_data(n_train=400, n_test=300, seed=42)
    hmm = GaussianHMM3State(n_states=3, max_iter=60, random_state=42).fit(
        data["returns_train"]
    )
    post_clean = HamiltonOnlineFilter(hmm).filter_online(data["returns_test"])

    states_clean, probs_clean = causal_majority_vote_smoother(post_clean, window=15)

    cutoff_t = 140
    post_corrupted = post_clean.copy()
    post_corrupted[cutoff_t + 1 :] = np.array([0.01, 0.01, 0.98])
    states_corrupted, probs_corrupted = causal_majority_vote_smoother(
        post_corrupted, window=15
    )

    assert np.array_equal(states_clean[: cutoff_t + 1], states_corrupted[: cutoff_t + 1])
    assert np.allclose(
        probs_clean[: cutoff_t + 1], probs_corrupted[: cutoff_t + 1], atol=1e-14
    )

    res = run_regimeshift_backtest(data=data, smoothing_window=15, tc_bps=10.0)
    assert res.annualized_turnover_smoothed < res.annualized_turnover_raw
    assert res.turnover_reduction_pct > 15.0
    assert res.cvar_99_reduction_pct > 10.0


def test_kalman_cointegration_strict_causality_and_spd_covariance() -> None:
    """Kalman state-space estimates beta_{t|t} must be strictly causal and P_{t|t} SPD."""
    pair_data = generate_cointegrated_pair_data(n_train=300, n_test=400, seed=42)
    kf = KalmanOUStatArb(delta=3.5e-4, obs_var=0.065)

    p1_clean = pair_data["p1_test"].copy()
    p2_clean = pair_data["p2_test"].copy()
    res_clean = kf.filter_series(p1_clean, p2_clean)

    # Verify Joseph-form covariance matrices are symmetric and strictly positive-definite
    for t in range(0, len(p1_clean), 20):
        cov_t = res_clean["covs"][t]
        assert np.allclose(cov_t, cov_t.T, atol=1e-14)
        eigvals = np.linalg.eigvalsh(cov_t)
        assert np.all(eigvals > 0.0)

    cutoff_t = 180
    p1_mut = p1_clean.copy()
    p2_mut = p2_clean.copy()
    p1_mut[cutoff_t + 1 :] *= 4.0
    p2_mut[cutoff_t + 1 :] -= 250.0
    res_mut = kf.filter_series(p1_mut, p2_mut)

    assert np.allclose(
        res_clean["betas"][: cutoff_t + 1],
        res_mut["betas"][: cutoff_t + 1],
        atol=1e-14,
    )
    assert np.allclose(
        res_clean["innovations"][: cutoff_t + 1],
        res_mut["innovations"][: cutoff_t + 1],
        atol=1e-14,
    )

    bt = run_kalman_statarb_backtest(seed=42)
    assert bt.tracking_error_reduction_pct > 25.0
    assert 3.0 <= bt.ou_params.half_life <= 25.0
    assert bt.sharpe_ratio > 1.5
