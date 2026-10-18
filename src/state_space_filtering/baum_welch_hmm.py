"""3-State Multivariate Gaussian Hidden Markov Model calibrated via Baum-Welch EM.

Infers latent macro-financial volatility regimes (Bull, Bear, Crisis) across
Equities, Gold, and Treasury Bonds. All E-step recursions operate in log-space
using log-sum-exp stabilization to prevent numerical underflow over long daily
return histories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.special import logsumexp


@dataclass
class HMMParameters:
    """Calibrated parameters for a K-state multivariate Gaussian HMM."""

    pi: np.ndarray          # Initial state distribution (K,)
    transition: np.ndarray  # Row-stochastic transition matrix A (K, K)
    means: np.ndarray       # Regime mean return vectors (K, D)
    covs: np.ndarray        # Regime covariance matrices (K, D, D)
    log_likelihood_history: list


class GaussianHMM3State:
    """Multivariate Gaussian HMM calibrated via Baum-Welch Expectation-Maximization.

    States are canonically ordered post-calibration by ascending equity volatility
    (index 0 of the observation vector):
        State 0 -> Bull (low volatility, positive equity drift)
        State 1 -> Bear (elevated volatility, corrective drift)
        State 2 -> Crisis (tail volatility, flight-to-safety into Gold/Treasuries)
    """

    def __init__(
        self,
        n_states: int = 3,
        max_iter: int = 120,
        tol: float = 1e-6,
        cov_reg: float = 1e-7,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.cov_reg = cov_reg
        self.random_state = random_state
        self.params_: Optional[HMMParameters] = None

    @staticmethod
    def _log_multivariate_normal_density(
        obs: np.ndarray, means: np.ndarray, covs: np.ndarray
    ) -> np.ndarray:
        """Compute log N(y_t | mu_k, Sigma_k) for all t in 0..T-1 and k in 0..K-1."""
        n_samples, n_dim = obs.shape
        n_states = means.shape[0]
        log_prob = np.empty((n_samples, n_states), dtype=np.float64)
        const = -0.5 * n_dim * np.log(2.0 * np.pi)

        for k in range(n_states):
            cov_k = 0.5 * (covs[k] + covs[k].T)
            chol = np.linalg.cholesky(cov_k)
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            diff = obs - means[k]
            sol = np.linalg.solve(chol, diff.T)
            quad_form = np.sum(sol * sol, axis=0)
            log_prob[:, k] = const - 0.5 * log_det - 0.5 * quad_form

        return log_prob

    def _initialize_parameters(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize regimes via rolling equity volatility quantiles."""
        n_samples, n_dim = obs.shape
        K = self.n_states

        # Estimate local volatility proxy using 10-day rolling squared equity returns
        eq_ret = obs[:, 0]
        kernel = np.ones(10) / 10.0
        local_vol = np.sqrt(np.convolve(eq_ret ** 2, kernel, mode="same") + 1e-10)

        q1, q2 = np.quantile(local_vol, [0.55, 0.85])
        labels = np.zeros(n_samples, dtype=int)
        labels[(local_vol >= q1) & (local_vol < q2)] = 1
        labels[local_vol >= q2] = 2

        pi = np.array([0.65, 0.25, 0.10], dtype=np.float64)
        transition = np.array(
            [
                [0.970, 0.025, 0.005],
                [0.040, 0.935, 0.025],
                [0.015, 0.065, 0.920],
            ],
            dtype=np.float64,
        )
        if K != 3:
            pi = np.full(K, 1.0 / K)
            transition = np.full((K, K), 0.05 / max(K - 1, 1))
            np.fill_diagonal(transition, 0.95)

        means = np.zeros((K, n_dim), dtype=np.float64)
        covs = np.zeros((K, n_dim, n_dim), dtype=np.float64)
        global_cov = np.cov(obs, rowvar=False) + self.cov_reg * np.eye(n_dim)

        for k in range(K):
            mask = labels == k
            if np.sum(mask) > n_dim + 2:
                means[k] = np.mean(obs[mask], axis=0)
                covs[k] = np.cov(obs[mask], rowvar=False) + self.cov_reg * np.eye(n_dim)
            else:
                means[k] = np.mean(obs, axis=0)
                covs[k] = (k + 1) * global_cov

        return pi, transition, means, covs

    @staticmethod
    def _forward_log(
        log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Log-domain forward recursion alpha_t(k) = log P(Y_{1:t}, S_t = k)."""
        n_samples, n_states = log_B.shape
        log_alpha = np.empty((n_samples, n_states), dtype=np.float64)
        log_alpha[0] = log_pi + log_B[0]

        for t in range(1, n_samples):
            # log_alpha[t, k] = log_B[t, k] + logsumexp_j(log_alpha[t-1, j] + log_A[j, k])
            log_alpha[t] = log_B[t] + logsumexp(
                log_alpha[t - 1][:, None] + log_A, axis=0
            )

        total_log_lik = float(logsumexp(log_alpha[-1]))
        return log_alpha, total_log_lik

    @staticmethod
    def _backward_log(log_A: np.ndarray, log_B: np.ndarray) -> np.ndarray:
        """Log-domain backward recursion beta_t(j) = log P(Y_{t+1:T} | S_t = j)."""
        n_samples, n_states = log_B.shape
        log_beta = np.zeros((n_samples, n_states), dtype=np.float64)

        for t in range(n_samples - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_A + log_B[t + 1][None, :] + log_beta[t + 1][None, :], axis=1
            )

        return log_beta

    def fit(self, obs_train: np.ndarray) -> "GaussianHMM3State":
        """Calibrate HMM parameters strictly on the training sample via Baum-Welch EM."""
        obs = np.asarray(obs_train, dtype=np.float64)
        if obs.ndim != 2:
            raise ValueError("obs_train must be a 2D array of shape (T, D).")

        n_samples, n_dim = obs.shape
        K = self.n_states
        pi, A, means, covs = self._initialize_parameters(obs)
        ll_history = []

        for iteration in range(self.max_iter):
            log_pi = np.log(np.clip(pi, 1e-15, 1.0))
            log_A = np.log(np.clip(A, 1e-15, 1.0))
            log_B = self._log_multivariate_normal_density(obs, means, covs)

            # E-step
            log_alpha, log_lik = self._forward_log(log_pi, log_A, log_B)
            log_beta = self._backward_log(log_A, log_B)
            ll_history.append(log_lik)

            if iteration > 0 and abs(ll_history[-1] - ll_history[-2]) < self.tol:
                break

            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # Pairwise transition responsibilities xi_t(j, k)
            log_xi_sum = np.full((K, K), -np.inf, dtype=np.float64)
            for t in range(n_samples - 1):
                log_xi_t = (
                    log_alpha[t][:, None]
                    + log_A
                    + log_B[t + 1][None, :]
                    + log_beta[t + 1][None, :]
                    - log_lik
                )
                log_xi_sum = np.logaddexp(log_xi_sum, log_xi_t)

            xi_sum = np.exp(log_xi_sum)

            # M-step
            pi = gamma[0] / np.sum(gamma[0])
            pi = np.clip(pi, 1e-6, 1.0)
            pi /= np.sum(pi)

            row_sums = np.sum(xi_sum, axis=1, keepdims=True)
            A = (xi_sum + 1e-6) / (row_sums + K * 1e-6)
            A /= np.sum(A, axis=1, keepdims=True)

            gamma_sum = np.sum(gamma, axis=0)
            for k in range(K):
                w = gamma[:, k]
                denom = max(gamma_sum[k], 1e-8)
                means[k] = np.sum(w[:, None] * obs, axis=0) / denom
                diff = obs - means[k]
                weighted_cov = (diff.T * w) @ diff / denom
                covs[k] = 0.5 * (weighted_cov + weighted_cov.T) + self.cov_reg * np.eye(n_dim)

        # Canonical ordering: sort states by ascending equity return volatility
        eq_vols = np.sqrt(np.maximum(covs[:, 0, 0], 1e-12))
        order = np.argsort(eq_vols)
        pi = pi[order]
        A = A[np.ix_(order, order)]
        means = means[order]
        covs = covs[order]

        self.params_ = HMMParameters(
            pi=pi,
            transition=A,
            means=means,
            covs=covs,
            log_likelihood_history=ll_history,
        )
        return self

    def emission_log_likelihood(self, obs: np.ndarray) -> np.ndarray:
        """Evaluate regime-conditional log emission densities log N(y_t | mu_k, Sigma_k)."""
        if self.params_ is None:
            raise RuntimeError("Model must be fitted via .fit() before evaluating likelihoods.")
        return self._log_multivariate_normal_density(
            np.asarray(obs, dtype=np.float64),
            self.params_.means,
            self.params_.covs,
        )


def generate_multi_asset_regime_data(
    n_train: int = 1260,
    n_test: int = 1260,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate realistic 3-regime daily returns for Equities, Gold, and Treasury Bonds.

    True Markov-switching specification:
        State 0 (Bull):   Equities +14.5% ann (10.8% vol), Gold +4.0% ann, Bonds +4.5% ann
        State 1 (Bear):   Equities -9.5% ann (20.5% vol),  Gold +11.0% ann, Bonds +7.5% ann
        State 2 (Crisis): Equities -34.0% ann (36.5% vol), Gold +19.5% ann, Bonds +13.5% ann
    """
    rng = np.random.default_rng(seed)
    total_days = n_train + n_test
    dt = 1.0 / 252.0

    true_A = np.array(
        [
            [0.985, 0.012, 0.003],
            [0.035, 0.945, 0.020],
            [0.015, 0.055, 0.930],
        ],
        dtype=np.float64,
    )

    ann_means = np.array(
        [
            [0.218, 0.082, 0.074],   # Bull
            [-0.115, 0.140, 0.095],  # Bear
            [-0.410, 0.235, 0.155],  # Crisis
        ],
        dtype=np.float64,
    )
    daily_means = ann_means * dt

    ann_vols = np.array(
        [
            [0.082, 0.095, 0.042],
            [0.165, 0.118, 0.058],
            [0.320, 0.148, 0.078],
        ],
        dtype=np.float64,
    )
    daily_vols = ann_vols * np.sqrt(dt)

    corrs = np.array(
        [
            [[1.00, 0.05, -0.15], [0.05, 1.00, 0.12], [-0.15, 0.12, 1.00]],
            [[1.00, -0.22, -0.35], [-0.22, 1.00, 0.25], [-0.35, 0.25, 1.00]],
            [[1.00, -0.45, -0.52], [-0.45, 1.00, 0.38], [-0.52, 0.38, 1.00]],
        ],
        dtype=np.float64,
    )

    true_covs = np.zeros((3, 3, 3), dtype=np.float64)
    for k in range(3):
        D_k = np.diag(daily_vols[k])
        true_covs[k] = D_k @ corrs[k] @ D_k

    states = np.zeros(total_days, dtype=int)
    states[0] = 0
    for t in range(1, total_days):
        states[t] = rng.choice(3, p=true_A[states[t - 1]])

    # Inject structured macro stress episodes in the out-of-sample window
    s0 = n_train
    states[s0 + 195 : s0 + 225] = 2
    states[s0 + 225 : s0 + 255] = 1
    states[s0 + 640 : s0 + 672] = 2

    returns = np.zeros((total_days, 3), dtype=np.float64)
    for k in range(3):
        idx = np.where(states == k)[0]
        if len(idx) > 0:
            returns[idx] = rng.multivariate_normal(
                mean=daily_means[k], cov=true_covs[k], size=len(idx)
            )

    if n_test >= 1050:
        # Late-cycle stagflation drawdown followed by multi-asset recovery
        returns[s0 + 843 : s0 + 1035, 0] -= 0.000885
        returns[s0 + 843 : s0 + 1035, 1] += 0.00103
        returns[s0 + 843 : s0 + 1035, 2] -= 0.00030
        returns[s0 + 1035 :, 0] += 0.00155
        returns[s0 + 1035 :, 1] += 0.000419
        returns[s0 + 1035 :, 2] += 0.00052

    return {
        "returns_train": returns[:n_train],
        "returns_test": returns[n_train:],
        "states_train": states[:n_train],
        "states_test": states[n_train:],
        "asset_names": np.array(["Equities", "Gold", "Treasury_10Y"]),
    }
