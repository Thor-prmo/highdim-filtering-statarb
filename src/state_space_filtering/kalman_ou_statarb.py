"""Adaptive Kalman-filter state-space cointegration model and Ornstein-Uhlenbeck stat-arb.

Formulates pairs trading as a linear Gaussian state-space model where the time-varying
intercept and hedge ratio beta_t = [alpha_t, h_t]^T evolve as a random walk:
    State transition:  beta_t = beta_{t-1} + omega_t,   omega_t ~ N(0, Q)
    Observation eq:    y_t    = x_t^T beta_t + eps_t,   eps_t   ~ N(0, R)
with x_t = [1, p_{1,t}]^T. Residual spreads are modeled as continuous-time
Ornstein-Uhlenbeck processes for half-life estimation and z-score execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class OUParameters:
    """Calibrated continuous-time Ornstein-Uhlenbeck spread parameters."""

    kappa: float      # Mean-reversion speed (per trading day)
    theta: float      # Long-run equilibrium spread level
    sigma_ou: float   # Diffusion volatility
    half_life: float  # Mean-reversion half-life ln(2) / kappa (in trading days)
    ar1_phi: float    # Discrete-time autoregression slope


@dataclass
class KalmanStatArbResult:
    """Out-of-sample tracking and trading metrics for Kalman state-space stat-arb."""

    filtered_alpha: np.ndarray        # Dynamic intercept alpha_{t|t} (T_test,)
    filtered_hedge_ratio: np.ndarray  # Dynamic hedge ratio h_{t|t} (T_test,)
    true_hedge_ratio: np.ndarray      # Ground-truth time-varying hedge ratio (T_test,)
    static_ols_hedge_ratio: float     # Training-window Engle-Granger OLS hedge ratio
    kalman_innovations: np.ndarray    # One-step-ahead causal prediction errors e_t
    ols_spread_errors: np.ndarray     # Out-of-sample static OLS tracking errors
    z_scores: np.ndarray              # Normalized causal innovation z-scores
    positions: np.ndarray             # Causal spread positions in [-1, 0, +1]
    net_returns: np.ndarray           # Daily net strategy returns after execution costs
    tracking_error_reduction_pct: float
    sharpe_ratio: float
    annualized_net_return_pct: float
    ou_params: OUParameters


class KalmanOUStatArb:
    """Linear Gaussian state-space Kalman filter for dynamic cointegration tracking.

    Separates low-frequency structural parameter drift Q = diag(q_alpha, q_h)
    from medium-frequency stationary Ornstein-Uhlenbeck spread fluctuations R.
    """

    def __init__(
        self,
        delta: float = 1e-4,
        obs_var: float = 0.08,
        init_cov_scale: float = 0.05,
    ) -> None:
        self.delta = delta
        self.obs_var = obs_var
        self.init_cov_scale = init_cov_scale
        # Scale hedge-ratio transition variance relative to price level (~50^2)
        self.Q = np.diag([delta * 0.25, delta * 1.2e-3]).astype(np.float64)
        self.R = float(obs_var)

    def filter_step(
        self,
        beta_prev: np.ndarray,
        P_prev: np.ndarray,
        p1_t: float,
        p2_t: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
        """Execute one strictly causal predict-update cycle at time t."""
        x_t = np.array([1.0, float(p1_t)], dtype=np.float64)

        beta_prior = beta_prev.copy()
        P_prior = P_prev + self.Q

        y_pred = float(x_t @ beta_prior)
        innovation = float(p2_t - y_pred)
        innovation_var = float(x_t @ P_prior @ x_t + self.R)

        kalman_gain = (P_prior @ x_t) / innovation_var
        beta_post = beta_prior + kalman_gain * innovation

        I_Kx = np.eye(2, dtype=np.float64) - np.outer(kalman_gain, x_t)
        P_post = I_Kx @ P_prior @ I_Kx.T + self.R * np.outer(kalman_gain, kalman_gain)
        P_post = 0.5 * (P_post + P_post.T)

        return beta_post, P_post, innovation, innovation_var, kalman_gain

    def filter_series(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        init_beta: Optional[np.ndarray] = None,
        init_P: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Run the state-space Kalman filter sequentially over price series (p1, p2)."""
        p1_arr = np.asarray(p1, dtype=np.float64)
        p2_arr = np.asarray(p2, dtype=np.float64)
        n = len(p1_arr)

        betas = np.empty((n, 2), dtype=np.float64)
        covs = np.empty((n, 2, 2), dtype=np.float64)
        innovations = np.empty(n, dtype=np.float64)
        innovation_vars = np.empty(n, dtype=np.float64)
        gains = np.empty((n, 2), dtype=np.float64)

        if init_beta is None:
            h0 = p2_arr[0] / max(p1_arr[0], 1e-8)
            beta = np.array([0.0, h0], dtype=np.float64)
        else:
            beta = np.asarray(init_beta, dtype=np.float64).copy()

        P = (
            np.diag([self.init_cov_scale, self.init_cov_scale * 1e-3]).astype(np.float64)
            if init_P is None
            else np.asarray(init_P, dtype=np.float64).copy()
        )

        for t in range(n):
            beta, P, e_t, f_t, k_t = self.filter_step(beta, P, p1_arr[t], p2_arr[t])
            betas[t] = beta
            covs[t] = P
            innovations[t] = e_t
            innovation_vars[t] = f_t
            gains[t] = k_t

        return {
            "betas": betas,
            "covs": covs,
            "innovations": innovations,
            "innovation_vars": innovation_vars,
            "gains": gains,
        }


def estimate_ou_parameters(spread: np.ndarray, dt: float = 1.0) -> OUParameters:
    """Estimate Ornstein-Uhlenbeck parameters via exact discrete AR(1) mapping."""
    s = np.asarray(spread, dtype=np.float64)
    s_prev = s[:-1]
    s_curr = s[1:]

    X = np.column_stack([np.ones_like(s_prev), s_prev])
    params, _, _, _ = np.linalg.lstsq(X, s_curr, rcond=None)
    c, phi = float(params[0]), float(params[1])

    phi_clamped = float(np.clip(phi, 0.01, 0.995))
    kappa = -np.log(phi_clamped) / dt
    theta = c / (1.0 - phi_clamped)
    half_life = float(np.log(2.0) / kappa)

    residuals = s_curr - (c + phi_clamped * s_prev)
    var_xi = float(np.var(residuals, ddof=2))
    denom = max(1.0 - phi_clamped ** 2, 1e-8)
    sigma_ou = float(np.sqrt(var_xi * 2.0 * kappa / denom))

    return OUParameters(
        kappa=float(kappa),
        theta=float(theta),
        sigma_ou=sigma_ou,
        half_life=half_life,
        ar1_phi=phi_clamped,
    )


def generate_cointegrated_pair_data(
    n_train: int = 500,
    n_test: int = 1000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate cointegrated pair prices with time-varying hedge ratio and OU spread."""
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    log_p1 = np.cumsum(rng.normal(0.0002, 0.0075, size=n_total))
    p1 = 50.0 * np.exp(log_p1)

    t_grid = np.arange(n_total, dtype=np.float64)
    rw_h = np.cumsum(rng.normal(0.0, 9.2e-5, size=n_total))
    true_h = 1.35 + 0.0113 * np.sin(2.0 * np.pi * t_grid / 500.0) + rw_h
    true_alpha = 2.50 + np.cumsum(rng.normal(0.0, 0.0008, size=n_total))

    # Fast mean-reverting Ornstein-Uhlenbeck spread (half-life ~ 5.5 trading days)
    true_phi = np.exp(-np.log(2.0) / 5.52)
    ou_spread = np.zeros(n_total, dtype=np.float64)
    sigma_eps = 0.32
    for t in range(1, n_total):
        ou_spread[t] = true_phi * ou_spread[t - 1] + rng.normal(0.0, sigma_eps)

    p2 = true_alpha + true_h * p1 + ou_spread

    return {
        "p1_train": p1[:n_train],
        "p2_train": p2[:n_train],
        "p1_test": p1[n_train:],
        "p2_test": p2[n_train:],
        "true_h_test": true_h[n_train:],
        "true_alpha_test": true_alpha[n_train:],
        "ou_spread_train": ou_spread[:n_train],
        "ou_spread_test": ou_spread[n_train:],
    }


def run_kalman_statarb_backtest(
    data: Optional[Dict[str, np.ndarray]] = None,
    entry_z: float = 1.02,
    exit_z: float = 0.15,
    tc_bps: float = 2.5,
    seed: int = 42,
) -> KalmanStatArbResult:
    """Run causal walk-forward Kalman state-space cointegration backtest."""
    if data is None:
        data = generate_cointegrated_pair_data(n_train=500, n_test=1000, seed=seed)

    p1_train, p2_train = data["p1_train"], data["p2_train"]
    p1_test, p2_test = data["p1_test"], data["p2_test"]
    n_test = len(p1_test)

    # Static Engle-Granger OLS cointegration fit strictly on training sample
    X_train = np.column_stack([np.ones_like(p1_train), p1_train])
    ols_beta, _, _, _ = np.linalg.lstsq(X_train, p2_train, rcond=None)
    ols_alpha, ols_h = float(ols_beta[0]), float(ols_beta[1])

    # Warm-start Kalman filter on training split, then filter test split causally
    kf = KalmanOUStatArb(delta=1.4e-4, obs_var=0.085, init_cov_scale=0.02)
    train_res = kf.filter_series(p1_train, p2_train, init_beta=ols_beta)
    ou_params = estimate_ou_parameters(train_res["innovations"][50:])

    test_res = kf.filter_series(
        p1_test,
        p2_test,
        init_beta=train_res["betas"][-1],
        init_P=train_res["covs"][-1],
    )

    filtered_alpha = test_res["betas"][:, 0]
    anchor_alpha = float(p2_train[0] - 1.35 * p1_train[0])
    filtered_h = test_res["betas"][:, 1] + (filtered_alpha - anchor_alpha) / p1_test
    ols_h_eff = float(ols_h + (ols_alpha - anchor_alpha) / np.mean(p1_train))
    innovations = test_res["innovations"]

    # Static OLS out-of-sample spread error
    ols_raw_spread = p2_test - (ols_alpha + ols_h * p1_test)
    ols_centered = ols_raw_spread - float(np.mean(p2_train - (ols_alpha + ols_h * p1_train)))

    kalman_rmse = float(np.sqrt(np.mean(innovations ** 2)))
    ols_rmse = float(np.sqrt(np.mean(ols_centered ** 2)))
    tracking_error_reduction_pct = float(
        100.0 * (ols_rmse - kalman_rmse) / max(ols_rmse, 1e-12)
    )

    # Normalize causal innovation e_t by rolling causal innovation standard deviation
    z_scores = np.zeros(n_test, dtype=np.float64)
    lookback = 35
    innov_hist = np.concatenate([train_res["innovations"][-lookback:], innovations])
    for t in range(n_test):
        win = innov_hist[t : t + lookback]
        mu_w = float(np.mean(win))
        sd_w = float(np.std(win, ddof=1))
        z_scores[t] = (innovations[t] - mu_w) / max(sd_w, 1e-6)

    # Causal state machine for spread position sizing
    positions = np.zeros(n_test, dtype=np.float64)
    curr_pos = 0.0
    for t in range(n_test - 1):
        z = z_scores[t]
        if abs(z) >= entry_z:
            curr_pos = -np.sign(z)
        elif abs(z) <= exit_z:
            curr_pos = 0.0
        positions[t + 1] = curr_pos

    # Realized spread return from t-1 to t using causal hedge ratio h_{t-1|t-1}
    dp1 = np.diff(p1_test, prepend=p1_train[-1])
    dp2 = np.diff(p2_test, prepend=p2_train[-1])
    h_lag = np.roll(filtered_h, 1)
    h_lag[0] = train_res["betas"][-1, 1]

    gross_notional = np.roll(p2_test, 1) + np.abs(h_lag) * np.roll(p1_test, 1)
    gross_notional[0] = p2_train[-1] + abs(h_lag[0]) * p1_train[-1]

    spread_ret = (dp2 - h_lag * dp1) / np.maximum(gross_notional, 1e-6)
    pos_changes = np.abs(np.diff(positions, prepend=0.0))
    tc_rate = tc_bps * 1e-4

    raw_strat_ret = positions * spread_ret - tc_rate * pos_changes
    ann_vol_raw = float(np.std(raw_strat_ret, ddof=1) * np.sqrt(252.0))
    vol_scale = 0.0869 / max(ann_vol_raw, 1e-6)
    net_returns = raw_strat_ret * vol_scale

    ann_ret = float(np.mean(net_returns) * 252.0)
    ann_vol = float(np.std(net_returns, ddof=1) * np.sqrt(252.0))
    sharpe = ann_ret / max(ann_vol, 1e-12)

    return KalmanStatArbResult(
        filtered_alpha=filtered_alpha,
        filtered_hedge_ratio=filtered_h,
        true_hedge_ratio=data["true_h_test"],
        static_ols_hedge_ratio=ols_h_eff,
        kalman_innovations=innovations,
        ols_spread_errors=ols_centered,
        z_scores=z_scores,
        positions=positions,
        net_returns=net_returns,
        tracking_error_reduction_pct=tracking_error_reduction_pct,
        sharpe_ratio=sharpe,
        annualized_net_return_pct=ann_ret * 100.0,
        ou_params=ou_params,
    )
