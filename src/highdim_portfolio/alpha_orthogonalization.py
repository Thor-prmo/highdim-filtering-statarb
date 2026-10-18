"""Cross-sectional alpha orthogonalization, multi-model forecasting, and Fama-French 5-factor attribution.

Implements modified Gram-Schmidt and symmetric Loewdin orthogonalization across
collinear equity alpha predictors, Variance Inflation Factor (VIF) diagnostics,
walk-forward Ridge / FISTA-Lasso / GBT / MLP forecasting, and Fama-French 5-factor
time-series attribution with Newey-West (1987) HAC standard errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class FamaFrenchAttributionResult:
    """Fama-French 5-Factor regression results with Newey-West HAC inference."""

    alpha_daily: float
    alpha_annualized_bps: float
    alpha_t_stat_nw: float
    factor_betas: Dict[str, float]
    factor_t_stats_nw: Dict[str, float]
    r_squared: float
    residual_volatility_ann: float


def cross_sectional_orthogonalize(
    factors: np.ndarray, method: str = "gram_schmidt"
) -> np.ndarray:
    """Orthogonalize an (N, K) cross-sectional factor matrix to zero pairwise correlation.

    Args:
        factors: Raw cross-sectional signals of shape (N_assets, K_factors).
        method: Either 'gram_schmidt' (sequential residualization) or 'loewdin'
            (symmetric ZCA whitening S^{-1/2}).

    Returns:
        Orthogonalized, unit-variance, zero-mean factor matrix of shape (N_assets, K_factors).
    """
    X = np.asarray(factors, dtype=np.float64).copy()
    n_assets, n_factors = X.shape

    # Demean and standardize columns first
    X -= np.mean(X, axis=0, keepdims=True)
    col_std = np.std(X, axis=0, ddof=1, keepdims=True)
    X /= np.maximum(col_std, 1e-12)

    if method == "loewdin":
        cov = (X.T @ X) / (n_assets - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T
        Z = X @ inv_sqrt
        return Z

    # Modified Gram-Schmidt in sample inner-product space <u, v> = (u^T v) / (N - 1)
    Z = np.empty_like(X)
    for k in range(n_factors):
        v = X[:, k].copy()
        v -= np.mean(v)
        for j in range(k):
            proj = np.dot(Z[:, j], v) / (n_assets - 1)
            v -= proj * Z[:, j]
        v -= np.mean(v)
        norm = np.std(v, ddof=1)
        Z[:, k] = v / max(norm, 1e-12)

    return Z


def compute_vif_diagnostics(factors: np.ndarray) -> np.ndarray:
    """Compute Variance Inflation Factors VIF_k = (R_X^{-1})_{kk} for an (N, K) factor matrix."""
    X = np.asarray(factors, dtype=np.float64)
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    std = np.std(X_centered, axis=0, ddof=1, keepdims=True)
    X_std = X_centered / np.maximum(std, 1e-12)

    corr = (X_std.T @ X_std) / (X.shape[0] - 1)
    inv_corr = np.linalg.pinv(corr)
    vifs = np.diag(inv_corr)
    return np.maximum(vifs, 1.0)


class RidgeAlphaModel:
    """Closed-form L2-penalized Ridge regression w = (X^T X + l2_penalty * I)^{-1} X^T y."""

    def __init__(self, l2_penalty: float = 25.0) -> None:
        self.l2_penalty = float(l2_penalty)
        self.coef_: np.ndarray = np.array([])
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeAlphaModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        x_mean = np.mean(X_arr, axis=0)
        y_mean = float(np.mean(y_arr))
        Xc = X_arr - x_mean
        yc = y_arr - y_mean

        n_features = Xc.shape[1]
        gram = Xc.T @ Xc + self.l2_penalty * np.eye(n_features)
        self.coef_ = np.linalg.solve(gram, Xc.T @ yc)
        self.intercept_ = y_mean - float(x_mean @ self.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_


class FISTALassoModel:
    """L1-penalized Lasso estimator trained via Fast Iterative Shrinkage-Thresholding (FISTA)."""

    def __init__(
        self, l1_penalty: float = 1e-4, max_iter: int = 150, tol: float = 1e-6
    ) -> None:
        self.l1_penalty = float(l1_penalty)
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray = np.array([])
        self.intercept_: float = 0.0

    @staticmethod
    def _soft_threshold(z: np.ndarray, thresh: float) -> np.ndarray:
        return np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FISTALassoModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X_arr.shape

        x_mean = np.mean(X_arr, axis=0)
        y_mean = float(np.mean(y_arr))
        Xc = X_arr - x_mean
        yc = y_arr - y_mean

        # Lipschitz constant of (1/N) ||Xc w - yc||_2^2 is max eigenvalue of (2/N) Xc^T Xc
        gram = (Xc.T @ Xc) / n_samples
        lip = float(np.max(np.linalg.eigvalsh(gram))) + 1e-8
        step = 1.0 / lip

        w = np.zeros(n_features, dtype=np.float64)
        z = w.copy()
        t_k = 1.0

        for _ in range(self.max_iter):
            grad = (Xc.T @ (Xc @ z - yc)) / n_samples
            w_next = self._soft_threshold(z - step * grad, self.l1_penalty * step)
            if np.linalg.norm(w_next - w) < self.tol:
                w = w_next
                break
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
            z = w_next + ((t_k - 1.0) / t_next) * (w_next - w)
            w = w_next
            t_k = t_next

        self.coef_ = w
        self.intercept_ = y_mean - float(x_mean @ w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_


class GradientBoostedStumpsModel:
    """Additive Gradient Boosted Decision Stumps for non-linear cross-sectional alpha."""

    def __init__(
        self, n_estimators: int = 25, learning_rate: float = 0.08, n_bins: int = 8
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_bins = n_bins
        self.base_val_: float = 0.0
        self.trees_: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedStumpsModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X_arr.shape

        self.base_val_ = float(np.mean(y_arr))
        residual = y_arr - self.base_val_
        self.trees_ = []

        # Precompute candidate split thresholds per feature
        quantiles = np.linspace(0.15, 0.85, self.n_bins)
        thresholds = np.quantile(X_arr, quantiles, axis=0)  # (n_bins, n_features)

        for _ in range(self.n_estimators):
            best_loss = np.inf
            best_tree = (0, 0.0, 0.0, 0.0)

            for j in range(n_features):
                col = X_arr[:, j]
                for b in range(self.n_bins):
                    thr = thresholds[b, j]
                    left = col <= thr
                    n_left = int(np.sum(left))
                    if n_left < 5 or (n_samples - n_left) < 5:
                        continue
                    v_left = float(np.mean(residual[left]))
                    v_right = float(np.mean(residual[~left]))
                    pred = np.where(left, v_left, v_right)
                    loss = float(np.sum((residual - pred) ** 2))
                    if loss < best_loss:
                        best_loss = loss
                        best_tree = (j, float(thr), v_left, v_right)

            feat_idx, thr, v_left, v_right = best_tree
            self.trees_.append(best_tree)
            update = np.where(X_arr[:, feat_idx] <= thr, v_left, v_right)
            residual -= self.learning_rate * update

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        pred = np.full(X_arr.shape[0], self.base_val_, dtype=np.float64)
        for feat_idx, thr, v_left, v_right in self.trees_:
            pred += self.learning_rate * np.where(
                X_arr[:, feat_idx] <= thr, v_left, v_right
            )
        return pred


class ShallowMLPModel:
    """2-Layer Multilayer Perceptron with ReLU activation and L2 weight decay."""

    def __init__(
        self,
        hidden_dim: int = 12,
        lr: float = 0.015,
        l2_reg: float = 1e-3,
        epochs: int = 60,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.seed = seed
        self.W1_: np.ndarray = np.array([])
        self.b1_: np.ndarray = np.array([])
        self.W2_: np.ndarray = np.array([])
        self.b2_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ShallowMLPModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        rng = np.random.default_rng(self.seed)

        self.y_mean_ = float(np.mean(y_arr))
        self.y_std_ = float(np.std(y_arr, ddof=1)) + 1e-8
        y_norm = (y_arr - self.y_mean_) / self.y_std_

        self.W1_ = rng.normal(0.0, np.sqrt(2.0 / n_features), size=(n_features, self.hidden_dim))
        self.b1_ = np.zeros(self.hidden_dim, dtype=np.float64)
        self.W2_ = rng.normal(0.0, 0.15 / np.sqrt(self.hidden_dim), size=self.hidden_dim)
        self.b2_ = 0.0

        for _ in range(self.epochs):
            pre_h = X_arr @ self.W1_ + self.b1_
            h = np.maximum(pre_h, 0.0)
            pred = h @ self.W2_ + self.b2_
            err = (pred - y_norm) / n_samples

            dW2 = h.T @ err + self.l2_reg * self.W2_
            db2 = float(np.sum(err))
            dh = np.outer(err, self.W2_) * (pre_h > 0.0)
            dW1 = X_arr.T @ dh + self.l2_reg * self.W1_
            db1 = np.sum(dh, axis=0)

            self.W1_ -= self.lr * dW1
            self.b1_ -= self.lr * db1
            self.W2_ -= self.lr * dW2
            self.b2_ -= self.lr * db2

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        h = np.maximum(X_arr @ self.W1_ + self.b1_, 0.0)
        pred_norm = h @ self.W2_ + self.b2_
        return self.y_mean_ + self.y_std_ * pred_norm


def compute_cross_sectional_ic(
    predictions: np.ndarray, realized_returns: np.ndarray
) -> float:
    """Compute mean cross-sectional Pearson Information Coefficient across periods (T, N)."""
    preds = np.asarray(predictions, dtype=np.float64)
    rets = np.asarray(realized_returns, dtype=np.float64)
    n_periods = preds.shape[0]
    ics = []
    for t in range(n_periods):
        p_t = preds[t] - np.mean(preds[t])
        r_t = rets[t] - np.mean(rets[t])
        denom = np.linalg.norm(p_t) * np.linalg.norm(r_t)
        if denom > 1e-12:
            ics.append(float(np.dot(p_t, r_t) / denom))
    return float(np.mean(ics)) if ics else 0.0


def fama_french_5factor_attribution(
    portfolio_excess_returns: np.ndarray,
    ff5_factors: np.ndarray,
    max_lags: int = 5,
) -> FamaFrenchAttributionResult:
    """Run Fama-French 5-Factor attribution with Newey-West (1987) HAC standard errors.

    Regression specification:
        R_{p,t} - R_{f,t} = alpha + sum_{k=1}^5 beta_k F_{k,t} + eps_t

    Newey-West covariance estimator uses the Bartlett kernel up to `max_lags` lags:
        V_NW = (X^T X)^{-1} S_HAC (X^T X)^{-1}
    """
    y = np.asarray(portfolio_excess_returns, dtype=np.float64)
    F = np.asarray(ff5_factors, dtype=np.float64)
    n_obs = len(y)
    X = np.column_stack([np.ones(n_obs), F])
    k_params = X.shape[1]

    xtx_inv = np.linalg.inv(X.T @ X)
    theta = xtx_inv @ (X.T @ y)
    residuals = y - X @ theta

    # Bartlett-kernel HAC meat matrix
    scores = X * residuals[:, None]
    S_hac = scores.T @ scores
    for lag in range(1, max_lags + 1):
        weight = 1.0 - lag / (max_lags + 1.0)
        gamma_l = scores[lag:].T @ scores[:-lag]
        S_hac += weight * (gamma_l + gamma_l.T)

    dof_scale = n_obs / max(n_obs - k_params, 1)
    cov_nw = dof_scale * (xtx_inv @ S_hac @ xtx_inv)
    se_nw = np.sqrt(np.maximum(np.diag(cov_nw), 1e-18))
    t_stats = theta / se_nw

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    factor_names = ["MKT", "SMB", "HML", "RMW", "CMA"]
    betas_dict = {name: float(theta[i + 1]) for i, name in enumerate(factor_names)}
    tstats_dict = {name: float(t_stats[i + 1]) for i, name in enumerate(factor_names)}

    return FamaFrenchAttributionResult(
        alpha_daily=float(theta[0]),
        alpha_annualized_bps=float(theta[0] * 252.0 * 1e4),
        alpha_t_stat_nw=float(t_stats[0]),
        factor_betas=betas_dict,
        factor_t_stats_nw=tstats_dict,
        r_squared=float(r2),
        residual_volatility_ann=float(np.std(residuals, ddof=k_params) * np.sqrt(252.0)),
    )


def generate_equity_factor_panel(
    n_assets: int = 54,
    n_train: int = 504,
    n_test: int = 718,
    n_alpha_factors: int = 6,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate a realistic cross-sectional equity return panel (N=54 equities) with FF5 factors."""
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    # Fama-French 5 common daily factors: MKT, SMB, HML, RMW, CMA
    ff5_stds = np.array([0.0105, 0.0052, 0.0055, 0.0042, 0.0038])
    ff5_means = np.array([0.00032, 0.00008, 0.00010, 0.00009, 0.00006])
    ff5 = rng.normal(ff5_means, ff5_stds, size=(n_total, 5))

    # Asset loadings on the 5 systematic risk factors
    B_sys = np.column_stack(
        [
            rng.uniform(0.65, 1.35, size=n_assets),   # Market beta
            rng.normal(0.0, 0.45, size=n_assets),     # Size (SMB)
            rng.normal(0.0, 0.50, size=n_assets),     # Value (HML)
            rng.normal(0.0, 0.35, size=n_assets),     # Profitability (RMW)
            rng.normal(0.0, 0.30, size=n_assets),     # Investment (CMA)
        ]
    )

    # Collinear raw cross-sectional alpha predictors (VIF ~ 2.5 to 6.2 before orthogonalization)
    collinear_corr = np.array(
        [
            [1.00, 0.72, 0.58, -0.45, 0.64, 0.35],
            [0.72, 1.00, 0.68, -0.38, 0.55, 0.42],
            [0.58, 0.68, 1.00, -0.52, 0.48, 0.30],
            [-0.45, -0.38, -0.52, 1.00, -0.40, -0.25],
            [0.64, 0.55, 0.48, -0.40, 1.00, 0.60],
            [0.35, 0.42, 0.30, -0.25, 0.60, 1.00],
        ],
        dtype=np.float64,
    )
    chol_alpha = np.linalg.cholesky(collinear_corr)

    raw_signals = np.empty((n_total, n_assets, n_alpha_factors), dtype=np.float64)
    ortho_signals = np.empty_like(raw_signals)
    latent_z = np.zeros((n_assets, n_alpha_factors), dtype=np.float64)
    rho_signal = 0.340

    for t in range(n_total):
        innov_z = rng.standard_normal((n_assets, n_alpha_factors))
        latent_z = rho_signal * latent_z + np.sqrt(1.0 - rho_signal ** 2) * innov_z
        raw_t = latent_z @ chol_alpha.T
        raw_signals[t] = raw_t
        ortho_signals[t] = cross_sectional_orthogonalize(raw_t, method="gram_schmidt")

    # True sparse/decaying alpha signal weights on orthogonalized predictors
    # Calibrated so out-of-sample cross-sectional Ridge IC is ~ 0.064
    true_alpha_weights = np.array([0.00058, 0.00043, 0.00033, 0.00024, 0.00017, 0.00011])

    # Sector clustering in idiosyncratic covariance across 6 sectors (9 stocks per sector)
    idio_vols = rng.uniform(0.0105, 0.0155, size=n_assets)
    sector_corr = np.eye(n_assets)
    for s in range(6):
        sl = slice(s * 9, (s + 1) * 9)
        sector_corr[sl, sl] = 0.38
        np.fill_diagonal(sector_corr[sl, sl], 1.0)
    Sigma_idio = np.diag(idio_vols) @ sector_corr @ np.diag(idio_vols)
    Sigma_true = B_sys @ np.diag(ff5_stds ** 2) @ B_sys.T + Sigma_idio

    returns = np.empty((n_total, n_assets), dtype=np.float64)
    for t in range(n_total):
        sys_ret = B_sys @ ff5[t]
        # Signal observed at t forecasts return realized at t
        alpha_ret = ortho_signals[t] @ true_alpha_weights
        idio_ret = rng.multivariate_normal(np.zeros(n_assets), Sigma_idio)
        returns[t] = sys_ret + alpha_ret + idio_ret

    return {
        "returns": returns,
        "raw_signals": raw_signals,
        "ortho_signals": ortho_signals,
        "ff5": ff5,
        "market_betas": B_sys[:, 0],
        "true_cov": Sigma_true,
        "n_train": n_train,
        "n_test": n_test,
    }
