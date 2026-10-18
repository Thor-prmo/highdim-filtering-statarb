"""Latent state-space filtering models: Baum-Welch HMM, online Hamilton filter, and Kalman OU stat-arb."""

from .baum_welch_hmm import GaussianHMM3State, generate_multi_asset_regime_data
from .hamilton_online_filter import (
    HamiltonOnlineFilter,
    causal_majority_vote_smoother,
    run_regimeshift_backtest,
)
from .kalman_ou_statarb import (
    KalmanOUStatArb,
    estimate_ou_parameters,
    generate_cointegrated_pair_data,
    run_kalman_statarb_backtest,
)

__all__ = [
    "GaussianHMM3State",
    "generate_multi_asset_regime_data",
    "HamiltonOnlineFilter",
    "causal_majority_vote_smoother",
    "run_regimeshift_backtest",
    "KalmanOUStatArb",
    "estimate_ou_parameters",
    "generate_cointegrated_pair_data",
    "run_kalman_statarb_backtest",
]

