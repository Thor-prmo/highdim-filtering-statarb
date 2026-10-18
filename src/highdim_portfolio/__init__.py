"""High-dimensional factor orthogonalization, Ledoit-Wolf shrinkage, convex Markowitz optimization, and credit policy optimization."""

from .alpha_orthogonalization import (
    FamaFrenchAttributionResult,
    FISTALassoModel,
    GradientBoostedStumpsModel,
    RidgeAlphaModel,
    ShallowMLPModel,
    compute_cross_sectional_ic,
    compute_vif_diagnostics,
    cross_sectional_orthogonalize,
    fama_french_5factor_attribution,
    generate_equity_factor_panel,
)
from .credit_threshold_opt import (
    CreditOptimizationResult,
    generate_credit_application_dataset,
    run_credit_policy_optimization,
)
from .ledoit_wolf_cvxpy import (
    PortfolioBacktestResult,
    ledoit_wolf_shrinkage,
    run_convex_portfolio_backtest,
    solve_markowitz_portfolio,
)

__all__ = [
    "FamaFrenchAttributionResult",
    "RidgeAlphaModel",
    "FISTALassoModel",
    "GradientBoostedStumpsModel",
    "ShallowMLPModel",
    "compute_cross_sectional_ic",
    "compute_vif_diagnostics",
    "cross_sectional_orthogonalize",
    "fama_french_5factor_attribution",
    "generate_equity_factor_panel",
    "PortfolioBacktestResult",
    "ledoit_wolf_shrinkage",
    "solve_markowitz_portfolio",
    "run_convex_portfolio_backtest",
    "CreditOptimizationResult",
    "generate_credit_application_dataset",
    "run_credit_policy_optimization",
]

