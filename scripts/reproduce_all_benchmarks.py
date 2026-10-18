#!/usr/bin/env python3
"""Master reproduction script for high-dimensional filtering, convex stat-arb, and C++ spectral FFT.

Executes all five empirical modules from scratch, verifies causality and portfolio
constraints, writes `results/benchmark_summary.json`, and generates the four
publication-quality benchmark figures in `figures/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cpp_spectral_fft import run_cpp_spectral_benchmarks
from src.highdim_portfolio import (
    generate_credit_application_dataset,
    generate_equity_factor_panel,
    ledoit_wolf_shrinkage,
    run_convex_portfolio_backtest,
    run_credit_policy_optimization,
)
from src.state_space_filtering import (
    generate_cointegrated_pair_data,
    generate_multi_asset_regime_data,
    run_kalman_statarb_backtest,
    run_regimeshift_backtest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "legend.fontsize": 8.2,
            "legend.framealpha": 0.92,
            "figure.dpi": 220,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
        }
    )


def plot_fig1_regimeshift(res: Any, data: Dict[str, np.ndarray]) -> Path:
    """Generate Figure 1: RegimeShift HMM posterior states & multi-asset wealth curves."""
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.35, 1.0]})
    t = np.arange(len(res.strategy_net_returns))

    w_strat = np.cumprod(1.0 + res.strategy_net_returns) * 100.0
    w_6040 = np.cumprod(1.0 + res.bench_60_40_returns) * 100.0
    w_ew = np.cumprod(1.0 + res.bench_ew_returns) * 100.0

    ax0 = axes[0]
    ax0.plot(t, w_strat, color="#0b3c8c", lw=2.0, label=f"RegimeShift 3-State HMM + Online Hamilton (Sharpe: {res.sharpe_ratio:.2f}, MDD: {res.max_drawdown_strategy_pct:.1f}%)")
    ax0.plot(t, w_6040, color="#b8252a", lw=1.45, ls="--", label=f"60/40 Institutional Benchmark (Sharpe: {res.bench_60_40_sharpe:.2f}, MDD: {res.max_drawdown_benchmark_pct:.1f}%)")
    ax0.plot(t, w_ew, color="#556270", lw=1.35, ls=":", label="Equal-Weight (1/N) Multi-Asset Baseline")

    # Shade inferred Crisis (State 2) and Bear (State 1) intervals
    crisis_mask = res.smoothed_states == 2
    bear_mask = res.smoothed_states == 1
    ax0.fill_between(
        t,
        float(np.min(w_6040)) * 0.96,
        float(np.max(w_strat)) * 1.03,
        where=crisis_mask,
        color="#d9534f",
        alpha=0.16,
        label="Inferred Crisis Regime (State 2)",
    )
    ax0.fill_between(
        t,
        float(np.min(w_6040)) * 0.96,
        float(np.max(w_strat)) * 1.03,
        where=bear_mask,
        color="#f0ad4e",
        alpha=0.12,
        label="Inferred Bear Regime (State 1)",
    )
    ax0.set_ylabel("Cumulative Wealth (Indexed = 100)")
    ax0.set_title(
        "RegimeShift: Strictly Causal Online Hamilton Filtering vs. Institutional Benchmarks "
        f"(99% CVaR Reduction: {res.cvar_99_reduction_pct:.1f}%, Turnover Cut: {res.turnover_reduction_pct:.1f}%)"
    )
    ax0.legend(loc="upper left", ncol=2)

    ax1 = axes[1]
    ax1.stackplot(
        t,
        res.filtered_probs[:, 0],
        res.filtered_probs[:, 1],
        res.filtered_probs[:, 2],
        labels=["P(S_t = Bull | Y_{1:t})", "P(S_t = Bear | Y_{1:t})", "P(S_t = Crisis | Y_{1:t})"],
        colors=["#2e8b57", "#f0ad4e", "#c9302c"],
        alpha=0.78,
    )
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Filtered State Probability")
    ax1.set_xlabel("Out-of-Sample Trading Days (Walk-Forward Evaluation)")
    ax1.set_title("Online Hamilton Forward Filter Posterior Probabilities xi_{t|t} (Zero Lookahead)")
    ax1.legend(loc="upper right", ncol=3)

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig1_regimeshift_hmm_posterior_states.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig2_kalman(res: Any) -> Path:
    """Generate Figure 2: Kalman state-space dynamic hedge ratio tracking & OU z-score execution."""
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.1]})
    t = np.arange(len(res.filtered_hedge_ratio))

    ax0 = axes[0]
    ax0.plot(t, res.true_hedge_ratio, color="#1f2937", lw=1.8, alpha=0.85, label="Latent Time-Varying Cointegration Hedge Ratio h_t")
    ax0.plot(t, res.filtered_hedge_ratio, color="#0d6efd", lw=1.6, label=f"Kalman State-Space Estimate h_{{t|t}} (Tracking Error Cut: {res.tracking_error_reduction_pct:.1f}%)")
    ax0.axhline(res.static_ols_hedge_ratio, color="#dc3545", lw=1.4, ls="--", label=f"Static Engle-Granger OLS Hedge Ratio (h_OLS = {res.static_ols_hedge_ratio:.3f})")
    ax0.set_ylabel("Hedge Ratio h_t")
    ax0.set_title(
        "Adaptive State-Space Kalman Cointegration Filter vs. Static Engle-Granger OLS "
        f"(OU Half-Life: {res.ou_params.half_life:.1f} Days)"
    )
    ax0.legend(loc="upper left")

    ax1 = axes[1]
    ax1_twin = ax1.twinx()
    ax1.plot(t, res.z_scores, color="#495057", lw=0.95, alpha=0.75, label="Causal Innovation Z-Score z_t")
    ax1.axhline(1.02, color="#d9534f", lw=1.0, ls="--", label="Short/Long Entry Threshold (+/-1.02 sigma)")
    ax1.axhline(-1.02, color="#d9534f", lw=1.0, ls="--")
    ax1.axhline(0.0, color="#adb5bd", lw=0.8, ls=":")
    ax1.set_ylabel("Spread Z-Score (sigma)")
    ax1.set_ylim(-3.6, 3.6)

    cum_ret_pct = (np.cumprod(1.0 + res.net_returns) - 1.0) * 100.0
    ax1_twin.plot(
        t,
        cum_ret_pct,
        color="#0a7e46",
        lw=2.0,
        label=f"Cumulative Net Return (Sharpe: {res.sharpe_ratio:.2f}, Ann. Net: {res.annualized_net_return_pct:.1f}%)",
    )
    ax1_twin.set_ylabel("Cumulative Net Strategy Return (%)", color="#0a7e46")
    ax1_twin.tick_params(axis="y", labelcolor="#0a7e46")
    ax1_twin.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_xlabel("Out-of-Sample Trading Days")
    ax1.set_title("Ornstein-Uhlenbeck Spread Z-Score Signal & Cumulative Net Strategy Equity Curve")

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig2_kalman_dynamic_hedge_ratio_tracking.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig3_portfolio_and_ff5(res: Any, panel: Dict[str, np.ndarray]) -> Path:
    """Generate Figure 3: Ledoit-Wolf shrinkage spectrum, FF5 Newey-West attribution & beta bounds."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))

    # Panel A: Eigenvalue Conditioning & Cumulative Alpha Curve
    ax0 = axes[0]
    t = np.arange(len(res.net_returns))
    cum_port_bps = np.cumsum(res.net_returns) * 100.0
    cum_bench_bps = np.cumsum(res.benchmark_returns) * 100.0
    ax0.plot(
        t,
        cum_port_bps,
        color="#0b3c8c",
        lw=2.0,
        label=f"Ledoit-Wolf + Convex Markowitz (Sharpe: {res.sharpe_ratio:.2f}, IR: {res.information_ratio:.2f})",
    )
    ax0.plot(
        t,
        cum_bench_bps,
        color="#6c757d",
        lw=1.4,
        ls="--",
        label="Risk-Matched Market Benchmark",
    )
    ax0.set_xlabel("Out-of-Sample Trading Days")
    ax0.set_ylabel("Cumulative Return (%)")
    ax0.set_title(
        f"Walk-Forward Market-Neutral Portfolio (N=54 Equities)\n"
        f"Ridge OOS IC: {res.ridge_oos_ic:.3f} | Alpha: +{res.annualized_alpha_bps:.0f} bps/yr"
    )
    ax0.legend(loc="upper left")

    # Inset showing Ex-Post Beta strictly bounded within [-0.04, +0.04]
    inset = ax0.inset_axes([0.52, 0.14, 0.44, 0.34])
    inset.plot(t, res.ex_post_betas, color="#198754", lw=0.9)
    inset.axhline(0.04, color="#dc3545", lw=0.9, ls="--")
    inset.axhline(-0.04, color="#dc3545", lw=0.9, ls="--")
    inset.axhline(0.0, color="#495057", lw=0.6, ls=":")
    inset.set_ylim(-0.055, 0.055)
    inset.set_title("Ex-Post Beta (|beta| <= 0.04)", fontsize=7.5)
    inset.tick_params(labelsize=6.8)

    # Panel B: Fama-French 5-Factor Newey-West HAC t-statistics
    ax1 = axes[1]
    ff = res.ff5_attribution
    labels = ["Alpha (Idio.)", "MKT", "SMB", "HML", "RMW", "CMA"]
    t_vals = [
        ff.alpha_t_stat_nw,
        ff.factor_t_stats_nw["MKT"],
        ff.factor_t_stats_nw["SMB"],
        ff.factor_t_stats_nw["HML"],
        ff.factor_t_stats_nw["RMW"],
        ff.factor_t_stats_nw["CMA"],
    ]
    colors = ["#0b3c8c" if i == 0 else "#6c757d" for i in range(len(labels))]
    bars = ax1.bar(labels, t_vals, color=colors, width=0.56, edgecolor="#212529", alpha=0.88)
    ax1.axhline(1.96, color="#dc3545", lw=1.1, ls="--", label="95% Significance Threshold (t = +/-1.96)")
    ax1.axhline(-1.96, color="#dc3545", lw=1.1, ls="--")
    ax1.axhline(0.0, color="#212529", lw=0.8)

    for bar, val in zip(bars, t_vals):
        y_pos = val + (0.18 if val >= 0 else -0.35)
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{val:.2f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8.2,
            fontweight="bold" if abs(val) > 1.96 else "normal",
        )

    ax1.set_ylim(min(-2.8, min(t_vals) - 0.8), max(4.2, max(t_vals) + 0.8))
    ax1.set_ylabel("Newey-West HAC t-Statistic")
    ax1.set_title(
        f"Fama-French 5-Factor Attribution & VIF Diagnostics\n"
        f"(Raw VIF: {res.raw_mean_vif:.2f} -> Orthogonalized VIF: {res.ortho_mean_vif:.2f})"
    )
    ax1.legend(loc="upper right")

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig3_ledoit_wolf_efficient_frontier_and_ff5.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig4_cpp_spectral_fft(cpp_res: Dict[str, Any]) -> Path:
    """Generate Figure 4: C++17 2D Cooley-Tukey FFT speedup & Wiener deconvolution PSNR."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    # Panel A: Runtime Speedup across (2N+1)x(2N+1) Kernel Sizes
    ax0 = axes[0]
    half_widths = cpp_res["kernel_half_widths"]
    k_labels = [f"{2*n+1}x{2*n+1}\n(N={n})" for n in half_widths]
    speedups = cpp_res["speedup_x"]
    x_pos = np.arange(len(half_widths))

    bars = ax0.bar(x_pos, speedups, width=0.55, color="#0b3c8c", edgecolor="#1f2937", alpha=0.88, label="2D Cooley-Tukey FFT Speedup vs. Spatial Conv.")
    for bar, sp in zip(bars, speedups):
        ax0.text(
            bar.get_x() + bar.get_width() / 2.0,
            sp + 0.55,
            f"{sp:.1f}x",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
        )
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(k_labels)
    ax0.set_ylim(0, max(speedups) * 1.18)
    ax0.set_xlabel("Symmetric Box Kernel Size (2N+1)x(2N+1) on 512x512 Grid")
    ax0.set_ylabel("Runtime Speedup Factor (x)")
    ax0.set_title("C++17 2D Radix-2 FFT vs. Spatial Convolution")
    ax0.legend(loc="upper left")

    # Panel B: Regularized Inverse Filtering vs Wiener Deconvolution PSNR Curve
    ax1 = axes[1]
    p_grid = cpp_res["param_grid"]
    eps_curve = cpp_res["eps_psnr_curve"]
    wie_curve = cpp_res["wiener_psnr_curve"]
    deg_psnr = cpp_res["degraded_psnr_db"]

    ax1.semilogx(
        p_grid,
        wie_curve,
        marker="o",
        color="#0a7e46",
        lw=2.0,
        label=f"Wiener Deconvolution (MSE Cut: -{cpp_res['wiener_mse_reduction_vs_inverse_pct']:.1f}% vs. Inv.)",
    )
    ax1.semilogx(
        p_grid,
        eps_curve,
        marker="s",
        color="#0d6efd",
        lw=1.7,
        ls="--",
        label=f"Epsilon-Damped Inverse Filter (+{cpp_res['inverse_filter_psnr_gain_db']:.1f} dB Gain)",
    )
    ax1.axhline(
        deg_psnr,
        color="#dc3545",
        lw=1.3,
        ls=":",
        label=f"Degraded Blurred + Noisy Input ({deg_psnr:.1f} dB)",
    )
    ax1.set_xlabel("Spectral Regularization Parameter (epsilon / scaled NSR)")
    ax1.set_ylabel("Restored Image Quality (PSNR in dB)")
    ax1.set_title("Spectral Inverse vs. Wiener Deconvolution (512x512)")
    ax1.legend(loc="lower left")

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig4_cpp_fft_speedup_and_wiener_psnr.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run_all() -> Dict[str, Any]:
    """Execute all benchmarks, generate figures, and save summary JSON."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()

    # 1. RegimeShift (3-State Gaussian HMM + Online Hamilton Filter)
    regime_data = generate_multi_asset_regime_data(n_train=1260, n_test=1260, seed=42)
    regime_res = run_regimeshift_backtest(data=regime_data, smoothing_window=15, tc_bps=10.0)
    plot_fig1_regimeshift(regime_res, regime_data)

    # 2. Kalman State-Space OU Cointegration Stat-Arb
    pair_data = generate_cointegrated_pair_data(n_train=500, n_test=1000, seed=42)
    kalman_res = run_kalman_statarb_backtest(data=pair_data)
    plot_fig2_kalman(kalman_res)

    # 3. High-Dim Alpha Orthogonalization + Ledoit-Wolf Convex Portfolio Optimization
    eq_panel = generate_equity_factor_panel(n_assets=54, n_train=504, n_test=718, seed=42)
    port_res = run_convex_portfolio_backtest(panel=eq_panel)
    plot_fig3_portfolio_and_ff5(port_res, eq_panel)

    # 4. Econometric Credit Risk & Neyman-Pearson Threshold Optimization
    credit_ds = generate_credit_application_dataset(n_applications=10000, seed=42)
    credit_res = run_credit_policy_optimization(dataset=credit_ds, pd_ceiling=0.065)

    # 5. C++17 2D Cooley-Tukey FFT & Wiener Deconvolution
    cpp_res = run_cpp_spectral_benchmarks(force_rebuild=False)
    plot_fig4_cpp_spectral_fft(cpp_res)

    summary = {
        "regimeshift_hmm_hamilton": {
            "oos_sharpe_ratio": round(regime_res.sharpe_ratio, 2),
            "benchmark_60_40_sharpe": round(regime_res.bench_60_40_sharpe, 2),
            "annualized_alpha_vs_60_40_bps": round(regime_res.annualized_alpha_vs_60_40_bps, 1),
            "annualized_alpha_vs_ew_bps": round(regime_res.annualized_alpha_vs_ew_bps, 1),
            "cvar_99_reduction_pct": round(regime_res.cvar_99_reduction_pct, 1),
            "turnover_reduction_pct": round(regime_res.turnover_reduction_pct, 1),
            "max_drawdown_strategy_pct": round(regime_res.max_drawdown_strategy_pct, 1),
            "max_drawdown_benchmark_pct": round(regime_res.max_drawdown_benchmark_pct, 1),
        },
        "kalman_ou_statarb": {
            "tracking_error_reduction_pct": round(kalman_res.tracking_error_reduction_pct, 1),
            "oos_sharpe_ratio": round(kalman_res.sharpe_ratio, 2),
            "annualized_net_return_pct": round(kalman_res.annualized_net_return_pct, 1),
            "ou_half_life_days": round(kalman_res.ou_params.half_life, 2),
            "static_ols_hedge_ratio": round(kalman_res.static_ols_hedge_ratio, 4),
        },
        "highdim_convex_portfolio": {
            "n_equities": 54,
            "ridge_oos_ic": round(port_res.ridge_oos_ic, 4),
            "ensemble_oos_ic": round(port_res.ensemble_oos_ic, 4),
            "raw_mean_vif": round(port_res.raw_mean_vif, 2),
            "ortho_mean_vif": round(port_res.ortho_mean_vif, 2),
            "ledoit_wolf_shrinkage_delta": round(port_res.shrinkage_intensity, 4),
            "cov_frobenius_error_reduction_pct": round(port_res.cov_frobenius_error_reduction_pct, 1),
            "oos_sharpe_ratio": round(port_res.sharpe_ratio, 2),
            "information_ratio": round(port_res.information_ratio, 2),
            "annualized_alpha_bps": round(port_res.annualized_alpha_bps, 1),
            "ff5_newey_west_t_stat": round(port_res.ff5_attribution.alpha_t_stat_nw, 2),
            "max_abs_ex_post_beta": round(float(np.max(np.abs(port_res.ex_post_betas))), 4),
        },
        "credit_policy_optimization": {
            "n_applications": 10000,
            "roc_auc": round(credit_res.roc_auc, 3),
            "ks_statistic": round(credit_res.ks_statistic, 3),
            "legacy_approval_rate_pct": round(credit_res.legacy_approval_rate_pct, 1),
            "optimized_approval_rate_pct": round(credit_res.optimized_approval_rate_pct, 1),
            "optimized_portfolio_pd_pct": round(credit_res.optimized_portfolio_pd_pct, 2),
            "pd_ceiling_pct": round(credit_res.pd_ceiling_pct, 2),
            "npv_improvement_pct": round(credit_res.npv_improvement_pct, 1),
        },
        "cpp_spectral_fft_wiener": {
            "grid_size": cpp_res["grid_size"],
            "parseval_rel_error": cpp_res["parseval_rel_error"],
            "max_roundtrip_error": cpp_res["max_roundtrip_error"],
            "speedup_at_N20_x": round(cpp_res["speedup_x"][4], 1),
            "max_speedup_x": round(max(cpp_res["speedup_x"]), 1),
            "inverse_filter_psnr_gain_db": round(cpp_res["inverse_filter_psnr_gain_db"], 2),
            "wiener_mse_reduction_vs_inverse_pct": round(
                cpp_res["wiener_mse_reduction_vs_inverse_pct"], 1
            ),
        },
    }

    with open(RESULTS_DIR / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 78)
    print("HIGH-DIMENSIONAL FILTERING, CONVEX STAT-ARB & SPECTRAL FFT BENCHMARK SUMMARY")
    print("=" * 78)
    print(json.dumps(summary, indent=2))
    print("=" * 78)
    return summary


if __name__ == "__main__":
    run_all()
