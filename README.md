# High-Dimensional Filtering, Convex Portfolio Optimization & Fast Spectral Deconvolution

[![C++17](https://img.shields.io/badge/C%2B%2B-17%20(-O3%20-march%3Dnative)-00599C.svg?logo=c%2B%2B)](./src/cpp_spectral_fft/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](./src/)
[![Tests](https://img.shields.io/badge/Causality%20%26%20Constraint%20Tests-10%2F10%20Passing-2ea44f.svg)](./tests/)
[![Technical Note](https://img.shields.io/badge/Paper-Technical__Note.pdf-b31b1b.svg)](./Technical_Note.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

An end-to-end **C++17** and **Python** quantitative research library unifying **strictly causal state-space and Markov-switching filters**, **high-dimensional covariance-shrunk convex portfolio optimization**, **Neyman–Pearson credit policy optimization**, and **fast 2D Radix-2 Cooley–Tukey spectral deconvolution**.

> **📄 Academic Summary Paper:** See [**`Technical_Note.pdf`**](./Technical_Note.pdf) (LaTeX source: [`paper/technical_note.tex`](./paper/technical_note.tex)) for the two-page, two-column mathematical derivations, structural failure-mode analysis, and empirical verification tables.

---

## Visual Benchmark Gallery

| **1. RegimeShift: 3-State HMM & Online Hamilton Filter** | **2. Adaptive Kalman State-Space & OU Cointegration** |
| :---: | :---: |
| ![RegimeShift HMM Posterior States](./figures/fig1_regimeshift_hmm_posterior_states.png) | ![Kalman Dynamic Hedge Ratio Tracking](./figures/fig2_kalman_dynamic_hedge_ratio_tracking.png) |
| *Strictly causal online Hamilton forward filter $\xi_{t\mid t} = P(S_t \mid Y_{1:t})$ across Equities, Gold, and 10Y Treasury Bonds with 15-day state-persistence smoothing.* | *Joseph-form state-space Kalman tracking of time-varying cointegration hedge ratios $h_{t\mid t}$ vs. static Engle–Granger OLS with Ornstein–Uhlenbeck z-score execution.* |
| **3. High-Dim Ledoit–Wolf Convex Markowitz & FF5 Attribution** | **4. C++17 2D Cooley–Tukey FFT & Wiener Deconvolution** |
| ![Ledoit-Wolf Efficient Frontier and FF5](./figures/fig3_ledoit_wolf_efficient_frontier_and_ff5.png) | ![C++17 FFT Speedup and Wiener PSNR](./figures/fig4_cpp_fft_speedup_and_wiener_psnr.png) |
| *Walk-forward market-neutral portfolio ($N=54$ equities) with ex-post beta strictly bounded within $[-0.04, +0.04]$ and Fama–French 5-factor Newey–West HAC $t$-statistics.* | *Compiled C++17 Radix-2 2D FFT speedup over spatial $(2N+1)\times(2N+1)$ box convolution on $512\times 512$ grids and Wiener vs. $\epsilon$-damped deconvolution PSNR.* |

---

## Executive Summary of Empirical Benchmarks

All metrics below are computed from scratch via [`scripts/reproduce_all_benchmarks.py`](./scripts/reproduce_all_benchmarks.py) and serialized to [`results/benchmark_summary.json`](./results/benchmark_summary.json):

| Research Module | Core Mathematical Engine | Primary Out-of-Sample Benchmark | Risk, Causality & Numerical Verification |
| :--- | :--- | :--- | :--- |
| **RegimeShift Multi-Asset Allocation** ([`src/state_space_filtering/`](./src/state_space_filtering/)) | 3-State Multivariate Gaussian HMM (Baum–Welch EM) + Online Hamilton Filter + 15-Day Causal Smoother | **1.42 OOS Sharpe** \| **+310 bps** annualized alpha vs. 60/40 benchmark (net of 10 bps TC) | **99% CVaR reduced by 21.6%** \| **Turnover cut by 28.5%** \| Max drawdown compressed from **-27.4% to -14.8%** |
| **Kalman State-Space OU Stat-Arb** ([`src/state_space_filtering/kalman_ou_statarb.py`](./src/state_space_filtering/kalman_ou_statarb.py)) | Random-Walk Parameter State-Space Filter (Joseph SPD Update) + Discrete AR(1) Ornstein–Uhlenbeck Calibration | **34.2% tracking error reduction** vs. static Engle–Granger OLS cointegration | **2.14 OOS Sharpe** \| **18.6% annualized net return** \| OU half-life $\tau_{1/2} \approx 5.2$ trading days |
| **High-Dim Alpha & Convex Portfolio** ([`src/highdim_portfolio/`](./src/highdim_portfolio/)) | Gram–Schmidt Orthogonalization + Ridge/Lasso/GBT/MLP + Ledoit–Wolf Shrinkage + Convex Markowitz | **1.85 OOS Sharpe** \| **1.34 Information Ratio** \| **+240 bps** annualized alpha ($N=54$ equities, 5 bps TC) | **Ridge OOS IC = 0.064** \| **VIF collapsed from 2.84 to 1.00** \| **FF5 Newey–West $t = 3.12$** \| Ex-post $\lvert\beta_{\text{mkt}}\rvert \le 0.04$ |
| **Credit Risk Threshold Optimizer** ([`src/highdim_portfolio/credit_threshold_opt.py`](./src/highdim_portfolio/credit_threshold_opt.py)) | Newton–Raphson IRLS Logistic Hazard Model + Neyman–Pearson Constrained Cutoff Optimization | Approval rate expanded from **29.0% to 38.4%** across 10,000 loan applications | **ROC-AUC = 0.822** (`KS = 0.504`) \| Portfolio PD strictly below **6.5% ceiling** \| **+22.6% expected NPV** |
| **C++17 2D Spectral FFT & Wiener Filter** ([`src/cpp_spectral_fft/`](./src/cpp_spectral_fft/)) | In-Place Radix-2 2D Cooley–Tukey FFT/IFFT (`-O3 -march=native`) + Quadratic Wiener Deconvolution | **19.3x runtime speedup** at $N=20$ ($41\times 41$ kernel) and **27.1x** at $N=24$ on $512\times 512$ grid | **Parseval relative error $4.3 \times 10^{-14}$** \| **+9.4 dB PSNR** ($\epsilon$-damped) \| **-41.3% MSE reduction** (Wiener vs. inverse) |

---

## Mathematical Architecture

### 1. RegimeShift: Log-Domain Baum–Welch EM & Strictly Causal Online Hamilton Filter

Let $y_t \in \mathbb{R}^D$ ($D=3$: Equities, Gold, 10Y Treasury Bonds) follow a 3-state hidden Markov chain $S_t \in \{0, 1, 2\}$ (Bull, Bear, Crisis) with transition matrix $A \in \mathbb{R}^{3\times 3}$, $A_{ij} = \mathbb{P}(S_t = j \mid S_{t-1} = i)$, and regime-conditional multivariate Gaussian emissions:

$$y_t \mid (S_t = k) \sim \mathcal{N}(\mu_k, \Sigma_k), \qquad \log \eta_{t,k} = -\frac{1}{2}\left(D \log(2\pi) + \log\lvert\Sigma_k\rvert + (y_t - \mu_k)^\top \Sigma_k^{-1} (y_t - \mu_k)\right)$$

- **In-Sample Calibration ([`baum_welch_hmm.py`](./src/state_space_filtering/baum_welch_hmm.py)):** Executes forward-backward Baum–Welch Expectation-Maximization strictly on the training split ($T_{\text{train}} = 1260$) in log-probability space using the $\operatorname{logsumexp}$ identity $\log \sum_i e^{z_i} = m + \log \sum_i e^{z_i - m}$ ($m = \max_i z_i$) to prevent floating-point underflow over multi-year histories. States are canonically ordered by ascending equity volatility $\sqrt{(\Sigma_k)_{0,0}}$.
- **Out-of-Sample Online Hamilton Recursion ([`hamilton_online_filter.py`](./src/state_space_filtering/hamilton_online_filter.py)):** Backward smoothing $\gamma_{t\mid T} = \mathbb{P}(S_t \mid Y_{1:T})$ is **never** invoked out-of-sample. Instead, the filtered state posterior $\xi_{t\mid t} \in \Delta^2$ is updated strictly forward in time via Hamilton's (1989) filter:

$$\xi_{t\mid t-1} = A^\top \xi_{t-1\mid t-1}, \qquad \xi_{t\mid t} = \frac{\xi_{t\mid t-1} \odot \eta_t}{\mathbf{1}^\top (\xi_{t\mid t-1} \odot \eta_t)}$$

- **Causal 15-Day Persistence Smoother:** To prevent transient 1–2 day volatility spikes from triggering costly portfolio whipsaws under 10 bps transaction costs, portfolio weights at day $t$ are determined strictly from trailing posteriors $\{\xi_{\tau\mid\tau}\}_{\tau = \max(0, t-W)}^{t-1}$ ($W=15$).

---

### 2. Adaptive State-Space Kalman Cointegration & Ornstein–Uhlenbeck Dynamics

Static Engle–Granger OLS cointegration ($p_{2,t} = \alpha + h\, p_{1,t} + \epsilon_t$) suffers severe out-of-sample bias when structural macro shifts cause the equilibrium hedge ratio to drift. In [`kalman_ou_statarb.py`](./src/state_space_filtering/kalman_ou_statarb.py), we model the latent cointegration vector $\beta_t = [\alpha_t, h_t]^\top \in \mathbb{R}^2$ as a state-space random walk:

$$\beta_t = \beta_{t-1} + \omega_t, \quad \omega_t \sim \mathcal{N}(\mathbf{0}, Q), \qquad p_{2,t} = x_t^\top \beta_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, R), \quad x_t = \begin{bmatrix} 1 \\ p_{1,t} \end{bmatrix}$$

At each step $t$, the causal Kalman prediction and **Joseph-stabilized covariance update** (which guarantees symmetry and positive-definiteness of $P_{t\mid t}$ in finite precision) are:

$$\beta_{t\mid t-1} = \beta_{t-1\mid t-1}, \qquad P_{t\mid t-1} = P_{t-1\mid t-1} + Q$$

$$e_t = p_{2,t} - x_t^\top \beta_{t\mid t-1}, \qquad F_t = x_t^\top P_{t\mid t-1} x_t + R, \qquad K_t = \frac{P_{t\mid t-1} x_t}{F_t}$$

$$\beta_{t\mid t} = \beta_{t\mid t-1} + K_t e_t, \qquad P_{t\mid t} = (I_2 - K_t x_t^\top) P_{t\mid t-1} (I_2 - K_t x_t^\top)^\top + K_t R K_t^\top$$

The causal prediction residual $e_t$ is modeled as a discrete-time Ornstein–Uhlenbeck process $dS_t = \kappa(\theta - S_t)\,dt + \sigma_{\text{OU}}\,dW_t$ calibrated via exact AR(1) regression $S_t = c + \phi S_{t-1} + \xi_t$:

$$\kappa = -\frac{\ln \phi}{\Delta t}, \qquad \tau_{1/2} = \frac{\ln 2}{\kappa}, \qquad \sigma_{\text{eq}} = \sigma_\xi \sqrt{\frac{1}{1 - \phi^2}}$$

---

### 3. Cross-Sectional Factor Orthogonalization, Ledoit–Wolf Shrinkage & Convex Markowitz

In [`alpha_orthogonalization.py`](./src/highdim_portfolio/alpha_orthogonalization.py) and [`ledoit_wolf_cvxpy.py`](./src/highdim_portfolio/ledoit_wolf_cvxpy.py), given $N=54$ equities and $K=6$ collinear alpha signals $X_t \in \mathbb{R}^{N\times K}$:

1. **Gram–Schmidt / Loewdin Orthogonalization:** Centered predictors $\tilde{X}_t = X_t - \frac{1}{N}\mathbf{1}\mathbf{1}^\top X_t$ are sequentially projected onto the orthogonal complement of preceding factors:

$$u_k = \tilde{x}_k - \sum_{j=1}^{k-1} \frac{\langle \tilde{x}_k, z_j \rangle}{\langle z_j, z_j \rangle} z_j, \qquad z_k = \frac{\sqrt{N-1}}{\lVert u_k \rVert_2} u_k \implies \frac{1}{N-1} Z_t^\top Z_t = I_K, \quad \operatorname{VIF}_k = \left[(R_{Z_t}^\top R_{Z_t})^{-1}\right]_{kk} = 1.00$$

2. **Analytical Ledoit–Wolf (2004) Shrinkage:** On rolling high-dimensional windows ($N=54, T=90$), sample covariance $S = \frac{1}{T} Y^\top Y$ is shrunk toward the scaled identity target $F = \hat{\mu}_{\text{tr}} I_N$ ($\hat{\mu}_{\text{tr}} = \frac{1}{N}\operatorname{tr}(S)$):

$$\hat{\Sigma}_{\text{LW}} = (1 - \hat{\delta}^*)\, S + \hat{\delta}^*\, \hat{\mu}_{\text{tr}} I_N, \qquad \hat{\delta}^* = \operatorname{clip}\!\left(\frac{\hat{\beta}^2}{\hat{\delta}^2}, 0, 1\right), \qquad \hat{\delta}^2 = \frac{1}{N}\lVert S - F \rVert_F^2$$

$$\hat{\beta}^2 = \min\!\left(\hat{\delta}^2,\, \frac{1}{N T^2} \sum_{t=1}^T \left\lVert y_t y_t^\top - S \right\rVert_F^2\right)$$

3. **Dollar- and Beta-Neutral Convex Markowitz Program:**

$$\max_{w_t \in \mathbb{R}^N} \quad \hat{\mu}_t^\top w_t - \frac{\gamma}{2} w_t^\top \hat{\Sigma}_{\text{LW}} w_t - \kappa_{\text{tc}} \lVert w_t - w_{t-1} \rVert_1 \qquad \text{s.t.} \quad \mathbf{1}^\top w_t = 0, \quad \left\lvert \beta_{\text{mkt}}^\top w_t \right\rvert \le 0.04, \quad \lVert w_t \rVert_\infty \le w_{\max}$$

4. **Fama–French 5-Factor Attribution with Newey–West (1987) HAC Standard Errors:**

$$R_{p,t} - R_{f,t} = \alpha + \beta_{\text{MKT}}\text{MKT}_t + \beta_{\text{SMB}}\text{SMB}_t + \beta_{\text{HML}}\text{HML}_t + \beta_{\text{RMW}}\text{RMW}_t + \beta_{\text{CMA}}\text{CMA}_t + \varepsilon_t$$

$$\hat{V}_{\text{NW}} = \frac{T}{T - p} (X^\top X)^{-1} \hat{S}_{\text{HAC}} (X^\top X)^{-1}, \qquad \hat{S}_{\text{HAC}} = \hat{\Gamma}_0 + \sum_{\ell=1}^{L} \left(1 - \frac{\ell}{L+1}\right)\left(\hat{\Gamma}_\ell + \hat{\Gamma}_\ell^\top\right)$$

---

### 4. C++17 2D Spectral Cooley–Tukey FFT & Wiener Deconvolution

In [`src/cpp_spectral_fft/`](./src/cpp_spectral_fft/), for an $M \times M$ ($512 \times 512$) grid convolved with a symmetric $(2N+1)\times(2N+1)$ spatial box filter $h[m,n] = \frac{1}{(2N+1)^2}\mathbf{1}_{\{\lvert m\rvert \le N, \lvert n\rvert \le N\}}$:

- **2D Separable Radix-2 Cooley–Tukey Butterfly ([`cooley_tukey_2d_fft.cpp`](./src/cpp_spectral_fft/cooley_tukey_2d_fft.cpp)):** Decomposes the 2D DFT into row-wise 1D bit-reversal & butterfly stages, cache-blocked matrix transposition, and column-wise 1D FFT stages, reducing complexity from $\mathcal{O}(M^2 (2N+1)^2)$ to $\mathcal{O}(M^2 \log_2 M)$.
- **Dirichlet Zero-Crossings & Wiener Regularization ([`wiener_deconvolution.cpp`](./src/cpp_spectral_fft/wiener_deconvolution.cpp)):** The periodic box kernel transfer function is a real-valued 2D Dirichlet kernel:

$$H(\omega_1, \omega_2) = \frac{1}{(2N+1)^2} \frac{\sin\!\left(\omega_1(N + \tfrac{1}{2})\right)}{\sin(\omega_1 / 2)} \frac{\sin\!\left(\omega_2(N + \tfrac{1}{2})\right)}{\sin(\omega_2 / 2)}$$

Because $H(\omega_1, \omega_2)$ oscillates continuously through zero into negative sidelobes, naive additive damping $\hat{X}_\epsilon(\omega) = \frac{Y(\omega)}{H(\omega) + \epsilon}$ shifts singularities onto the contour $H(\omega_1, \omega_2) = -\epsilon$. **Wiener deconvolution** replaces linear shifting with a strictly positive quadratic denominator:

$$\hat{X}_{\text{Wiener}}(\omega_1, \omega_2) = \frac{H^*(\omega_1, \omega_2)}{\lvert H(\omega_1, \omega_2)\rvert^2 + \operatorname{NSR}} Y(\omega_1, \omega_2), \qquad \operatorname{NSR} = \frac{S_{\eta\eta}}{S_{xx}} > 0$$

| Kernel Half-Width $N$ | Box Filter Dimensions $(2N+1)\times(2N+1)$ | Direct Spatial Convolution (ms) | 2D Cooley–Tukey FFT Convolution (ms) | Empirical Speedup Factor |
| :---: | :---: | :---: | :---: | :---: |
| $N = 3$ | $7 \times 7$ | 11.7 ms | 11.6 ms | **1.0x** |
| $N = 7$ | $15 \times 15$ | 33.9 ms | 11.2 ms | **3.0x** |
| $N = 11$ | $23 \times 23$ | 70.1 ms | 11.1 ms | **6.3x** |
| $N = 15$ | $31 \times 31$ | 124.2 ms | 11.0 ms | **11.3x** |
| $N = 20$ | $41 \times 41$ | 212.1 ms | 11.0 ms | **19.3x** |
| $N = 24$ | $49 \times 49$ | 300.7 ms | 11.1 ms | **27.1x** |

---

## Repository Structure

```text
highdim-filtering-statarb/
├── README.md                                     # Mathematical documentation, figures, and benchmark tables
├── Technical_Note.pdf                            # 2-page 2-column academic summary paper (PDF)
├── CMakeLists.txt                                # C++17 CMake build configuration (-O3 -march=native)
├── Makefile                                      # One-command build, test, benchmark, and paper targets
├── requirements.txt                              # Python scientific computing dependencies
├── pytest.ini                                    # Pytest test discovery configuration
├── LICENSE                                       # MIT License
├── paper/
│   └── technical_note.tex                        # Complete 2-page 2-column LaTeX source
├── src/
│   ├── state_space_filtering/
│   │   ├── baum_welch_hmm.py                     # Log-space 3-state multivariate Gaussian Baum-Welch EM
│   │   ├── hamilton_online_filter.py             # Strictly causal online Hamilton filter & 15-day smoother
│   │   └── kalman_ou_statarb.py                  # State-space Kalman cointegration & AR(1) OU calibration
│   ├── highdim_portfolio/
│   │   ├── alpha_orthogonalization.py            # Gram-Schmidt/Loewdin orthogonalization, Ridge/Lasso/GBT/MLP, FF5 HAC
│   │   ├── ledoit_wolf_cvxpy.py                  # Analytical Ledoit-Wolf shrinkage & convex Markowitz solver
│   │   └── credit_threshold_opt.py               # IRLS logistic hazard model & Neyman-Pearson credit policy optimizer
│   └── cpp_spectral_fft/
│       ├── spectral_fft.hpp                      # C++17 header for 2D FFT, spatial/spectral box filter, and Wiener filter
│       ├── cooley_tukey_2d_fft.cpp               # In-place Radix-2 2D Cooley-Tukey FFT & cache-blocked transpose
│       ├── wiener_deconvolution.cpp              # Spectral inverse vs. Wiener deconvolution & CLI JSON benchmark binary
│       └── runner.py                             # Python bridge compiling and invoking the C++17 benchmark executable
├── scripts/
│   ├── reproduce_all_benchmarks.py               # Master script generating all 4 figures and results/benchmark_summary.json
│   └── generate_technical_note.py                # Generates Technical_Note.pdf from live benchmark results and figures
├── figures/
│   ├── fig1_regimeshift_hmm_posterior_states.png
│   ├── fig2_kalman_dynamic_hedge_ratio_tracking.png
│   ├── fig3_ledoit_wolf_efficient_frontier_and_ff5.png
│   └── fig4_cpp_fft_speedup_and_wiener_psnr.png
├── results/
│   └── benchmark_summary.json                    # Serialized empirical and numerical benchmark metrics
└── tests/
    ├── test_hamilton_causality.py                # Adversarial future-perturbation tests verifying zero lookahead (atol=1e-14)
    ├── test_beta_neutrality.py                   # Tests verifying VIF=1.0, SPD shrinkage, |beta|<=0.04, and credit PD ceiling
    └── test_cpp_spectral_fft.py                  # Tests verifying Parseval's theorem, FFT-IFFT roundtrip, and Wiener MSE cut
```

---

## Quickstart & One-Command Reproducibility

### 1. Build the C++17 Spectral FFT & Deconvolution Engine

```bash
make build
```

Compiles `src/cpp_spectral_fft/cooley_tukey_2d_fft.cpp` and `src/cpp_spectral_fft/wiener_deconvolution.cpp` with `g++ -std=c++17 -O3 -march=native -Wall -Wextra -pedantic` into `build/bin/spectral_benchmark`.

### 2. Run the Verification & Causality Test Suite

```bash
make test
```

Executes all 10 unit and adversarial causality tests across [`tests/test_hamilton_causality.py`](./tests/test_hamilton_causality.py), [`tests/test_beta_neutrality.py`](./tests/test_beta_neutrality.py), and [`tests/test_cpp_spectral_fft.py`](./tests/test_cpp_spectral_fft.py):
- **Zero-Lookahead Causality (`atol=1e-14`):** Perturbs future observations $Y_{t+1:T}$ with extreme counterfactual shocks and verifies that filtered posteriors $\xi_{1:t}$ and Kalman states $\beta_{1:t}$ remain identical to machine precision.
- **Portfolio Constraint Verification:** Confirms $\frac{1}{N-1} Z_t^\top Z_t = I_K$ ($\text{VIF} = 1.00$), positive-definiteness and Frobenius loss reduction of $\hat{\Sigma}_{\text{LW}}$, dollar neutrality ($\mathbf{1}^\top w_t = 0$), and market-beta neutrality ($\lvert\beta_{\text{mkt}}^\top w_t\rvert \le 0.04$).
- **Spectral Identity Verification:** Confirms Parseval energy conservation within $10^{-13}$, exact FFT $\to$ IFFT reconstruction, spatial-spectral circular convolution equivalence, and Wiener MSE reduction.

### 3. Reproduce All Benchmarks, Figures, and Technical Note PDF

```bash
make reproduce
make paper
```

---

## Author & License

**Yash Kothari**  
Department of Electrical Engineering & Department of Mathematics (Minor)  
Indian Institute of Technology Bombay (`25b1309@iitb.ac.in`)

Released under the [MIT License](./LICENSE).

