#ifndef HIGHDIM_SPECTRAL_FFT_HPP
#define HIGHDIM_SPECTRAL_FFT_HPP

#include <complex>
#include <cstddef>
#include <vector>

namespace spectral {

using Complex = std::complex<double>;

// In-place 1D Radix-2 Cooley-Tukey FFT (invert = false for forward, true for inverse)
void fft1d(std::vector<Complex>& a, bool invert);

// In-place 2D Radix-2 Cooley-Tukey FFT on row-major (height x width) complex buffer
void fft2d(std::vector<Complex>& data, std::size_t height, std::size_t width, bool invert);

// Construct wrap-around zero-phase (2N+1)x(2N+1) symmetric box kernel on (height x width) grid
std::vector<Complex> build_periodic_box_kernel(
    std::size_t height, std::size_t width, int half_width_N
);

// Direct spatial 2D periodic convolution with symmetric (2N+1)x(2N+1) box filter
std::vector<double> spatial_box_convolve(
    const std::vector<double>& img, std::size_t height, std::size_t width, int half_width_N
);

// Spectral 2D FFT convolution using precomputed or on-the-fly transfer function H(w1, w2)
std::vector<double> spectral_box_convolve(
    const std::vector<double>& img, std::size_t height, std::size_t width, int half_width_N
);

// Regularized frequency-domain inverse filter: X_hat(w) = Y(w) / (H(w) + epsilon)
std::vector<double> damped_inverse_filter(
    const std::vector<double>& degraded,
    const std::vector<Complex>& H_freq,
    std::size_t height,
    std::size_t width,
    double epsilon
);

// Wiener frequency-domain deconvolution: X_hat(w) = H*(w) Y(w) / (|H(w)|^2 + nsr)
std::vector<double> wiener_deconvolve(
    const std::vector<double>& degraded,
    const std::vector<Complex>& H_freq,
    std::size_t height,
    std::size_t width,
    double nsr
);

// Image quality metrics (assuming [0, 255] dynamic range)
double compute_mse(const std::vector<double>& ref, const std::vector<double>& est);
double compute_psnr(const std::vector<double>& ref, const std::vector<double>& est);
double compute_ssim(
    const std::vector<double>& ref,
    const std::vector<double>& est,
    std::size_t height,
    std::size_t width
);

}  // namespace spectral

#endif  // HIGHDIM_SPECTRAL_FFT_HPP

