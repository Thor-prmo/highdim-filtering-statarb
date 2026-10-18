"""Verification tests for the compiled C++17 2D Cooley-Tukey FFT & Wiener deconvolution engine."""

from src.cpp_spectral_fft import run_cpp_spectral_benchmarks


def test_cpp_2d_fft_parseval_and_invertibility() -> None:
    """2D Cooley-Tukey FFT must satisfy Parseval's theorem and exact IFFT roundtrip."""
    res = run_cpp_spectral_benchmarks(force_rebuild=False)
    assert res["grid_size"] == [512, 512]
    assert res["parseval_rel_error"] < 1e-11
    assert res["max_roundtrip_error"] < 1e-10
    assert res["max_spatial_spectral_diff"] < 1e-9


def test_cpp_fft_speedup_and_wiener_deconvolution() -> None:
    """2D FFT must achieve >10x speedup on large box kernels and Wiener must beat inverse filter."""
    res = run_cpp_spectral_benchmarks(force_rebuild=False)
    max_speedup = max(res["speedup_x"])
    assert max_speedup > 10.0
    assert res["inverse_filter_psnr_gain_db"] > 6.0
    assert res["wiener_mse_reduction_vs_inverse_pct"] > 30.0
    assert res["wiener_psnr_db"] > res["inverse_filter_psnr_db"]

