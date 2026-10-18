#include "spectral_fft.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spectral {

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;

bool is_power_of_two(std::size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

void fft1d_strided(Complex* a, std::size_t n, bool invert) {
    // Bit-reversal permutation
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    // Precompute unit roots per butterfly stage
    std::vector<Complex> twiddles(n / 2);
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::size_t half = len >> 1;
        const double angle_step = 2.0 * kPi / static_cast<double>(len) * (invert ? 1.0 : -1.0);
        for (std::size_t j = 0; j < half; ++j) {
            const double theta = angle_step * static_cast<double>(j);
            twiddles[j] = Complex(std::cos(theta), std::sin(theta));
        }

        for (std::size_t i = 0; i < n; i += len) {
            for (std::size_t j = 0; j < half; ++j) {
                const Complex u = a[i + j];
                const Complex v = a[i + j + half] * twiddles[j];
                a[i + j] = u + v;
                a[i + j + half] = u - v;
            }
        }
    }

    if (invert) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] *= inv_n;
        }
    }
}
}  // namespace

void fft1d(std::vector<Complex>& a, bool invert) {
    const std::size_t n = a.size();
    if (!is_power_of_two(n)) {
        throw std::invalid_argument("fft1d requires power-of-two length.");
    }
    fft1d_strided(a.data(), n, invert);
}

void fft2d(std::vector<Complex>& data, std::size_t height, std::size_t width, bool invert) {
    if (data.size() != height * width) {
        throw std::invalid_argument("fft2d buffer size mismatch.");
    }

    // Transform rows in-place
    for (std::size_t r = 0; r < height; ++r) {
        fft1d_strided(data.data() + r * width, width, invert);
    }

    // In-place transpose for square power-of-two grids, fallback buffer otherwise
    if (height == width) {
        for (std::size_t r = 0; r < height; ++r) {
            for (std::size_t c = r + 1; c < width; ++c) {
                std::swap(data[r * width + c], data[c * width + r]);
            }
        }
        for (std::size_t c = 0; c < width; ++c) {
            fft1d_strided(data.data() + c * height, height, invert);
        }
        for (std::size_t r = 0; r < height; ++r) {
            for (std::size_t c = r + 1; c < width; ++c) {
                std::swap(data[r * width + c], data[c * width + r]);
            }
        }
    } else {
        std::vector<Complex> transposed(width * height);
        for (std::size_t r = 0; r < height; ++r) {
            for (std::size_t c = 0; c < width; ++c) {
                transposed[c * height + r] = data[r * width + c];
            }
        }
        for (std::size_t c = 0; c < width; ++c) {
            fft1d_strided(transposed.data() + c * height, height, invert);
        }
        for (std::size_t r = 0; r < height; ++r) {
            for (std::size_t c = 0; c < width; ++c) {
                data[r * width + c] = transposed[c * height + r];
            }
        }
    }
}

std::vector<Complex> build_periodic_box_kernel(
    std::size_t height, std::size_t width, int half_width_N
) {
    const int ksize = 2 * half_width_N + 1;
    const double weight = 1.0 / static_cast<double>(ksize * ksize);
    std::vector<Complex> kernel(height * width, Complex(0.0, 0.0));

    for (int dr = -half_width_N; dr <= half_width_N; ++dr) {
        const std::size_t r = static_cast<std::size_t>(
            (dr + static_cast<int>(height)) % static_cast<int>(height)
        );
        for (int dc = -half_width_N; dc <= half_width_N; ++dc) {
            const std::size_t c = static_cast<std::size_t>(
                (dc + static_cast<int>(width)) % static_cast<int>(width)
            );
            kernel[r * width + c] = Complex(weight, 0.0);
        }
    }
    return kernel;
}

std::vector<double> spatial_box_convolve(
    const std::vector<double>& img, std::size_t height, std::size_t width, int half_width_N
) {
    const int ksize = 2 * half_width_N + 1;
    const double weight = 1.0 / static_cast<double>(ksize * ksize);
    const int H = static_cast<int>(height);
    const int W = static_cast<int>(width);
    std::vector<double> out(height * width, 0.0);

    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            double acc = 0.0;
            for (int dr = -half_width_N; dr <= half_width_N; ++dr) {
                int rr = (r + dr) % H;
                if (rr < 0) rr += H;
                const std::size_t row_off = static_cast<std::size_t>(rr) * width;

                for (int dc = -half_width_N; dc <= half_width_N; ++dc) {
                    int cc = (c + dc) % W;
                    if (cc < 0) cc += W;
                    acc += img[row_off + static_cast<std::size_t>(cc)];
                }
            }
            out[static_cast<std::size_t>(r) * width + static_cast<std::size_t>(c)] = acc * weight;
        }
    }
    return out;
}

std::vector<double> spectral_box_convolve(
    const std::vector<double>& img, std::size_t height, std::size_t width, int half_width_N
) {
    const std::size_t total = height * width;
    std::vector<Complex> X(total);
    for (std::size_t i = 0; i < total; ++i) {
        X[i] = Complex(img[i], 0.0);
    }
    std::vector<Complex> H = build_periodic_box_kernel(height, width, half_width_N);

    fft2d(X, height, width, false);
    fft2d(H, height, width, false);

    for (std::size_t i = 0; i < total; ++i) {
        X[i] *= H[i];
    }
    fft2d(X, height, width, true);

    std::vector<double> out(total);
    for (std::size_t i = 0; i < total; ++i) {
        out[i] = X[i].real();
    }
    return out;
}

}  // namespace spectral

