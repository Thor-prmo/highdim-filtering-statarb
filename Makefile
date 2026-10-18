CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -Wall -Wextra -DNDEBUG
PYTHON ?= python3
PYTEST ?= PYTHONPATH=. python3 -m pytest

CPP_SRC = src/cpp_spectral_fft/cooley_tukey_2d_fft.cpp src/cpp_spectral_fft/wiener_deconvolution.cpp
CPP_BIN = build/bin/spectral_benchmark

.PHONY: all build test reproduce plots paper clean

all: build reproduce

build: $(CPP_BIN)

$(CPP_BIN): $(CPP_SRC) src/cpp_spectral_fft/spectral_fft.hpp
	@mkdir -p build/bin
	$(CXX) $(CXXFLAGS) -Isrc/cpp_spectral_fft $(CPP_SRC) -o $(CPP_BIN)

test: build
	$(PYTEST) tests/

reproduce: build
	MPLCONFIGDIR=/tmp PYTHONPATH=. $(PYTHON) scripts/reproduce_all_benchmarks.py

plots: reproduce

paper: reproduce
	MPLCONFIGDIR=/tmp PYTHONPATH=. $(PYTHON) scripts/generate_technical_note.py

clean:
	rm -rf build/ .pytest_cache/
	find . -name "__pycache__" -type d -exec rm -rf {} +
