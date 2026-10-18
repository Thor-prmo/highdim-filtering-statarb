"""Python bridge to compile and execute the C++17 2D Cooley-Tukey FFT & Wiener deconvolution binary."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_DIR = Path(__file__).resolve().parent
BIN_PATH = REPO_ROOT / "build" / "bin" / "spectral_benchmark"


def build_cpp_spectral_binary(force_rebuild: bool = False) -> Path:
    """Compile the C++17 spectral FFT benchmark binary with -O3 -march=native."""
    if BIN_PATH.exists() and not force_rebuild:
        return BIN_PATH

    BIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++",
        "-std=c++17",
        "-O3",
        "-march=native",
        "-Wall",
        "-Wextra",
        "-DNDEBUG",
        f"-I{CPP_DIR}",
        str(CPP_DIR / "cooley_tukey_2d_fft.cpp"),
        str(CPP_DIR / "wiener_deconvolution.cpp"),
        "-o",
        str(BIN_PATH),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return BIN_PATH


def run_cpp_spectral_benchmarks(force_rebuild: bool = False) -> Dict[str, Any]:
    """Run the compiled C++17 spectral engine and return parsed benchmark metrics."""
    binary = build_cpp_spectral_binary(force_rebuild=force_rebuild)
    proc = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return json.loads(proc.stdout)

