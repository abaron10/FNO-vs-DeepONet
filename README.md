# Benchmarks of Neural Operators: **FNO** vs **DeepONet**

> A standardized and visual benchmarking platform to evaluate accuracy, efficiency, and reproducibility of **neural operators** on the **Darcy flow (64×64)** benchmark.  

![status](https://img.shields.io/badge/status-active-success) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

---

## Novelty of this project

This repository is not just a collection of scripts: it is a **benchmarking platform** built with **software engineering best practices** to compare neural operators in a **fair, repeatable, and visual** way.

Key contributions:

- **Full standardization of the experimental cycle**
    - Same *dataset loader*, *normalization*, *metrics*, and *seeds* for all models.
    - Unified API (`BaseOperator`) → easily add new operators without breaking pipelines.

- **Consistent metrics and reporting**
    - MSE, MAE, **Rel-L²**, and **Accuracy = 100·(1−Rel-L²)**.
    - Results stored in CSV/JSON + scripts for plots and paper-ready tables.

- **Reproducible and visual pipeline**
    - Declarative configs (YAML/CLI), seed logging, automatic figure generation (training/validation curves, *time vs accuracy*, *model efficiency*, *relative L²*, and *compact prediction rows*).

---


## Models included

- **FNO (Fourier Neural Operator)**
    - Truncated spectral convolution (modes \(k_{\max}\)), 1×1 projection, **GELU** activation, *weight decay*, *StepLR*.
    - Variants: *Standard*, *Smaller (shared weights)*, *Ensemble*, *Enhanced Smaller*, *Optimized Target*.

- **DeepONet (branch–trunk MLP)**
    - **Branch (MLP)** processes sensors of \(k(x)\).
    - **Trunk (MLP)** encodes coordinates \((x,y)\).
    - Dot product \( \langle b, t \rangle + b_0 \).
    - Sensor strategies: *random*, *uniform*, *chebyshev*, *adaptive*.
    - Advanced variants: **MSFF** (multi-scale Fourier features), **SWA**, optional **Sobolev loss**.


## Installation

```bash
# 1) Clone
git clone https://github.com/<your-username>/FNO-vs-DeepONet.git
cd FNO-vs-DeepONet

# 2) Create environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

# 3) Dependencies
pip install -U pip
pip install -r requirements.txt

