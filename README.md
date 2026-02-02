# Neural Network From Scratch (NumPy)

A fully functional neural network framework implemented **from scratch using only NumPy**.

This project is designed as an **engineering-grade learning and portfolio repository**, focusing on correctness, reproducibility, and clean API design rather than raw performance.

No deep learning frameworks (PyTorch, TensorFlow, JAX) are used.

---

## Project Scope

This repository implements a minimal but complete neural network stack:

- Forward and backward propagation
- Binary and multiclass classification
- Multiple training regimes (GD, mini-batch, SGD, momentum)
- Model persistence with reproducibility guarantees
- Automated testing and validation

The goal is to demonstrate a deep understanding of **how neural networks work internally**, not to compete with production frameworks.

---

## Implemented Features

### Core Model
- Fully connected network (1 hidden layer)
- Sigmoid activation for binary classification
- Softmax activation for multiclass classification
- Binary Cross-Entropy and Cross-Entropy losses
- Clean separation of activations, losses, metrics, and utilities

### Training
- Full-batch Gradient Descent
- Mini-batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Momentum support
- Deterministic training via explicit seeding

### Evaluation
- Binary and multiclass accuracy
- Unified `predict`, `predict_proba`, `score`, and `evaluate` API
- Gradient checking (numeric vs analytic)

### Persistence
- Save/load model weights
- Directory-based model versioning
- Save/load validation included in experiments

### Testing
- Pytest-based test suite
- Shape checks
- Softmax probability validation
- Loss finiteness checks
- Deterministic training behavior

---

## Experiments

### XOR (Binary Classification)
- Demonstrates non-linear separability
- Verifies forward/backward correctness
- Includes save/load validation

```bash
python -m experiments.xor
```

### Softmax Toy Dataset (Multiclass)
- 3-class synthetic clustering problem
- Softmax + cross-entropy training
- Save/load consistency check

```bash
python -m experiments.softmax_toy
```

### Iris Dataset (Real Dataset)
- Classic 3-class real-world dataset
- Train / test split
- Loaded model evaluation

```bash
python -m experiments.iris
```

### Gradient Check
- Numeric vs analytic gradient comparison
- Confirms correctness of backpropagation

```bash
python -m experiments.grad_check
```

---

## Project Structure

```text
nn-from-scratch/
├── core/               # Core neural network components
│   ├── activations.py
│   ├── losses.py
│   ├── metrics.py
│   ├── utils.py
│   └── network.py
├── experiments/        # Reproducible experiments
├── data/               # Real datasets (Iris)
├── models/             # Saved models and configs
├── tests/              # Automated tests (pytest)
├── docs/               # Reports and experiment notes
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

---

## Running Tests

```bash
python -m pytest -q
```

All tests are expected to pass.

---

## Design Philosophy

- Explicit over implicit
- Determinism over randomness
- Correctness over performance
- Clear APIs over convenience hacks

This repository is intentionally small, readable, and inspectable.

---

## Status

This project is **feature-complete** for its intended scope.

Future extensions (CNNs, deeper architectures, GPU acceleration) are intentionally out of scope.

---

## Author

Built as a deep learning fundamentals and engineering portfolio project.
