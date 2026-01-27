# 🧠 Neural Network From Scratch (NumPy)

This repository contains a simple feedforward neural network implemented **from scratch** using only **NumPy**. The goal is to understand the training pipeline's mathematics without relying on high-level deep learning frameworks like PyTorch or TensorFlow.

---

## 🚀 Features

- **Architecture:** Fully connected neural network (1 hidden layer).
- **Math:** Forward propagation and manual backpropagation.
- **Activations:** Sigmoid (binary) and Softmax (multiclass).
- **Loss Functions:** Binary Cross-Entropy (BCE) and Categorical Cross-Entropy.
- **Optimization:** Mini-batch gradient descent with optional shuffling.
- **Utility:** Model save/load support and accuracy tracking.
- **Validation:** Gradient checking (comparing numeric vs. analytic gradients).

---

## 📂 Project Structure

```text
nn-from-scratch/
├── core/
│   └── network.py        # Main neural network implementation
├── experiments/
│   ├── xor.py            # XOR binary classification experiment
│   ├── softmax_toy.py    # 3-class softmax toy dataset experiment
│   └── grad_check.py     # Gradient check validation
├── docs/
│   └── report.md         # Comparison notes and learning report
└── requirements.txt      # Minimal dependency list
```

---

## 🛠️ Setup

1. **Create a virtual environment:**
```bash
python -m venv .venv
```

2. **Activate it:**

- **Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

- **Linux/macOS:**
```bash
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🧪 Running Experiments

### 1. XOR (Binary Classification)

Trains the network to solve the non-linear XOR problem.

```bash
python -m experiments.xor
```

- **Expected:** Loss decreases; Accuracy reaches 1.00.

### 2. Softmax Toy Dataset (Multiclass)

Trains on a 3-class synthetic dataset.

```bash
python -m experiments.softmax_toy
```

- **Expected:** Cross-entropy loss drops; Training accuracy approaches 1.00.

### 3. Gradient Check

Verifies the correctness of the backpropagation implementation.

```bash
python -m experiments.grad_check
```

- **Expected:** Numeric and analytic gradients match (low relative difference).

---

## 🎯 Learning Goals

This project provides hands-on experience with:

- The internal mechanics of **backpropagation**
- Differences between **Sigmoid + BCE** and **Softmax + Cross-Entropy**
- **Mini-batch** vs. full-batch training dynamics
- Using **gradient checking** to debug neural networks

---

## 🗺️ Roadmap

- [ ] Implement Adam Optimizer
- [ ] Xavier/He weight initialization
- [ ] Add regularization techniques (L2, Dropout)
- [ ] Visualize decision boundaries using Matplotlib
- [ ] Benchmark on a subset of MNIST

---

## 👤 Author

Built as an educational deep learning project to master the fundamentals.

---
