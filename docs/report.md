# Neural Network From Scratch — Report Notes

This report documents the design, implementation, and validation of a neural network framework built entirely from scratch using NumPy.

The primary objective of the project is not raw performance, but **correctness, reproducibility, and conceptual clarity**—demonstrating a full understanding of neural network internals such as forward propagation, backpropagation, optimization dynamics, and evaluation.

The project evolves progressively through clearly defined levels, each adding capability only after correctness is validated.

---

## Project Levels

- **Level 1**: Core neural network foundations and mathematical validation  
- **Level 2**: Training strategies, optimization behavior, and engineering improvements  
- **Level 3**: Real dataset usage and automated testing

---

## Features

The framework was developed incrementally, with each feature introduced only after correctness was verified through controlled experiments.

- Binary classification (XOR) with backpropagation  
- Gradient checking for analytic validation  
- Multiclass classification with softmax  
- Support for various training strategies: full-batch, mini-batch, SGD  
- Momentum optimization  
- Model save/load standard using `.npz` and `.json`  
- Unified inference and evaluation API  
- Automated testing with pytest  

---

## Level 1 — Core Neural Network Foundations

### 1. XOR Experiment (Binary Classification)

#### Task

Learn the XOR function:

| x1 | x2 | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

> XOR is not linearly separable; a hidden layer is required.

#### Model Architecture

- Input: 2 neurons  
- Hidden: 3 neurons (Sigmoid)  
- Output: 1 neuron (Sigmoid)  
- Loss: Binary Cross-Entropy  
- Optimizer: Gradient Descent  

#### Training Snapshot

**Before Training:**

```
## x1 x2 | prob    | pred
0  0  | 0.50777 |  1
0  1  | 0.44197 |  0
1  0  | 0.58439 |  1
1  1  | 0.52100 |  1
```

**Loss & Accuracy Progression:**

```
Epoch     1 | Loss: 0.700229 | Acc: 0.25  
Epoch  1500 | Loss: 0.371707 | Acc: 1.00  
Epoch  5000 | Loss: 0.013339 | Acc: 1.00  
```

**After Training:**

```
## x1 x2 | prob    | pred
0  0  | 0.01302 |  0
0  1  | 0.98319 |  1
1  0  | 0.98714 |  1
1  1  | 0.01027 |  0
```

#### Save / Load Validation

```
Loaded model output matches trained output.  
Max abs diff: 0.0
```

This confirms correct serialization and deterministic inference.

---

### 2. Gradient Check (Backprop Validation)

Numerical gradient checking was implemented to validate the analytic gradients produced by backpropagation.

Example:

```
W1[0,0] — Numeric: -0.0076243557 | Analytic: -0.0076243557  
Abs diff: 2.45e-12 | Rel diff: 1.60e-10
```

✔️ This confirms the correctness of the backward pass implementation.

---

### 3. Multiclass Extension (Softmax)

The network was extended to support multiclass classification:

- Output Activation: Softmax  
- Loss: Cross-Entropy  
- Accuracy: Multiclass  

**Training Example (3-class toy data):**

```
Epoch     1 | Loss: 0.977540 | Acc: 0.40  
Epoch   900 | Loss: 0.019156 | Acc: 1.00  
Epoch  3000 | Loss: 0.007524 | Acc: 1.00  
```

✔️ Model reload preserves accuracy after serialization.

---

### Level 1 Summary

- XOR task with BCE loss and sigmoid activations  
- Gradient checking implemented and validated  
- Model save/load correctness verified  
- Softmax-based multiclass classifier with perfect accuracy  

---

## Level 2 — Training Dynamics and Engineering

After validating the mathematical correctness of the network in Level 1, the focus shifted from *“can it learn?”* to *“how does it learn under different training dynamics and engineering constraints?”*

---

### 4. Batch Training Strategies

Support was added for:

- Full-Batch Gradient Descent  
- Mini-Batch Gradient Descent  
- Stochastic Gradient Descent (SGD)  

All controlled via the `batch_size` parameter.

---

### 5. Batch Size Comparison

**Setup:** `2 → 3 → 1`, BCE loss, 3000 epochs

| Method      | Batch Size | Final Loss | Accuracy |
|-------------|------------|------------|----------|
| Full-Batch  | 4          | 0.5112     | 0.75     |
| Mini-Batch  | 2          | 0.0542     | 1.00     |
| SGD         | 1          | 0.0135     | 1.00     |

These results highlight that while full-batch training may stagnate on small datasets, mini-batch and stochastic updates introduce beneficial gradient noise that improves convergence.

---

### 6. Momentum-Based Optimization

Momentum was introduced to stabilize and accelerate training:

- GD + momentum  
- Mini-batch + momentum (best overall performance)  
- SGD + momentum (requires careful tuning)  

Momentum consistently improved convergence speed and final loss.

---

### 7. Model I/O Standard and Unified API

#### Save / Load Format

```
models/
└── <model_name>/
    ├── weights.npz
    └── config.json
```

- `weights.npz`: learned parameters  
- `config.json`: architecture and training configuration  

#### Inference API

```python
predict_proba(X, task)
predict(X, task)
evaluate(X, y, task)
```

Legacy methods are retained for backward compatibility but are not used in experiments.

✔️ The API was validated across binary and multiclass tasks.

---

### Level 2 Summary

- Modular and maintainable code structure  
- Unified inference and evaluation API  
- Batch training with momentum support  
- Reproducible model persistence  
- Verified correctness after reload  

This phase transitions the codebase into a small but coherent neural network framework.

---

## Level 3 — Real Dataset and Testing

Level 3 transitions the project from controlled toy experiments to a realistic machine learning workflow, combining real data, reproducible evaluation, and automated testing.

---

### 1. Pytest Foundation

A pytest-based test suite was added to validate core behavior:

- Forward output shapes (binary and multiclass)  
- Softmax row sums equal 1  
- Loss functions return finite values  
- Score and evaluate APIs accept common label formats  

**Run:**

```bash
python -m pytest -q
```

All tests pass consistently.

---

### 2. Iris Dataset Experiment (Real Data)

A real-world dataset experiment was added using a local CSV file:

- **Dataset**: `data/iris.csv` (4 features, 3 classes)  
- Train/test split with a fixed seed  
- Standardization using training statistics only  
- **Model**: input=4 → hidden=8 → output=3  
- **Optimizer**: Gradient Descent with momentum  
- Save/load validation via `save_dir` / `load_dir`  

**Run:**

```bash
python -m experiments.iris
```

**Example output:**

```
Train acc: 1.000  
Test  acc: 1.000  
Loaded model test acc: 1.000  
```

This confirms that:

- the multiclass pipeline generalizes to real data,  
- the API supports reproducible evaluation,  
- and model I/O produces identical inference after reload.

---

## Final Remarks

This project intentionally stops at a compact, fully verified scope.

Rather than expanding indefinitely, the framework was stabilized once:

- learning correctness was validated,  
- training dynamics were understood,  
- model persistence was proven,  
- and behavior was covered by automated tests.  

This deliberate limitation ensures that the codebase remains readable, inspectable, and suitable as a reference implementation and engineering portfolio artifact.
