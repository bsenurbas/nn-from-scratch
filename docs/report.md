# Neural Network From Scratch — Report Notes

This project builds a minimal yet correct neural network framework using only NumPy, validating both learning and gradients. It evolves progressively toward a clean, reusable engineering structure.

## Project Levels

- **Level 1**: Core neural network foundations and validation  
- **Level 2**: Training strategies, optimization behavior, and engineering improvements

---

##  Features

- Binary classification (XOR) with backpropagation
- Gradient checking for analytic validation
- Multiclass classification with softmax
- Support for various training strategies: full-batch, mini-batch, SGD
- Momentum optimization
- Model save/load standard using `.npz` and `.json`
- Unified inference API

---

##  Level 1 — Core Neural Network Foundations

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
x1 x2 | prob    | pred
----------------------
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
x1 x2 | prob    | pred
----------------------
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

---

### 2. Gradient Check (Backprop Validation)

Numerical gradient checking validates correctness.

Example:

```
W1[0,0] — Numeric: -0.0076243557 | Analytic: -0.0076243557
Abs diff: 2.45e-12 | Rel diff: 1.60e-10
```

✔️ Confirms accurate backpropagation.

---

### 3. Multiclass Extension (Softmax)

- Output Activation: Softmax  
- Loss: Cross-Entropy  
- Accuracy: Multiclass  

**Training Example (3-class toy data):**
```
Epoch     1 | Loss: 0.977540 | Acc: 0.40
Epoch   900 | Loss: 0.019156 | Acc: 1.00
Epoch  3000 | Loss: 0.007524 | Acc: 1.00
```

✔️ Model reload preserves accuracy.

---

###  Level 1 Summary

- XOR task with BCE loss and sigmoid activations  
- Gradient checking implemented  
- Model save/load verified  
- Softmax classifier with perfect accuracy

---

##  Level 2 — Training Dynamics and Engineering

### 4. Batch Training Strategies

Support added for:

- Full-Batch Gradient Descent  
- Mini-Batch Gradient Descent  
- Stochastic Gradient Descent (SGD)

Configurable via `batch_size`.

---

### 5. Batch Size Comparison

**Setup:** `2 → 3 → 1`, BCE loss, 3000 epochs

| Method      | Batch Size | Final Loss | Accuracy |
|-------------|------------|------------|----------|
| Full-Batch  | 4          | 0.5112     | 0.75     |
| Mini-Batch  | 2          | 0.0542     | 1.00     |
| SGD         | 1          | 0.0135     | 1.00     |

✔️ Mini-batch and SGD converge faster and better.

---

### 6. Momentum-Based Optimization

Momentum improves convergence:

- GD + momentum  
- Mini-batch + momentum (best overall)  
- SGD + momentum (needs tuning)  

---

### 7. Model I/O Standard and API

#### Save / Load Format

```
models/
└── <model_name>/
    ├── weights.npz
    └── config.json
```

- `weights.npz`: learned parameters  
- `config.json`: architecture config  

#### Inference API

```python
predict_proba(X, task)
predict(X, task)
evaluate(X, y, task)
```

Legacy methods retained but deprecated in use.

✔️ Validated for both XOR and softmax models.

---

###  Level 2 Summary

- Modular, clean code structure  
- Unified prediction API  
- Batch training with momentum  
- Reloadable model structure  
- Verified correctness post-serialization  

This phase transitions the codebase into a mini-framework with reusable components and clean abstraction layers.

## Level 3 — Real Dataset + Testing

### 1) Pytest Foundation

A minimal `pytest` suite was added to validate core correctness:

- Forward output shapes (binary and multiclass)
- Softmax row sums equal 1
- Loss functions return finite values
- Score/Evaluate API accepts common label formats

**Run:**

```bash
python -m pytest -q
```

---

### 2) Iris Dataset Experiment (Real Data)

A real dataset experiment was added using a local CSV file:

- **Dataset**: `data/iris.csv` (4 features, 3 classes)
- Train/Test split with a fixed seed
- Standardization using train statistics only
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

- the multiclass pipeline works on a real dataset,  
- the API supports reproducible evaluation,  
- and model IO produces identical inference after reload.

---
