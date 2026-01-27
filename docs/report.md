# 📝 Report Notes (Level 1)

This document summarizes the implemented experiments and provides initial comparisons between different training settings.

---

## 1️⃣ XOR Experiment (Binary Classification)

### 🔧 Task
Learn the XOR function:

| x1 | x2 | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

### 🧠 Model Architecture

- **Input:** 2 neurons  
- **Hidden:** 3 neurons  
- **Output:** 1 neuron  
- **Activation:** Sigmoid  
- **Loss:** Binary Cross-Entropy (BCE)

### 📈 Results

The network successfully learns the XOR pattern:

- Loss decreases steadily
- Accuracy reaches **1.00**
- Predictions align with expected outputs

**Example Predictions After Training:**
```
0 0 -> 0.01 (→ 0)
0 1 -> 0.98 (→ 1)
1 0 -> 0.98 (→ 1)
1 1 -> 0.01 (→ 0)
```

### 💾 Save/Load Validation

Model persistence is verified:

- **Max absolute difference after reloading:** `0.0`

---

## 2️⃣ Softmax Toy Experiment (Multiclass Classification)

### 🔧 Task

Train on a synthetic dataset with 3 distinct clusters:

- Class 0  
- Class 1  
- Class 2  

### 🧠 Model Architecture

- **Input:** 2 neurons  
- **Hidden:** 8 neurons  
- **Output:** 3 neurons  
- **Activation:** Softmax  
- **Loss:** Cross-Entropy

### 📈 Results

The model converges quickly and performs well:

- Loss drops close to **0**
- Accuracy reaches **1.00**

**Example Training Log:**
```
Epoch 300 | Loss: 0.035 | Acc: 0.99  
Epoch 900 | Loss: 0.019 | Acc: 1.00  
Final train acc: 1.000
```

---

## 3️⃣ Full-Batch vs Mini-Batch Training

### ⚖️ Comparison Overview

#### Full Batch
- Processes all data in one update
- Produces stable gradients
- Slower convergence

#### Mini Batch
- Processes small batches (e.g., 32 samples)
- Faster convergence
- Slightly noisier gradients

### 🧪 Observations
- Mini-batch training reached high accuracy faster
- Training remained stable with batch shuffling

---

## ⏭️ Next Steps (Level 2)

Planned improvements and experiments:

- [ ] Adam Optimizer
- [ ] Xavier Initialization
- [ ] Decision Boundary Visualization
- [ ] Unit Testing Framework

---
