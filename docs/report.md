# Report Notes (Level 1)

This report summarizes what has been implemented so far in **Level 1** of the project and documents the core experiments.

The focus of this stage was building a minimal neural network from scratch, validating learning behavior, and extending the model from XOR to multiclass softmax training.

---

## 1) XOR Experiment (Binary Classification)

### Task

Learn the XOR function:

| x1 | x2 | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

XOR cannot be solved with a single linear layer, so a hidden layer is required.

---

### Model

- Input layer: 2 neurons  
- Hidden layer: 3 neurons  
- Output layer: 1 neuron  
- Hidden activation: Sigmoid  
- Output activation: Sigmoid  
- Loss: Binary Cross-Entropy (BCE)  
- Optimizer: Gradient Descent  

---

### Training Output

Before training, predictions are random:

```
Before training
x1 x2 | prob    | pred
----------------------

0  0 | 0.615216 |  1
0  1 | 0.647575 |  1
1  0 | 0.620719 |  1
1  1 | 0.650983 |  1
```

Loss decreases steadily and the model reaches full accuracy:

```
Epoch     1 | Loss: 0.752723 | Acc: 0.50
Epoch  2000 | Loss: 0.283365 | Acc: 1.00
Epoch  5000 | Loss: 0.017335 | Acc: 1.00
```

After training, predictions match XOR correctly:

```
After training
x1 x2 | prob    | pred
----------------------

0  0 | 0.019250 |  0
0  1 | 0.984142 |  1
1  0 | 0.984112 |  1
1  1 | 0.017702 |  0
```

---

### Save / Load Check

The trained model was saved and loaded back successfully:

```
Loaded model output
x1 x2 | prob    | pred
----------------------

0  0 | 0.019250 |  0
0  1 | 0.984142 |  1
1  0 | 0.984112 |  1
1  1 | 0.017702 |  0

Max abs diff (after vs loaded): 0.0
```

This confirms that serialization works correctly.

---

## 2) Gradient Check (Backprop Validation)

To confirm the correctness of the analytic gradients, a numeric gradient check was implemented.

Example check on parameter `W1[0,0]`:

```
Gradient check for W1[0,0]
Numeric  : 0.0067181729
Analytic : 0.0067181729
Abs diff : 2.4281297459e-12
Rel diff : 1.8071354964e-10
```

The extremely small difference confirms the backward pass is correct.

---

## 3) Mini-Batch Training Support

The training loop was extended to support:

- Full-batch Gradient Descent
- Mini-batch Gradient Descent
- Stochastic Gradient Descent (SGD)

Batch size is now configurable through `batch_size`.

---

## 4) Batch Strategy Comparison (GD vs Mini-batch vs SGD)

A comparison experiment was added to observe convergence differences.

---

### Full-Batch Gradient Descent (batch_size=4)

```
Epoch  3000 | Loss: 0.511175 | Acc: 0.75
```

Final predictions:

```
## x1 x2 | prob    | pred | true

0  0 | 0.197559 |  0   |  0
0  1 | 0.656794 |  1   |  1
1  0 | 0.590402 |  1   |  1
1  1 | 0.583839 |  1   |  0
```

This run converged slower and failed to fully solve XOR.

---

### Mini-Batch Training (batch_size=2)

```
Epoch  3000 | Loss: 0.054176 | Acc: 1.00
```

Final predictions:

```
## x1 x2 | prob    | pred | true

0  0 | 0.052257 |  0   |  0
0  1 | 0.950494 |  1   |  1
1  0 | 0.950836 |  1   |  1
1  1 | 0.059566 |  0   |  0
```

Mini-batching improved convergence speed significantly.

---

### SGD (batch_size=1)

```
Epoch  3000 | Loss: 0.013491 | Acc: 1.00
```

Final predictions:

```
## x1 x2 | prob    | pred | true

0  0 | 0.015313 |  0   |  0
0  1 | 0.987699 |  1   |  1
1  0 | 0.987684 |  1   |  1
1  1 | 0.013540 |  0   |  0
```

SGD converged fastest in this toy case.

---

## 5) Softmax Toy Experiment (Multiclass Classification)

To extend beyond binary classification, the network was upgraded with:

- Softmax output activation
- Cross-entropy loss
- Multiclass accuracy metric

A 3-class toy dataset with 3 clusters was trained.

---

### Training Output

```
Training softmax classifier on 3-class toy data
Epoch     1 | Loss: 1.156448 | Acc: 0.40
Epoch   300 | Loss: 0.035091 | Acc: 0.99
Epoch   900 | Loss: 0.019907 | Acc: 1.00
Epoch  3000 | Loss: 0.007751 | Acc: 1.00

Final train acc: 1.000
```

This confirms the model supports multiclass learning successfully.

---

## Level 1 Summary

By the end of Level 1, the project includes:

- Fully working XOR neural network
- BCE loss and accuracy metric
- Model save/load support
- Gradient checking
- Mini-batch + SGD training support
- Softmax + cross entropy multiclass extension
- Batch convergence comparison experiments

---

## Level 2: Batch Size Comparison – Full-Batch, Mini-Batch, SGD

This section presents a comparative analysis of three batch training strategies on the XOR problem: full-batch (GD), mini-batch, and stochastic gradient descent (SGD). All experiments used the same model architecture with parameters `hidden_size=3`, `epochs=3000`, and `learning_rate=0.1`.

### 1) Full-Batch (batch_size=4)

```
Epoch     1 | Loss: 0.729777 | Acc: 0.50
Epoch   500 | Loss: 0.691604 | Acc: 0.50
Epoch  1000 | Loss: 0.687696 | Acc: 0.50
Epoch  1500 | Loss: 0.674883 | Acc: 0.75
Epoch  2000 | Loss: 0.641111 | Acc: 0.75
Epoch  2500 | Loss: 0.580095 | Acc: 0.75
Epoch  3000 | Loss: 0.511175 | Acc: 0.75

Final predictions
x1 x2 | prob    | pred | true
-----------------------------

0  0 | 0.197559 |  0   |  0
0  1 | 0.656794 |  1   |  1
1  0 | 0.590402 |  1   |  1
1  1 | 0.583839 |  1   |  0
```

### 2) Mini-Batch (batch_size=2)

```
Epoch     1 | Loss: 0.752723 | Acc: 0.50
Epoch   500 | Loss: 0.689332 | Acc: 0.50
Epoch  1000 | Loss: 0.653483 | Acc: 0.75
Epoch  1500 | Loss: 0.548127 | Acc: 0.75
Epoch  2000 | Loss: 0.283365 | Acc: 1.00
Epoch  2500 | Loss: 0.101509 | Acc: 1.00
Epoch  3000 | Loss: 0.054176 | Acc: 1.00

Final predictions
x1 x2 | prob    | pred | true
-----------------------------

0  0 | 0.052257 |  0   |  0
0  1 | 0.950494 |  1   |  1
1  0 | 0.950836 |  1   |  1
1  1 | 0.059566 |  0   |  0
```

### 3) SGD (batch_size=1)

```
Epoch     1 | Loss: 0.748187 | Acc: 0.50
Epoch   500 | Loss: 0.702365 | Acc: 0.50
Epoch  1000 | Loss: 0.422140 | Acc: 1.00
Epoch  1500 | Loss: 0.066593 | Acc: 1.00
Epoch  2000 | Loss: 0.029607 | Acc: 1.00
Epoch  2500 | Loss: 0.018625 | Acc: 1.00
Epoch  3000 | Loss: 0.013491 | Acc: 1.00

Final predictions
x1 x2 | prob    | pred | true
-----------------------------

0  0 | 0.015313 |  0   |  0
0  1 | 0.987699 |  1   |  1
1  0 | 0.987684 |  1   |  1
1  1 | 0.013540 |  0   |  0
```

### Overall Evaluation

- **Full-Batch (GD)** showed a stable but slow convergence and failed to fully solve XOR.
- **Mini-Batch** converged faster and more reliably, achieving 100% accuracy within 2000 epochs.
- **SGD** was the fastest method in this case, reaching high accuracy in fewer epochs. However, note that for non-linear datasets it may introduce high variance in learning.

---
