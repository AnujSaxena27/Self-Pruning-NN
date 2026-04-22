# Self-Pruning Neural Network on CIFAR-10

##  Overview
This project implements a self-pruning neural network using PyTorch on the CIFAR-10 image classification dataset. The model learns both classification and sparsity simultaneously by applying learnable gates to its weights. These gates are optimized during training to automatically prune unnecessary connections, effectively reducing model complexity. 

**Primary Objective:** To study and quantify the trade-off between model accuracy and sparsity.

---

##  Key Features
* **Custom PrunableLinear Layer:** Integrates learnable gates directly into the network architecture.
* **End-to-End Training:** Incorporates sparsity regularization natively into the training loop.
* **Lambda Scheduling:** Ensures controlled pruning to prevent early model degradation.
* **Comprehensive Evaluation:** Features Accuracy vs. Sparsity analysis and confusion matrices for performance evaluation.
* **Rich Visualizations:** Tracks and visualizes training behavior, including sparsity growth curves.

---

##  Methodology

### 1. The Prunable Layer
Each linear layer is modified to include a learnable gate mechanism:
* Gate values are passed through a `sigmoid` function.
* **Effective Weight** = `Original Weight × Gate Value`
* Gates "close" (approach `0`) for unimportant connections, effectively pruning them.

### 2. Loss Function Optimization
The model optimizes a combined loss function to balance accuracy and size:
* **Cross-Entropy Loss:** Drives classification accuracy.
* **Sparsity Loss:** Encourages gates to become zero.

> **Total Loss** = `CrossEntropy_Loss + ($\lambda$ × Scale × Sparsity_Loss)`

### 3. Lambda Scheduling & Sparsity Measurement
* **Scheduling:** The $\lambda$ (lambda) multiplier is gradually increased during initial epochs to avoid premature over-pruning.
* **Measurement:** Sparsity is calculated as the percentage of gates that fall below a predefined threshold.

---

##  Training Configuration
| Parameter | Value |
| :--- | :--- |
| **Optimizer** | Adam |
| **Learning Rate** | `0.001` |
| **Batch Size** | `128` |
| **Epochs** | `25` |
| **Sparsity Threshold** | `0.2` |
| **Sparsity Scale** | `5` |
| **Lambda Values Tested** | `[0.0, 0.1, 0.5, 2.0]` |

---

##  Results
The model demonstrates a clear trade-off between accuracy and sparsity. As the lambda penalty increases, the network becomes sparser at the cost of classification accuracy.

| Lambda ($\lambda$) | Accuracy (%) | Sparsity (%) |
| :---: | :---: | :---: |
| **0.0** | ~53 | ~0 |
| **0.1** | ~50 | ~10 – 30 |
| **0.5** | ~45 | ~40 – 70 |
| **2.0** | Lower | ~70 – 90 |

### Generated Visualizations
Running the evaluation generates the following plots:
* Accuracy vs. Sparsity
* Training Loss vs. Epochs
* Validation Accuracy vs. Epochs
* Sparsity Growth Curve
* Confusion Matrix

---

##  Dataset & Project Structure

The model is trained on the **CIFAR-10** dataset (10 classes: *airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*). 

```text
self-pruning-nn/
│
├── dataset/                 # Not included in repo (size constraints)
│    └── train/
│         ├── class_1/
│         └── class_2/
├── saved_models/            # Not included in repo
├── notebook.ipynb           # Interactive exploration and visualization
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
