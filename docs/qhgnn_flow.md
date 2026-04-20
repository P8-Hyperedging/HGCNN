# QHGNN: End-to-End Flow

This document walks through the complete QHGNN pipeline, step by step, explaining **what** happens at each stage and **why**.

---

## Step 1: Load Reviews from PostgreSQL

**What:** `load_postgres_review_data()` fetches all reviews from the Yelp database. Each review is a `(user_id, business_id, stars)` tuple.

**Why:** Reviews are the raw material for everything that follows. They define both the hypergraph structure (which users reviewed which businesses) and the quality scores (how consistent each user's ratings are).

**Source:** `src/data/data.py` -> `load_postgres_review_data()`

---

## Step 2: Build the Incidence Matrix H

**What:** `build_hypergraph_incidence_matrix(reviews)` constructs an N x E binary matrix where N = number of businesses (nodes) and E = number of users (hyperedges). `H[i, j] = 1` if business `i` was reviewed by user `j`.

**Why:** This is how we represent the hypergraph. In a regular graph, edges connect two nodes. In a hypergraph, a single hyperedge can connect any number of nodes. Here, each user who reviewed multiple businesses creates a hyperedge linking all those businesses together. The incidence matrix is the standard way to encode this structure for matrix operations.

**Source:** `src/data/n_preprocessing.py:10-32`

---

## Step 3: Load Business Data and Build Feature Matrix

**What:** `load_postgres_business_list_data()` and `load_postgres_business_list_opening_hours()` fetch business attributes. Then `create_business_feature_matrix()` assembles them into an N x F matrix:

| Index | Feature | Encoding |
|---|---|---|
| 0 | Review count | Raw integer |
| 1 | Longitude | Raw float |
| 2 | Latitude | Raw float |
| 3-16 | Opening hours | Minutes since midnight for each open/close time across 7 days; -1 if missing |
| 17+ | Categories | One-hot encoded |

**Why:** The model needs a numeric feature vector for each business node. These features capture what each business *is* (its type, location, popularity, and availability) independently of the hypergraph structure. The model will learn how to combine these features with the relational information from the hypergraph.

**Source:** `src/data/n_preprocessing.py:101-117`

---

## Step 4: Create Label Vector

**What:** `create_label_vector()` converts each business's star rating into one of 9 integer classes:

```
class = round(stars * 2) - 2

1.0 stars -> 0,  1.5 -> 1,  2.0 -> 2,  ...  4.5 -> 7,  5.0 -> 8
```

**Why:** This is a multi-class classification problem. The model will predict which of these 9 bins a business belongs to. We use discrete classes rather than regression because CrossEntropyLoss is well-suited for ordinal categories and gives cleaner gradients than MSE on a continuous target.

**Source:** `src/data/n_preprocessing.py:121-127`

---

## Step 5: Train/Validation Split

**What:** `rand_train_test_idx_simple(n, train_prop)` randomly partitions the node indices into a training set and a validation set (default 80/20 split).

**Why:** Standard ML practice. The model trains on the training nodes and is evaluated on held-out validation nodes to detect overfitting. Importantly, the model still sees the *full* hypergraph structure during both phases (all of H, LS, RS, Q) -- only the loss and accuracy are computed on the respective subsets. This is transductive learning: the graph is fixed, and we're predicting missing labels.

**Source:** `src/data/n_preprocessing.py:129-140`

---

## Step 6: Compute Quality Vector Q

**What:** `create_quality_matrix_from_H(reviews)` produces a vector of length E (one value per user/hyperedge):

```
For each user j:
    mean_j     = average of all star ratings user j gave
    variance_j = average squared deviation from mean_j
    Q[j]       = 1 - (variance_j / 4.0)
```

`MAX_POSSIBLE_VARIANCE = 4.0` is the theoretical maximum (e.g., a user who only gives 1-star and 5-star reviews: `((1-3)^2 + (5-3)^2) / 2 = 4`).

**Why:** This is what distinguishes QHGNN from a standard HGNN. The idea is that **not all hyperedges are equally informative**. A user who gives every restaurant 4 stars (Q close to 1.0) is expressing a strong, consistent signal: the businesses they connect are genuinely similar in their eyes. A user who gives wildly varying ratings (Q close to 0.0) provides a weaker, noisier signal. By encoding this as a quality score, we let the model weight information flow through high-quality hyperedges more heavily.

**Source:** `src/data/n_preprocessing.py:75-98`

---

## Step 7: Decompose H into LS and RS

**What:** `generate_G_from_H(H, variable_weight=True)` computes two normalized matrices from the incidence matrix:

```
DV = diagonal matrix of node degrees (how many hyperedges each business appears in)
DE = diagonal matrix of edge degrees (how many businesses each user reviewed)

LS = DV^(-0.5) * H              (N x E)  -- "left side"
RS = DE^(-1) * H^T * DV^(-0.5)  (E x N)  -- "right side"
```

**Why:** In a standard HGNN, you would pre-compute the full graph Laplacian `G = LS @ RS` (an N x N matrix) and use it directly. But QHGNN needs to **inject quality weights between LS and RS** during the forward pass, so we keep them separated. The degree normalization (`DV^(-0.5)` and `DE^(-1)`) prevents high-degree nodes and hyperedges from dominating the aggregation purely due to having more connections.

When the model runs, it will compute `G = (LS * Q_updated) @ RS` -- plugging quality weights into the gap where the identity weight matrix would normally sit.

**Source:** `src/utils/utils.py:95-124`

---

## Step 8: Convert to Tensors and Move to Device

**What:** All numpy arrays (feature matrix, labels, LS, RS, Q, train/val indices) are converted to `torch.Tensor` and moved to GPU if available. The sparse LS and RS matrices are converted to dense with `.toarray()`.

**Why:** PyTorch requires tensors for gradient computation. GPU acceleration speeds up the matrix multiplications significantly. The sparse-to-dense conversion is noted as a pain point in the code (`"Unsparsing :("`) -- it uses more memory but is required because the subsequent operations (element-wise multiplication with Q, matmul) don't currently use sparse tensor operations.

**Source:** `src/model/QualityHGNN/train.py:97-105`

---

## Step 9: Initialize the QHGNN Model

**What:** The model is a 2-layer network:

```python
QHGNN(in_ch=F, n_class=9, n_hid=hidden_layer_size, dropout=0.5)
```

This creates:
- `hgc1`: QHGNN_conv(F -> hidden_layer_size) -- projects input features to hidden dim
- `hgc2`: QHGNN_conv(hidden_layer_size -> 9) -- projects hidden features to class logits

**Why:** Two layers means information propagates through the hypergraph **twice**. In the first layer, each business aggregates information from businesses that share a reviewer with it (1-hop neighbors). In the second layer, it aggregates information from 2-hop neighbors. This is enough to capture local community structure without over-smoothing (where all node representations become identical).

**Source:** `src/model/QualityHGNN/QHGNN.py:6-20`

---

## Step 10: Set Up Optimizer, Scheduler, and Loss

**What:**
- **Optimizer:** Adam with learning rate and weight decay (L2 regularization)
- **Scheduler:** MultiStepLR -- multiplies the LR by `gamma` (default 0.5) at each milestone epoch (default: 50, 100)
- **Loss:** CrossEntropyLoss

**Why:** Adam adapts per-parameter learning rates, which works well for sparse hypergraph data where different features update at different rates. The step scheduler reduces the LR partway through training so the model can fine-tune after the initial rapid learning phase. CrossEntropyLoss is the standard choice for multi-class classification.

**Source:** `src/model/QualityHGNN/train.py:121-124`

---

## Step 11: Training Loop (per epoch)

**What:** Each epoch has two phases:

### Train Phase
1. Zero gradients
2. **Forward pass**: `outputs = model(fts, LS, RS, Q)` -- runs all N nodes through both QHGNN_conv layers (see Step 12 for detail)
3. Compute loss on **training nodes only**: `loss = CrossEntropy(outputs[idx_train], labels[idx_train])`
4. Backpropagate and update weights

### Validation Phase
1. Same forward pass (all nodes), but with `torch.no_grad()`
2. Compute loss and accuracy on **validation nodes only**
3. Compute macro F1, Precision, Recall, and confusion matrix
4. If this is the best validation accuracy so far, save a copy of the model weights

**Why:** The model processes all nodes every pass (transductive setting), but the loss signal only comes from the relevant subset. Tracking best validation accuracy and restoring those weights at the end prevents returning an overfit model from a later epoch.

**Source:** `src/model/QualityHGNN/train.py:154-271`

---

## Step 12: Inside the QHGNN_conv Forward Pass (the core)

This is where the quality-aware aggregation happens. Each QHGNN_conv layer does:

### 12a. Linear Transformation

```python
x = x @ W + b       # (N x in_features) @ (in_features x out_features) = (N x out_features)
```

**Why:** Projects node features into a new representation space. This is the learnable part -- the weight matrix W and bias b are the parameters the optimizer updates. It allows the model to learn which combinations of input features are useful before aggregation.

### 12b. Compute Hyperedge Centroids

```python
membership = (LS > 0).float()                              # (N x E) binary: which nodes belong to which edges
centroids = (membership.T @ x) / nodes_per_edge.unsqueeze(1)  # (E x out_features)
```

**Why:** The centroid of a hyperedge is the average feature vector of all its member nodes (after the linear transform). It represents "what a typical business looks like" according to that reviewer. This is computed fresh each forward pass, meaning it reflects the current state of the learned features, not the raw input features.

### 12c. Compute Node-to-Centroid Distances

```python
for each chunk of hyperedges:
    diffs = x.unsqueeze(1) - centroids[chunk].unsqueeze(0)  # (N x chunk x F)
    dists = diffs.norm(dim=2) * membership[:, chunk]          # zero out non-members
    total_dists[chunk] = dists.sum(dim=0)                     # sum distances per edge
```

**Why:** This measures how *spread out* the nodes are within each hyperedge in the current feature space. A hyperedge whose member businesses have converged to similar representations has low total distance; one whose members are still scattered has high total distance. The chunking (100 edges at a time) prevents memory overflow when E is large.

### 12d. Update Quality Scores

```python
distance_scores = 1.0 / (1.0 + total_dists)       # (E,) -- high dist -> low score
Q_updated = clamp(distance_scores * Q, min=0, max=10)
```

**Why:** The final quality of a hyperedge combines two signals:
- **Initial quality Q** (from Step 6): how consistent the user's star ratings were (a static, data-derived prior)
- **Distance score** (from Step 12c): how tight the cluster is in the current learned feature space (a dynamic, model-derived signal)

Multiplying them means a hyperedge needs *both* consistent ratings *and* similar learned features to get a high weight. This is computed inside `torch.no_grad()` -- the quality update is a heuristic that guides aggregation but is not itself learned through backpropagation.

### 12e. Build Dynamic Graph and Aggregate

```python
G = (LS * Q_updated.unsqueeze(0)) @ RS    # (N x E) * (E,) -> (N x E) @ (E x N) = (N x N)
x = G @ x                                  # (N x N) @ (N x out_features) = (N x out_features)
```

**Why:** This is the hypergraph convolution. `LS * Q_updated` scales each column of LS (each hyperedge) by its quality score. The resulting G matrix encodes how much influence each business has on every other business, weighted by the quality of the hyperedges connecting them. The final `G @ x` aggregates each node's features with a quality-weighted combination of its neighbors' features.

**Source:** `src/model/QualityHGNN/layers.py:25-58`

---

## Step 13: Full Forward Pass (both layers together)

```python
x = relu(QHGNN_conv_1(x, LS, Q, RS))   # (N x F) -> (N x hidden)
x = dropout(x)
x = QHGNN_conv_2(x, LS, Q, RS)          # (N x hidden) -> (N x 9)
```

**Why:** ReLU introduces non-linearity between layers, letting the model learn non-linear feature combinations. Dropout randomly zeros out hidden features during training to prevent over-reliance on any single feature (regularization). The second layer has no ReLU because its output feeds directly into CrossEntropyLoss, which expects raw logits.

Note: Q is updated independently in *each* layer based on the current features at that layer. Layer 1 computes quality based on raw-feature-space distances; Layer 2 computes quality based on hidden-representation distances.

**Source:** `src/model/QualityHGNN/QHGNN.py:16-20`

---

## Step 14: Evaluation Metrics

**What:** On validation nodes, the following are computed each epoch:
- **Accuracy**: fraction of correctly classified nodes
- **Baseline accuracy**: accuracy of always predicting the majority class (sanity check)
- **Macro F1, Precision, Recall**: averaged across all 9 classes equally (via TorchMetrics)
- **Confusion matrix**: 9x9 grid of predicted vs actual classes

**Why:** Accuracy alone can be misleading with imbalanced classes (if 40% of businesses are 4-star, a model that always predicts 4-star gets 40% accuracy for free). Macro F1 ensures the model performs well across *all* star ratings, not just common ones. The baseline comparison shows whether the model is actually learning beyond the trivial strategy.

**Source:** `src/model/QualityHGNN/train.py:157-163, 212-221`

---

## Step 15: Save Results to Database

**What:** After training completes, `output_metrics_to_db()` writes the final results (best validation accuracy, training time, hyperparameters, seed) back to PostgreSQL.

**Why:** Enables experiment tracking across runs. By storing the seed, any run can be reproduced exactly. By storing hyperparameters as JSON alongside results, you can compare which configurations work best.

**Source:** `src/model/QualityHGNN/train.py:142-151`, `src/data/data.py:209-217`

---

## Summary: The Key Idea

Standard hypergraph neural networks treat all hyperedges equally -- if two businesses were reviewed by the same user, they get the same amount of influence on each other regardless of whether that user gives thoughtful, consistent reviews or chaotic, random ones.

QHGNN adds a **quality gate** to this information flow. It combines a static prior (review consistency from the data) with a dynamic signal (feature-space tightness from the current model state) to determine how much each hyperedge should contribute to the aggregation. High-quality hyperedges pass information freely; low-quality ones are dampened. This quality assessment is recomputed at every layer with every forward pass, adapting as the model learns better representations.
