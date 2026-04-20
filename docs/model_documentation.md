# HGCNN Model Documentation

This document describes the three hypergraph neural network models in this repository, how data flows through the system, and how the models differ from one another.

## 1. Overview

This system classifies Yelp businesses by their star rating (9 classes, from 1.0 to 5.0 in 0.5 increments) using **hypergraph neural networks**. The core insight is modelling the relationship between businesses and users as a hypergraph:

- **Nodes** = businesses
- **Hyperedges** = users (each user who reviewed multiple businesses forms a hyperedge connecting those businesses)

Three models are implemented, each taking a different approach to learning on this hypergraph structure:

| Model | Approach | Complexity |
|---|---|---|
| **MoonLabHGNN** | Static hypergraph Laplacian convolution | Low |
| **QHGNN** | Quality-weighted hypergraph convolution | Medium |
| **AllSetTransformer** | Attention-based bipartite message passing | High |

All three are served through a Flask REST API with WebSocket support for real-time training progress.

---

## 2. Data Pipeline & Preprocessing

### 2.1 Data Source

All data is loaded from a PostgreSQL database containing Yelp review data (`src/data/data.py`). The key entities are:

- **Business** - id, name, categories, star rating, review count, latitude, longitude
- **Review** - user_id, business_id, star rating
- **User** - user_id, review count
- **OpeningHours** - 7 days x (open time, close time) as datetime values

### 2.2 Hypergraph Incidence Matrix (shared by QHGNN and MoonLabHGNN)

Constructed in `src/data/n_preprocessing.py` -> `build_hypergraph_incidence_matrix()`:

```
H is an N x E matrix where:
  N = number of businesses (nodes)
  E = number of users (hyperedges)
  H[i, j] = 1 if business i was reviewed by user j
```

This matrix is the foundation for both MoonLabHGNN and QHGNN. It encodes which businesses are connected through shared reviewers.

### 2.3 Feature Matrix (shared by QHGNN and MoonLabHGNN)

Constructed in `src/data/n_preprocessing.py` -> `create_business_feature_matrix()`:

Each business gets a feature vector of length `17 + num_categories`:

| Index | Feature |
|---|---|
| 0 | Review count |
| 1 | Longitude |
| 2 | Latitude |
| 3-16 | Opening hours (7 days x open/close, in minutes since midnight; -1 if missing) |
| 17+ | One-hot encoded business categories |

### 2.4 Label Vector

Constructed in `src/data/n_preprocessing.py` -> `create_label_vector()`:

Star ratings (1.0 to 5.0) are mapped to 9 integer classes (0 to 8):

```
class = round(stars * 2) - 2

Examples: 1.0 stars -> class 0, 3.0 stars -> class 4, 5.0 stars -> class 8
```

### 2.5 Quality Matrix (QHGNN only)

Constructed in `src/data/n_preprocessing.py` -> `create_quality_matrix_from_H()`:

A vector of length E (one score per user/hyperedge), measuring how **consistent** a reviewer is:

```
mean     = average star rating across all of a user's reviews
variance = average squared deviation from that mean
quality  = 1 - (variance / MAX_POSSIBLE_VARIANCE)

MAX_POSSIBLE_VARIANCE = 4.0  (the theoretical max for ratings between 1 and 5)
```

A quality score of **1.0** means the user gives identical ratings to every business (maximally consistent). A score near **0.0** means the user's ratings vary wildly. The intuition: consistent reviewers form more meaningful hyperedges because the businesses they connect likely share genuine similarities.

### 2.6 Graph Laplacian / Decomposition (MoonLabHGNN and QHGNN)

Computed in `src/utils/utils.py` -> `generate_G_from_H()`:

Given the incidence matrix H, two forms are produced:

**For MoonLabHGNN** (`variable_weight=False`):
```
G = DV^(-0.5) * H * W * DE^(-1) * H^T * DV^(-0.5)

Where:
  DV = diagonal matrix of node degrees (how many hyperedges each business belongs to)
  DE = diagonal matrix of hyperedge degrees (how many businesses each user reviewed)
  W  = identity (all edges weighted equally)
```

G is a single N x N matrix that is pre-computed once and reused throughout training.

**For QHGNN** (`variable_weight=True`):
```
LS = DV^(-0.5) * H                    (left side, N x E)
RS = DE^(-1) * H^T * DV^(-0.5)        (right side, E x N)
```

These are returned separately so the quality weights Q can be injected between them during the forward pass: `G_dynamic = (LS * Q) @ RS`.

### 2.7 AllSetTransformer Preprocessing (separate pipeline)

The AllSetTransformer uses its own preprocessing in `src/data/AllSetTransformer_preprocessing.py` -> `load_yelp_dataset()`, which differs in two key ways:

**Different features**: Instead of opening hours and categories, it uses a **bag-of-words** encoding of business names (up to 1000 features), plus latitude, longitude, and review count.

**Different graph representation**: Instead of an incidence matrix, it produces a bipartite `edge_index` tensor for torch_geometric's message passing framework:
```
edge_index = [[node_0, node_1, ...],     # source (business) indices
              [hedge_0, hedge_1, ...]]    # target (user/hyperedge) indices

Hyperedge IDs are offset by num_nodes to avoid ID collisions.
```

Additional preprocessing steps:
- `ExtractV2E()` - ensures only vertex-to-edge direction is kept
- `Add_Self_Loops()` - adds self-loops for isolated nodes
- `expand_edge_index()` - expands hyperedges into star-graph structures
- `norm_contruction()` - computes edge normalization weights (uniform or degree-based)
- `rand_train_test_idx()` - balanced train/validation/test split

---

## 3. MoonLabHGNN

**Files**: `src/model/MoonLabHGNN/HGNN.py`, `src/model/MoonLabHGNN/layers.py`

### Architecture

The simplest of the three models. A 2-layer hypergraph neural network that applies a pre-computed, static graph Laplacian.

```
Input features X (N x F)
        |
  [HGNN_conv layer 1]  ->  X' = G @ (X @ W1 + b1)     (N x hidden)
        |
      ReLU
        |
     Dropout
        |
  [HGNN_conv layer 2]  ->  X'' = G @ (X' @ W2 + b2)   (N x num_classes)
        |
     Output logits
```

### How HGNN_conv Works

Each convolution layer (`src/model/MoonLabHGNN/layers.py:8-30`) does two things:

1. **Linear transformation**: Multiply node features by a learnable weight matrix (and add bias). This projects features into a new space.
2. **Hypergraph aggregation**: Multiply by the pre-computed Laplacian G. This propagates information between nodes that share hyperedges (i.e., businesses reviewed by the same users).

```python
def forward(self, x, G):
    x = x.matmul(self.weight)   # linear transform
    if self.bias is not None:
        x = x + self.bias
    x = G.matmul(x)             # hypergraph aggregation
    return x
```

The Laplacian G encodes the full hypergraph structure in a single N x N matrix. Because it is pre-computed and static, the aggregation pattern is fixed throughout training - the model only learns the weight matrices W1 and W2.

### Training

- **Optimizer**: Adam with MultiStepLR scheduler (decays LR at specified milestones)
- **Loss**: CrossEntropyLoss
- **Split**: Simple random train/validation split
- **Best model**: Saved based on validation accuracy

---

## 4. QHGNN (Quality Hypergraph Neural Network)

**Files**: `src/model/QualityHGNN/QHGNN.py`, `src/model/QualityHGNN/layers.py`

### Architecture

Same 2-layer structure as MoonLabHGNN, but the convolution layer dynamically adjusts hyperedge weights based on quality scores.

```
Input features X (N x F),  LS (N x E),  RS (E x N),  Q (E,)
        |
  [QHGNN_conv layer 1]  ->  quality-weighted aggregation   (N x hidden)
        |
      ReLU
        |
     Dropout
        |
  [QHGNN_conv layer 2]  ->  quality-weighted aggregation   (N x num_classes)
        |
     Output logits
```

### How QHGNN_conv Works

This is the core innovation (`src/model/QualityHGNN/layers.py:7-58`). Each forward pass:

**Step 1 - Linear transformation** (same as MoonLab):
```python
x = x @ W + b
```

**Step 2 - Dynamic quality update** (unique to QHGNN, computed with `torch.no_grad()`):

First, compute the **centroid** of each hyperedge - the average feature vector of all member nodes:
```python
membership = (LS > 0).float()                        # which nodes belong to which edges
centroids = (membership.T @ x) / nodes_per_edge      # mean feature vector per edge (E x F)
```

Then, measure how far each node is from its hyperedge's centroid. Hyperedges where nodes are tightly clustered get higher quality; spread-out hyperedges get lower quality:
```python
# For each hyperedge, sum the distances from member nodes to centroid
total_dists[e] = sum of ||node_i - centroid_e|| for all nodes i in edge e

# Convert distances to quality scores
distance_scores = 1 / (1 + total_dists)              # high distance -> low score
Q_updated = clamp(distance_scores * Q_initial, 0, 10) # combine with initial quality
```

**Step 3 - Quality-weighted aggregation**:
```python
G = (LS * Q_updated) @ RS    # inject quality weights between left and right matrices
x = G @ x                     # aggregate
```

### How QHGNN Differs from MoonLabHGNN

| Aspect | MoonLabHGNN | QHGNN |
|---|---|---|
| Graph matrix | Single static G, pre-computed once | LS and RS separated; G is recomputed every forward pass |
| Edge weighting | All hyperedges weighted equally | Hyperedges weighted by quality (consistency + centroid distance) |
| Quality scores | None | Initial quality from review variance; updated dynamically based on feature-space distances |
| Adaptivity | Fixed aggregation pattern | Aggregation pattern changes as features evolve during training |

The key insight: not all hyperedges are equally informative. A user who rates every restaurant 4 stars (high initial quality) and connects restaurants with similar features (low centroid distance) provides a strong learning signal. A user with erratic ratings connecting dissimilar businesses provides a weaker signal. QHGNN downweights the latter.

### Training

- **Optimizer**: Adam with MultiStepLR scheduler
- **Loss**: CrossEntropyLoss
- **Metrics**: Accuracy, macro F1, macro Precision, macro Recall, confusion matrix (via TorchMetrics)
- **Split**: Simple random train/validation split
- **Best model**: Saved based on validation accuracy

---

## 5. AllSetTransformer (SetGNN)

**Files**: `src/model/AllSetTransformer/AllSetTransformer.py`, `src/model/AllSetTransformer/layers.py`

Adapted from the [AllSet paper](https://github.com/jianhao2016/AllSet).

### Architecture

A fundamentally different approach. Instead of computing a single graph Laplacian, it uses **bipartite message passing** - alternating between aggregating information from nodes to hyperedges (V2E) and from hyperedges back to nodes (E2V).

```
Input features X (N x F)
        |
   Input dropout (p=0.2)
        |
   For each layer i = 0..num_layers-1:
        |
     [V2E convolution]  ->  aggregate node features into hyperedge representations
        |  ReLU + Dropout
        |
     [E2V convolution]  ->  aggregate hyperedge representations back into node features
        |  ReLU + Dropout
        |
   [MLP Classifier]  ->  final node-level predictions
        |
     Output logits
```

### How HalfNLHconv Works

Each V2E and E2V step is a `HalfNLHconv` layer (`src/model/AllSetTransformer/layers.py:205-277`). It operates in one of two modes:

**With PMA (attention=True, default)**: Delegates to the PMA (Pooling by Multihead Attention) module.

**Without PMA (attention=False)**: Uses an MLP encoder, scatter-based message passing, then an MLP decoder.

### How PMA Works

PMA (`src/model/AllSetTransformer/layers.py:26-159`) is the heart of AllSetTransformer. It implements attention-weighted set aggregation:

1. **Project** input features into key (K) and value (V) spaces via linear layers
2. **Compute attention** weights: `alpha = softmax(leaky_relu(K * seed_vector))` where the seed vector is a learnable parameter
3. **Aggregate** values weighted by attention: `out = sum(V * alpha)`
4. **Skip connection** with the seed vector: `out = out + seed_vector`
5. **LayerNorm + Feed-forward**: `out = LN(out + ReLU(FF(LN(out))))`

This allows the model to learn **which nodes matter most** within each hyperedge (for V2E) and **which hyperedges matter most** for each node (for E2V), rather than treating all connections equally.

### How AllSetTransformer Differs from the Other Two

| Aspect | MoonLabHGNN / QHGNN | AllSetTransformer |
|---|---|---|
| Graph framework | Dense matrix multiplication | torch_geometric message passing |
| Aggregation | Pre-computed Laplacian (fixed or quality-weighted) | Learned attention weights per edge |
| Message direction | Implicit in G (node-edge-node in one step) | Explicit V2E then E2V in separate steps |
| Hyperedge features | None (hyperedges are structural only) | Hyperedges get learned intermediate representations |
| Feature input | Location, hours, categories | Bag-of-words from business names, location, review count |
| Learnable components | Weight matrices only (2 per layer) | Attention parameters, MLPs, LayerNorms, seed vectors |
| Scalability | Dense matrices limit scalability | Sparse message passing scales better |

### Training

- **Optimizer**: Adam (no LR scheduler)
- **Loss**: NLLLoss with log_softmax
- **Split**: Balanced or random train/validation/test split
- **Evaluation**: Accuracy on train, validation, and test sets

---

## 6. How It All Hangs Together

### End-to-End Flow

```
                        PostgreSQL (Yelp data)
                               |
                    +----------+----------+
                    |                     |
           n_preprocessing.py    AllSetTransformer_preprocessing.py
           (for MoonLab/QHGNN)  (for AllSetTransformer)
                    |                     |
              +-----+-----+              |
              |           |              |
          Features    Incidence      edge_index
          Labels      Matrix H       BoW features
          Quality Q   -> G or        normalization
                        (LS,RS)      weights
              |           |              |
              +-----+-----+       +------+
                    |              |
              MoonLabHGNN     AllSetTransformer
              or QHGNN          (SetGNN)
                    |              |
                    +------+-------+
                           |
                    Training Loop
                    (Adam optimizer,
                     loss, metrics)
                           |
                  output_metrics_to_db()
                  (results -> PostgreSQL)
```

### Flask API (`src/flaskeladen.py`)

The system is served through a REST API:

- **`GET /models`** - Lists available models (MoonLabHGNN, QHGNN, AllSetTransformer)
- **`GET /params/<model>`** - Returns configurable hyperparameters for a model (from `src/parameters.py`)
- **`POST /train/<model>`** - Starts an async training job; returns a job ID
- **WebSocket** - Streams real-time training progress (loss, accuracy per epoch)

Training runs asynchronously in background threads. The `job_store.py` module tracks job state with thread-safe locking.

### Entry Points

- **CLI**: `src/main.py` - directly instantiates a trainer and runs it
- **API**: `src/flaskeladen.py` - `train_model_async()` routes to the appropriate trainer class based on model name

### Default Hyperparameters (`src/parameters.py`)

| Parameter | AllSetTransformer | MoonLabHGNN | QHGNN |
|---|---|---|---|
| Epochs | 500 | 500 | 500 |
| Learning Rate | 0.001 | 0.001 | 0.001 |
| Hidden Size | 64 | 128 | 128 |
| Train Proportion | 0.5 | 0.5 | 0.5 |
| Dropout | 0 | 0 | 0 |
| Weight Decay | 0 | 0.0005 | 0.0005 |
| LR Decay (gamma) | - | 0.5 | 0.5 |
| LR Milestones | - | 50 | 50 |

---

## 7. Summary

The three models represent a spectrum of approaches to hypergraph learning:

**MoonLabHGNN** is the baseline - it pre-computes a fixed graph Laplacian and learns simple linear transformations. It is fast and easy to understand but treats all hyperedges as equally important.

**QHGNN** adds the concept of **hyperedge quality** - users who give consistent reviews and connect similar businesses get higher weight. Quality scores are updated dynamically during each forward pass based on feature-space distances, making the aggregation pattern adaptive.

**AllSetTransformer** takes the most expressive approach - it learns attention weights that determine how much each node/hyperedge contributes to its neighbors. It also gives hyperedges their own learned representations (not just structural connections). This comes at the cost of more parameters and a different preprocessing pipeline.

All three share the same underlying data (Yelp businesses and reviews), the same classification task (predicting star ratings), and the same serving infrastructure (Flask API with async training).
