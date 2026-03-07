# TAWOS Story Point Estimation - Federated Learning Feasibility Guide

## Quick Answer
✅ **YES, it's feasible** to run federated learning on TAWOS dataset. Your current framework can be adapted with moderate changes.

---

## 1. Data Extraction & Features

### Option A: Tabular Features (RECOMMENDED ✅)
**What it means:** Extract structured numerical/categorical features from database issues
- **Example features:** Issue description length, num_comments, num_links, assignee_changes, time_to_resolution, sprint_duration, component_count
- **Format:** CSV with Features (columns) × Issues (rows)
- **Effort:** Low (SQL queries → CSV)
- **Why best for FL:** Lightweight, easy to distribute across workers, reproducible

### Option B: Text Embeddings
**What it means:** Convert issue descriptions → numerical vectors (using BERT/embeddings)
- **Example:** 768-dim vector per issue description
- **Format:** High-dimensional numpy arrays
- **Effort:** Medium (need embedding model, storage intensive)
- **Why not ideal now:** Overkill for MVP, much larger data, harder to distribute

**→ RECOMMENDATION: Start with Option A (Tabular Features)**

---

## 2. Task Type

### Option A: Regression (RECOMMENDED ✅)
**What it means:** Predict exact story point values (could be 1, 2, 3, 5, 8, 13, 21, etc.)
- **Model output:** Continuous number (neural network outputs 4.2, then round to nearest SP)
- **Loss function:** MSE (Mean Squared Error)
- **Evaluation:** RMSE, MAE (Mean Absolute Error)
- **Data:** TAWOS has actual SP values → ideal for regression
- **Why best:** Most common in industry, matches agile reality

### Option B: Classification
**What it means:** Categorize issues into SP buckets (Class_1, Class_2, Class_3, etc.)
- **Model output:** Probability for each category
- **Loss function:** Cross-Entropy Loss
- **Evaluation:** Accuracy, F1-Score
- **Why not ideal:** Loses granularity, treating SP as categories vs continuous scale

**→ RECOMMENDATION: Use Regression**

---

## 3. Model Architecture

### For Federated Learning with Tabular Data:

**Use: Multi-Layer Perceptron (MLP) Neural Network**
```
Input Features (e.g., 20 features)
    ↓
Hidden Layer 1 (128 neurons, ReLU)
    ↓
Hidden Layer 2 (64 neurons, ReLU)
    ↓
Output Layer (1 neuron, Linear) → Story Point Prediction
```

**Why MLP?**
- ✅ Works great for tabular data
- ✅ Easy to aggregate parameters in federated averaging (FedAvg)
- ✅ Fits existing FL framework perfectly (just swap CNN → MLP)
- ✅ Fast training

**Why NOT:**
- ❌ Transformers: overkill, need text encoding first
- ❌ Gradient boosting (XGBoost): hard to federate, not deep learning
- ❌ Linear models: too simple for complex feature interactions

**→ RECOMMENDATION: Custom PyTorch MLP + keep existing FedAvg infrastructure**

---

## 4. Data Distribution Strategy

### Option A: By Project (Simulates Real Teams) ✅ RECOMMENDED
```
Worker 1: All issues from Apache Mesos project
Worker 2: All issues from Apache MXNet project
Worker 3: All issues from Atlassian Confluence project
...
Worker N: Issues from last project
```
- **Advantage:** Non-IID data (realistic), each team has different estimation patterns
- **Simulates:** Different teams collaborate via FL without sharing raw data
- **Number of workers:** 39 projects → could use 10-15 major projects

### Option B: Random Distribution (IID)
```
Randomly shuffle all 450K issues, divide equally
```
- **Advantage:** Simpler, standard ML practice
- **Disadvantage:** Unrealistic for real federated scenario
- **Why less ideal:** Loses the heterogeneous data characteristic of FL challenges

### Option C: Other Strategies
- By timezone/geography (not available in TAWOS)
- By issue type (Bug vs Feature) - possible but less natural

**→ RECOMMENDATION: Option A (By Project) - more realistic and interesting for FL research**

---

## 5. Attack/Defense Scope

**Your choice: NO ATTACKS, just aggregation**

✅ **Good decision for MVP** because:
- Focuses on core FL functionality first
- Tests whether framework adapts to new data
- Can add poisoning later if needed
- Cleaner baseline results

**Scope:** Pure FedAvg without adversarial scenarios

---

## Alternative Approaches Not Recommended

### Data Extraction Alternative: Text Embeddings (Option B)
**What would happen if we chose this:**
- Use BERT/Sentence Transformers to embed issue descriptions
- Store 768-dim vectors per issue (massive storage)
- Workers train on embedding vectors
- **Problems:** Overkill for SP estimation, harder to federate, slower training, need pre-trained model
- **When to use:** If you had complex text analysis needs (not for simple SP prediction)

### Task Type Alternative: Classification (Option B)
**What would happen if we chose this:**
- Bucket story points: [1-3] = "Small", [5-8] = "Medium", [13+] = "Large"
- Train classifier instead of regressor
- **Problems:** Loses precision, doesn't match agile continuity
- **When to use:** If you needed risk categories instead of exact estimates

### Model Alternative: Gradient Boosting (XGBoost)
**What would happen if we chose this:**
- Traditional ML approach (not deep learning)
- Better explainability for features
- **Problems:** Hard to federate (tree-based, not continuous), harder parameter aggregation
- **When to use:** If explainability > federated learning research

### Distribution Alternative: Random IID (Option B)
**What would happen if we chose this:**
- All 450K issues shuffled, divided equally
- Standard data distribution approach
- **Problems:** Unrealistic for real teams, misses heterogeneous data challenges
- **When to use:** If you wanted to compare IID vs non-IID performance

---

## 6. Timeline & Feasibility

**For quick feasibility check: 3-5 days work**

| Task | Effort | Time |
|------|--------|------|
| Download TAWOS, extract features | Low | 1 day |
| Create feature loader module | Low | 1 day |
| Build MLP model | Low | 1 day |
| Integrate with FL framework | Medium | 1-2 days |
| Run test experiment | Low | 1 day |

---

## Summary: Recommended Setup

```
DATA PIPELINE:
TAWOS MySQL Database
    ↓ (SQL queries)
Extract ~20 tabular features per issue
    ↓
CSV dataset (450K rows × 20 cols)
    ↓
Distribute by Project across 10-15 workers
    ↓
TRAINING:
Each worker trains local MLP model on their project issues
    ↓
FedAvg aggregates models across workers
    ↓
Global model predicts story points
```

---

## Changes Needed to Current Framework

| Component | Current | New | Effort |
|-----------|---------|-----|--------|
| Data Loader | Image loading (CIFAR/Fashion) | Tabular CSV loading | LOW |
| Model | CNN | MLP | LOW |
| Loss Function | Cross-Entropy | MSE (Regression) | LOW |
| Data Distribution | IID batches | By-project partition | MEDIUM |
| No changes needed | Server.py, Client.py, FedAvg, API | ✅ Reusable | - |

---

## Final Verdict

✅ **FEASIBLE** - Your current federated learning framework is flexible enough.
- No major architecture changes needed
- Reuse: Server, Client, FedAvg, API, Web UI
- Main work: Replace data loading + model architecture
- **Estimated total: 3-5 days** for working POC

🎯 **Recommended approach:** Start with tabular features + MLP + by-project distribution (Option A for all questions) = easiest + most realistic for your use case.

---

## Implementation Plan: How to Do This in Current Setup

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CURRENT SETUP                             │
│  (Federated Learning Framework for Label Flipping Attacks)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │  REUSABLE COMPONENTS (NO CHANGES)        │
        ├──────────────────────────────────────────┤
        │ ✅ server.py → FL training loop          │
        │ ✅ client.py → Local worker training     │
        │ ✅ federated_learning/utils/fed_avg.py   │
        │    → Parameter averaging                 │
        │ ✅ api_server.py → Web API               │
        │ ✅ frontend/ → React visualization       │
        │ ✅ arguments.py → Config system           │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │  NEW/MODIFIED COMPONENTS                 │
        ├──────────────────────────────────────────┤
        │ 🆕 tawos_feature_extractor.py            │
        │    → Download TAWOS, extract 20 features │
        │    → Output: CSV (450K issues × 20 cols) │
        │                                          │
        │ 🆕 tawos_data_loader.py                  │
        │    → Load CSV by project partition       │
        │    → Distribute to workers (10-15)       │
        │                                          │
        │ 🆕 tawos_mlp_model.py                    │
        │    → Replace CNN with MLP network        │
        │    → Input: 20 features                  │
        │    → Output: 1 (story point prediction)  │
        │    → Loss: MSE (not CrossEntropy)        │
        │                                          │
        │ 🔄 Modified: generate_data_distribution.py
        │    → Use TAWOS CSV instead of images     │
        │                                          │
        │ 🔄 Modified: arguments.py                │
        │    → Add TAWOS dataset option            │
        │    → Change loss function to MSE         │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │  EXECUTION FLOW                          │
        ├──────────────────────────────────────────┤
        │ 1. Download TAWOS MySQL dump             │
        │ 2. Run tawos_feature_extractor.py        │
        │    → Creates: data/tawos_features.csv    │
        │                                          │
        │ 3. Run generate_data_distribution.py     │
        │    → Partitions by project               │
        │    → Creates 15 local data pickles       │
        │                                          │
        │ 4. Run api_server.py + frontend          │
        │    → Select "TAWOS" dataset              │
        │    → Configure MLP for SP regression     │
        │    → Start federated training            │
        │                                          │
        │ 5. Monitor via web UI                    │
        │    → See test RMSE/MAE per round         │
        │    → See global model performance        │
        │                                          │
        │ 6. Results saved to results/ folder      │
        │    → Model checkpoints                   │
        │    → CSV with metrics                    │
        └──────────────────────────────────────────┘
```

### Detailed File-by-File Changes

#### 1. **New File: `tawos_feature_extractor.py`**
```python
# Download TAWOS, connect to MySQL database
# Extract features using SQL queries:
#   - Issue: description_length, type (bug/feature)
#   - Comments: total_count, avg_sentiment
#   - Components: count
#   - Links: count
#   - Time: resolution_time_min, in_progress_time_min
#   - People: num_assignee_changes, num_reporters
#   - Change Log: num_status_changes, num_priority_changes
#   - Target: Story_Point (actual label)
# Output: CSV with 450K rows × 20 features
```

#### 2. **New File: `tawos_data_loader.py`**
```python
# Load CSV created by extractor
# Partition strategy:
#   Worker_0: Apache Mesos issues (~5000)
#   Worker_1: Apache MXNet issues (~6000)
#   ...
#   Worker_14: Last project issues
# Create DataLoader for each worker (regression task)
```

#### 3. **New File: `federated_learning/nets/tawos_mlp.py`**
```python
class TAWOSMLP(torch.nn.Module):
    def __init__(self, input_features=20):
        super().__init__()
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Single output for regression
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Linear output for regression
        return x
```

#### 4. **Modify: `federated_learning/arguments.py`**
```python
# Add new dataset option
def set_dataset(self, dataset_name):
    if dataset_name == "tawos":
        self.net = TAWOSMLP
        self.loss_function = torch.nn.MSELoss  # REGRESSION
        self.train_data_loader_pickle_path = "data_loaders/tawos/train_data_loader.pickle"
        # ... etc
```

#### 5. **Modify: `client.py`**
```python
# Handle regression evaluation differently
# Change test() method:
#   - Instead of accuracy, compute RMSE and MAE
#   - Adjust loss function call for 1D output
```

#### 6. **Modify: `server.py`**
```python
# Minimal changes - same FedAvg logic works for regression
# Just update epoch evaluation to show RMSE instead of accuracy
```

#### 7. **Modify: `api_server.py`**
```python
# Add TAWOS dataset option to API endpoints
# Add request parameter: dataset="tawos"
# Frontend can select dataset from dropdown
```

---

### Step-by-Step Implementation Guide

**Phase 1: Data Preparation (Day 1)**
1. Download TAWOS database (SQL file)
2. Create `tawos_feature_extractor.py`
3. Extract features → `data/tawos_features.csv`
4. Verify: Check CSV has 450K rows, no missing values

**Phase 2: Data Loaders (Day 1-2)**
1. Create `tawos_data_loader.py`
2. Partition CSV by 15 major projects
3. Create DataLoaders for each worker
4. Pickle and save to `data_loaders/tawos/`

**Phase 3: Model (Day 2)**
1. Create `TAWOSMLP` model in `federated_learning/nets/`
2. Test model shape: (batch=32, input=20) → (batch=32, output=1)
3. Verify forward pass works

**Phase 4: Framework Integration (Day 2-3)**
1. Update `arguments.py` to support TAWOS dataset
2. Update `client.py` evaluation for regression (MSE/RMSE/MAE)
3. Update `server.py` logging for regression metrics
4. Update `api_server.py` with TAWOS endpoints

**Phase 5: Testing (Day 3-4)**
1. Run via CLI: `python label_flipping_attack.py` with TAWOS config
2. Run via API: POST to `/run-experiment` with `dataset="tawos"`
3. Check results in `results/` folder
4. Verify FedAvg aggregation works (model improves over rounds)

**Phase 6: Validation (Day 4-5)**
1. Train global model for 50 rounds
2. Check RMSE decreases legitimately
3. Compare: 15-worker federated vs single global model
4. Ensure federating is better than baseline

---

### Expected Results & Metrics

**Output Metrics (saved as CSV):**
- Per epoch: Global RMSE, Global MAE, Test Loss
- Per worker: Local RMSE, Training Loss
- Worker selection: Which workers participated each round

**Graph Examples:**
```
Round → Global RMSE (decreasing)
Round → Global MAE (decreasing)
Worker_0 → Local Loss (noisy but trending down)
Worker_1 → Local Loss (different pattern - non-IID!)
```

**Success Criteria:**
- ✅ RMSE improves over 50 rounds (e.g., 5.0 → 3.2)
- ✅ Federated model RMSE < baseline (single global model)
- ✅ Each worker convergence visible in logs
- ✅ Non-IID effect visible (workers have different local optima)

---

### Complexity Comparison

| Aspect | Current (CNN/CIFAR) | New (MLP/TAWOS) | Difficulty |
|--------|-------------------|-----------------|-----------|
| Data shape | Images (28×28) | Tabular (20 features) | ✅ Easier |
| Output | 10 classes | 1 continuous value | ✅ Simpler |
| Model size | ~500k params | ~20k params | ✅ Lighter |
| Loss function | CrossEntropy | MSE | ✅ Simpler |
| Metrics | Accuracy/F1 | RMSE/MAE | ✅ Standard |
| Data distribution | IID batches | Project-based (non-IID) | 🟠 New pattern |

---

### Summary: What You're Really Doing

**In essence:**
You're swapping the **"What class is this image?"** problem with **"What story points for this issue?"** problem. The federated learning machinery stays exactly the same - it's just learning on different data with a different model architecture.

**Key insight:** If FedAvg works for image classification across 50 workers, it will work for SP estimation across 15 workers. No algorithmic changes needed - just plumbing.
