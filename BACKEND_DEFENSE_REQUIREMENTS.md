# Backend Requirements for Defense Visualization

## Overview
To support the new frontend defense visualization features, you need to make these changes to the backend.

## 1. Update API Server (`api_server.py`)

### Add Defense Configuration to ExperimentConfig Model

```python
class ExperimentConfig(BaseModel):
    dataset: str = "fashion_mnist"
    num_poisoned_workers: int = 0
    replacement_method: str = "default_no_change"
    selection_strategy: str = "RandomSelectionStrategy"
    workers_per_round: int = 5
    quick_mode: bool = True
    enable_defense: bool = False  # NEW
    defense_method: str = "byzantine_robust"  # NEW
    kwargs: Dict = {"NUM_WORKERS_PER_ROUND": 5}
```

### Add Defense Method Mappings

```python
DEFENSE_METHODS = {
    "byzantine_robust": {
        "function": None,  # Implement in federated_learning/utils/defense.py
        "description": "Byzantine-robust aggregation using median/trimmed mean"
    },
    "anomaly_detection": {
        "function": None,
        "description": "Statistical anomaly detection on gradients"
    },
    "gradient_clipping": {
        "function": None,
        "description": "Clip gradients to limit attack impact"
    },
    "client_filtering": {
        "function": None,
        "description": "Filter suspicious clients based on behavior"
    }
}
```

### Update `execute_experiment` Function

Modify the experiment execution to apply defense when `enable_defense=True`:

```python
def execute_experiment(exp_id: str, config: ExperimentConfig):
    try:
        # ... existing code ...
        
        # NEW: Apply defense if enabled
        if config.enable_defense:
            defense_fn = DEFENSE_METHODS[config.defense_method]["function"]
            # Apply defense logic during aggregation
            # This will be integrated in server.py
        
        # ... rest of existing code ...
```

### Update Results Structure

Add defense metrics to experiment results:

```python
final_results = {
    "results": {
        "epochs": list(range(1, len(results) + 1)),
        "accuracy": [r[0] for r in results],
        "loss": [r[1] for r in results],
        # ... existing fields ...
    },
    "worker_selection": worker_selections,
    "poisoned_workers": list(range(min(config.num_poisoned_workers, 30))),
    "config": config.dict(),
    "defense_stats": {  # NEW
        "enabled": config.enable_defense,
        "method": config.defense_method if config.enable_defense else None,
        "blocked_updates": 0,  # Count of blocked malicious updates
        "filtered_workers": [],  # List of workers filtered by defense
    } if config.enable_defense else None,
    # ... existing metadata ...
}
```

## 2. Update Server Orchestration (`server.py`)

### Modify `train_subset_of_clients` Function

Add defense logic to the aggregation step:

```python
def train_subset_of_clients(epoch, args, clients, poisoned_workers):
    """
    Train a subset of clients per round with optional defense.
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    random_workers = args.get_round_worker_selection_strategy().select_round_workers(
        list(range(args.get_num_workers())),
        poisoned_workers,
        kwargs)

    for client_idx in random_workers:
        args.get_logger().info("Training epoch #{} on client #{}", str(epoch), str(clients[client_idx].get_client_index()))
        clients[client_idx].train(epoch)

    args.get_logger().info("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]
    
    # NEW: Apply defense mechanism if enabled
    if args.get_enable_defense():
        defense_method = args.get_defense_method()
        if defense_method == "byzantine_robust":
            from federated_learning.utils.defense import byzantine_robust_aggregation
            new_nn_params = byzantine_robust_aggregation(parameters, poisoned_workers)
        elif defense_method == "gradient_clipping":
            from federated_learning.utils.defense import clip_and_aggregate
            new_nn_params = clip_and_aggregate(parameters)
        else:
            new_nn_params = average_nn_parameters(parameters)
    else:
        new_nn_params = average_nn_parameters(parameters)

    for client in clients:
        args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        client.update_nn_parameters(new_nn_params)

    return clients[0].test(), random_workers
```

## 3. Create Defense Module (`federated_learning/utils/defense.py`)

Create a new file with defense implementations:

```python
import torch
import numpy as np
from loguru import logger

def byzantine_robust_aggregation(parameters, poisoned_workers=None):
    """
    Byzantine-robust aggregation using coordinate-wise median.
    """
    # Stack all parameters
    all_params = []
    for param_dict in parameters:
        flattened = []
        for key in sorted(param_dict.keys()):
            flattened.append(param_dict[key].flatten())
        all_params.append(torch.cat(flattened))
    
    # Stack into tensor
    stacked = torch.stack(all_params)
    
    # Compute coordinate-wise median
    median_params = torch.median(stacked, dim=0).values
    
    # Reconstruct parameter dict
    result = {}
    offset = 0
    for key in sorted(parameters[0].keys()):
        shape = parameters[0][key].shape
        size = parameters[0][key].numel()
        result[key] = median_params[offset:offset+size].reshape(shape)
        offset += size
    
    return result

def clip_and_aggregate(parameters, clip_norm=10.0):
    """
    Clip gradients and aggregate.
    """
    clipped_params = []
    for param_dict in parameters:
        clipped = {}
        for key, value in param_dict.items():
            norm = torch.norm(value)
            if norm > clip_norm:
                clipped[key] = value * (clip_norm / norm)
            else:
                clipped[key] = value
        clipped_params.append(clipped)
    
    # Average the clipped parameters
    from federated_learning.utils import average_nn_parameters
    return average_nn_parameters(clipped_params)

def detect_anomalies(parameters, threshold=2.0):
    """
    Detect anomalous updates using statistical methods.
    Returns indices of suspicious parameters.
    """
    suspicious = []
    
    # Compute mean and std of parameter norms
    norms = []
    for param_dict in parameters:
        total_norm = 0
        for value in param_dict.values():
            total_norm += torch.norm(value).item() ** 2
        norms.append(total_norm ** 0.5)
    
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    
    # Flag outliers
    for idx, norm in enumerate(norms):
        z_score = abs((norm - mean_norm) / std_norm) if std_norm > 0 else 0
        if z_score > threshold:
            suspicious.append(idx)
    
    return suspicious
```

## 4. Update Arguments Class (`federated_learning/arguments.py`)

Add defense configuration fields:

```python
class Arguments:
    def __init__(self, logger):
        # ... existing fields ...
        
        # Defense configuration
        self.enable_defense = False
        self.defense_method = "byzantine_robust"
    
    def set_enable_defense(self, enable):
        self.enable_defense = enable
    
    def get_enable_defense(self):
        return self.enable_defense
    
    def set_defense_method(self, method):
        self.defense_method = method
    
    def get_defense_method(self):
        return self.defense_method
```

## 5. Update `run_exp` Function (`server.py`)

Wire defense configuration from API to arguments:

```python
def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx, dataset="fashion_mnist", enable_defense=False, defense_method="byzantine_robust"):
    # ... existing setup ...
    
    args = Arguments(logger)
    args.set_dataset(dataset)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    
    # NEW: Set defense configuration
    args.set_enable_defense(enable_defense)
    args.set_defense_method(defense_method)
    
    args.log()
    
    # ... rest of existing code ...
```

## 6. Update API Endpoints

### Add Defense Configuration Endpoint

```python
@app.get("/api/defenses")
async def get_defense_methods():
    """Get available defense methods"""
    return {
        "defense_methods": [
            {"value": k, "label": k.replace("_", " ").title(), "description": v["description"]}
            for k, v in DEFENSE_METHODS.items()
        ]
    }
```

## Summary of Changes

1. **api_server.py**: Add `enable_defense` and `defense_method` to ExperimentConfig, add DEFENSE_METHODS mapping, update execute_experiment
2. **server.py**: Modify train_subset_of_clients to apply defense, update run_exp signature
3. **federated_learning/utils/defense.py**: Create new file with defense functions
4. **federated_learning/arguments.py**: Add defense configuration fields
5. **federated_learning/utils/__init__.py**: Export defense functions

## Testing the Implementation

Run two experiments:
1. Baseline (no defense): `enable_defense=False`
2. With defense: `enable_defense=True, defense_method="byzantine_robust"`

The frontend will automatically compare and visualize the difference.

## Next Steps

1. Implement the defense functions in `federated_learning/utils/defense.py`
2. Update `api_server.py` with the new config fields
3. Modify `server.py` to use defense during aggregation
4. Test with the existing frontend (already updated)
5. Run experiments and view defense effectiveness in the UI
