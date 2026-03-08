# FL Security Research Platform — In-Depth Developer Guide

This repository implements a research and demo platform for federated learning (FL) security experiments. The project focuses on label-flipping attacks, multiple client selection strategies, experiment orchestration, lightweight defenses and visualization. It provides:

- A FastAPI backend that queues and executes experiments (real FL runs or a simulation fallback).
- A research-quality FL runtime implemented in Python under `federated_learning/`.
- A React frontend that lets users configure and run experiments and inspect results.
- Command-line scripts for running targeted experiments and analysis.

This README is written to give a software engineer (with no prior context) a complete, in-depth understanding of the repository: architecture, internals, API, data formats, security considerations, and development workflows.

Table of contents
- Overview
- Quick Start (development)
- Architecture & Data Flow
- Backend internals (detailed file-by-file)
- Federated learning package deep dive
- Frontend overview
- API Reference (endpoints, request/response)
- Experiment lifecycle and artifacts
- Security and hardening guidance
- Testing, debugging, and development tips
- Recommended improvements and roadmap
- Contributing

--------------------------------------------------------------------------------

Overview
--------

This project is intended for experimentation and research into poisoning attacks in FL (primarily label-flipping). Two modes of operation are supported:

- Simulation mode — a fast, deterministic-ish simulator used for demos when FL dependencies aren't available.
- Real FL mode — runs a simplified federated learning pipeline that distributes data to clients, applies poisoning to selected clients, trains client models, aggregates updates, and saves model/results artifacts.

The backend API orchestrates experiments and stores results in `results/` and logs in `logs/`. The frontend provides a point-and-click UI for launching experiments and inspecting results.

Quick start (development)
-------------------------

1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

2. Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.pip
```

3. Install frontend dependencies and run dev server:

```bash
cd frontend
npm install
npm run dev
```

4. (One-time) Generate data and default models for experiments:

```bash
python generate_data_distribution.py
python generate_default_models.py
```

5. Start the backend API (development):

```bash
python api_server.py
```

6. Frontend (if not already running) lives at http://localhost:5173 and the backend default API is at http://localhost:8000.

Architecture & data flow
------------------------

High-level components
- `api_server.py` — FastAPI application exposing REST endpoints to queue experiments, query status and results, and run image classification against trained models.
- `server.py` — Orchestration layer which loads data, distributes it across simulated clients, applies poisoning, creates `Client` objects (in `client.py`), runs local training, aggregates parameters, and persists results.
- `federated_learning/` — Core implementation of FL logic: argument parsing and configuration, model architectures, dataset loaders, poisoning utilities, worker-selection strategies, dimensionality reduction helpers, and serialization helpers.
- `frontend/` — React UI and API client.
- `results/`, `logs/` — Persistent output artifacts from runs.

Data flow (experiment run)
1. Client (frontend or curl) POSTs a JSON `ExperimentConfig` to `/api/run`.
2. `api_server.py` validates the config, creates a unique `exp_id`, stores an entry in the in-memory `experiments` dict, and schedules execution with a `ThreadPoolExecutor`.
3. Background worker invokes either `server.run_exp` (real) or a simulation routine in `api_server.py` depending on availability of the `federated_learning` modules.
4. `server.run_exp` loads data-loaders, distributes batches to `num_workers`, selects `poisoned_workers`, modifies their labels (via `poison_data`), creates per-client data loaders and `Client` objects, runs training across epochs, averages parameters, tests aggregated model, and saves artifacts.
5. Clients (UI) poll `/api/status/{exp_id}` and finally GET `/api/results/{exp_id}` to fetch JSON results and CSVs for visualization.

Backend internals — file-by-file deep dive
----------------------------------------

This section lists the most important files and explains their purpose and key behaviors.

- `api_server.py`
  - FastAPI app which implements the public API.
  - Main responsibilities:
    - Expose endpoints: `/api/run`, `/api/status/{exp_id}`, `/api/results/{exp_id}`, `/api/experiments`, `/api/config`, `/api/classify/{exp_id}`.
    - Maintain an in-memory `experiments` dictionary with `experiment_lock` protecting concurrent mutation.
    - Create experiments (unique `exp_id`), schedule execution with `ThreadPoolExecutor(max_workers=3)`.
    - Provide a simulation fallback when `federated_learning` imports fail. This is useful for demos or CI without heavy ML deps.
    - Provide multiple model-loading and image preprocessing helper functions (note: some duplication exists — see "improvements").
  - Important behaviors to understand:
    - `ExperimentConfig` and `ExperimentResponse` are Pydantic models used for validation.
    - Experiment artifacts are written to `results/experiment_results_{exp_id}.json` and CSVs.
    - `load_experiment_model` and `create_trained_{dataset}_model` load models (often via `torch.load`) — torch's load uses pickle under the hood.

- `server.py`
  - Implements `run_exp(...)` which is the core FL orchestration for a single experiment id.
  - Steps performed in `run_exp`:
    1. Generate file paths and logger via `generate_experiment_ids`.
    2. Instantiate `Arguments` to encapsulate hyperparams.
    3. Load train and test data loaders using utilities in `federated_learning.utils`.
    4. Distribute training batches equally to `num_workers` using `distribute_batches_equally`.
    5. Identify `poisoned_workers`, call `poison_data` to modify the dataset labels for those workers.
    6. Build per-client data loaders and `Client` objects.
    7. Run training across multiple epochs while selecting a subset of clients per round.
    8. Average parameters (via `average_nn_parameters`), test the aggregated model, and write results to CSV/JSON and save final model to `results/{exp_id}_final_model.pth`.

- `federated_learning/arguments.py`
  - A programmatic configuration object. It provides getters/setters for dataset paths, batch sizes, number of workers, scheduler parameters, training epochs and more.
  - Used across `server.py` and `Client` to avoid passing many parameters.

- `federated_learning/worker_selection/` (directory)
  - Implements different strategies for selecting clients at each round:
    - `random.py` — RandomSelectionStrategy: simple uniform random selection.
    - `breakpoint_before.py`, `breakpoint_after.py` — temporal bias strategies favoring early/late epochs.
    - `poisoner_probability.py` — probabilistically biases toward selecting poisoned workers.
  - Selection strategy implementations are instantiated and provided through the API config.

- `federated_learning/nets/` (directory)
  - Model architecture definitions used by experiments (FashionMNIST CNN, CIFAR10 CNN). Each file exposes a PyTorch `nn.Module` class. These are loaded by experiments and by the classification endpoint.

- `federated_learning/utils/` (directory)
  - `poison_data.py` — applies label replacement attacks (calls `apply_class_label_replacement`). This is the core poisoning mechanism.
  - `file_storage_utils.py`, `data_loader_utils.py` — helpers for saving/loading CSVs, pickled data loaders.
  - `tensor_converter.py`, `model_list_parser.py` — helpers for converting tensors and parsing model filenames.
  - `generate_data_loaders_from_distributed_dataset` — builds DataLoader objects from distributed arrays.

- `client.py`
  - Implements the `Client` abstraction: local training, model parameter extraction (`get_nn_parameters()`), updating parameters from the server, and local testing.

- `label_flipping_attack.py`, `malicious_participant_availability.py`, `attack_timing.py`, `defense.py`
  - CLI scripts and experiments that demonstrate various attacks and analyses. They are intended to be run locally for research runs without the web UI.

Federated learning package deep dive
-----------------------------------

The `federated_learning` package implements the FL runtime. The most important concepts and helper functions are:

- Dataset distribution: the training dataset is divided into per-client subsets with `distribute_batches_equally`.
- Poisoning: `poison_data` replaces labels in selected clients' datasets according to a replacement function (e.g., `replace_1_with_9`). Replacement functions are small helpers that map labels.
- Client lifecycle: each `Client` holds a local model and dataset. `train(epoch)` performs local updates on the client's data. After selected clients train, server aggregates parameters via `average_nn_parameters` and distributes new parameters to the clients.
- Selection strategies: pluggable strategies allow experiments to investigate how client selection affects poisoning success.

Frontend overview
-----------------

- React app located in `frontend/`.
- `frontend/src/services/api.js` is a thin client that wraps `fetch` for the backend API. It uses `http://localhost:8000/api` as `API_BASE_URL` during development — consider replacing with an env-configured URL for production.
- UI components:
  - `ExperimentForm.jsx` — configuration UI for experiment parameters.
  - `ResultsView.jsx` — rendering experiment charts and CSV data.
  - `DefenseView.jsx` — visualization to compare defense vs baseline.

API Reference (detailed)
------------------------

All API endpoints are mounted under `/api`.

1) POST /api/run
- Purpose: Queue a new experiment.
- Request body: JSON matching `ExperimentConfig` Pydantic model:
  - `dataset` (string): "fashion_mnist" or "cifar10" (default: fashion_mnist)
  - `num_poisoned_workers` (int): number of clients whose labels will be poisoned
  - `replacement_method` (string): key in `ATTACK_METHODS` mapping (e.g. "replace_1_with_9")
  - `selection_strategy` (string): key in `SELECTION_STRATEGIES`
  - `workers_per_round` (int)
  - `quick_mode` (bool): if true, uses fewer epochs / faster simulation
  - `kwargs` (object): additional runtime kwargs (e.g. `NUM_WORKERS_PER_ROUND`)
- Response: `ExperimentResponse` with `exp_id`, `status`="submitted".

Notes: the server will raise 409 if another experiment is currently running — the implementation enforces single active experiment at a time.

2) GET /api/status/{exp_id}
- Purpose: query experiment progress.
- Response: `StatusResponse` containing status (submitted/running/done/error), progress (0.0–1.0), `current_epoch`, and `total_epochs`.

3) GET /api/results/{exp_id}
- Purpose: fetch full experiment results. If the experiment is in-memory, returns the results object. If not, attempts to load `results/experiment_results_{exp_id}.json` or legacy CSV formats.
- Response: results JSON with fields:
  - `results` -> `epochs`, `accuracy`, `loss`, `per_class_precision`, `per_class_recall`
  - `worker_selection` -> list of worker IDs selected per epoch
  - `raw_csv` -> CSV string for easy download

4) GET /api/experiments
- Purpose: list known experiments (in memory and files on disk). Returns metadata for each experiment (status, created_at, progress, basic config summary).

5) GET /api/config
- Purpose: return available attack methods and selection strategies (for UI dropdowns), and a `system_info` object with `fl_modules_available`.

6) POST /api/classify/{exp_id}
- Purpose: classify an uploaded image using the trained model for `exp_id` (or fallback default models).
- Request: multipart form with `file` argument (image) and optional `dataset` form field.
- Response: classification result with predicted class, confidence, and per-class probabilities.

Experiment lifecycle and artifacts
---------------------------------

- When an experiment runs (real or simulated) the system saves:
  - `results/experiment_results_{exp_id}.json` — canonical JSON describing results and metadata.
  - `results/{exp_id}_final_model.pth` — final model weights saved via `torch.save`.
  - `results/{exp_id}_results.csv` and `results/{exp_id}_workers_selected.csv` — legacy CSV outputs.
  - `logs/{exp_id}.log` — run log for the experiment.

Files and formats
- JSON results schema: `results` (with arrays for `epochs`, `accuracy`, `loss`), `worker_selection` (list of lists), `config` (ExperimentConfig), `metadata` (execution_mode, total_epochs, attack method).
- Model files: saved via `torch.save(state_dict)` in many places. These files use PyTorch serialization (which relies on Python pickle).

Simulation vs Real mode
-----------------------

- The backend attempts to import `federated_learning.*` modules at startup. If import fails, `FL_MODULES_AVAILABLE` is set to `False` and `api_server.py` runs a simulation routine that produces realistic-looking results quickly.
- Simulation mode is useful for demoing the frontend or running CI without heavy ML dependencies.

Security, hardening, and operational concerns
--------------------------------------------

This repository is research-focused and currently configured for local development. The following items are critical for any deployment or multi-user environment.

1. Authentication & Authorization
   - The API exposes endpoints that can initiate heavy workloads. Add authentication (API key, JWT, or OAuth) to:
     - `/api/run` — starting experiments
     - `/api/classify/{exp_id}` — loading models
   - Add role-based access if multiple users share the service.

2. Network exposure
   - `api_server.py` currently starts uvicorn bound to `0.0.0.0` by default. For local-only development bind to `127.0.0.1`. For production, run behind a secure reverse proxy (TLS, authentication).

3. Model loading safety
   - `torch.load` deserializes Python objects via pickle — **loading arbitrary `.pth` files from untrusted sources is remote code execution risk**. Recommended mitigations:
     - Only load models produced by your system (validate filenames and paths).
     - Store a metadata JSON alongside models (including expected parameter shapes and keys). Verify state dict keys & shapes before calling `load_state_dict`.
     - Consider using `torch.jit` or saving raw tensor payloads that are parsed without pickle, or load into a separate sandbox process.

4. Path handling
   - `exp_id` values are used to construct file paths. Validate `exp_id` strictly (allow `[A-Za-z0-9_\\-]{1,64}`), reject slashes, and canonicalize paths with `Path.resolve()` ensuring files live under expected directories.

5. Resource and DoS protection
   - Experiments are CPU/GPU intensive and currently scheduled with an in-process `ThreadPoolExecutor`. For multi-user/prod:
     - Move experiment execution to a dedicated worker queue (Celery, RQ, or Kubernetes Jobs).
     - Enforce quotas and concurrency limits per user.
     - Add rate-limiting to public API surface.

6. Logging and error handling
   - Avoid returning stack traces to unauthenticated clients. Return sanitized error messages. Persist detailed logs to `logs/` with log rotation.

Development, testing, and debugging tips
--------------------------------------

- Use the simulation mode for fast iteration of the frontend and basic API tests.
- To run a full FL experiment locally you need PyTorch + torchvision installed (see `requirements.pip`). Expect long runtimes depending on dataset and epochs.
- Unit testing:
  - Add unit tests around: `poison_data`, `average_nn_parameters`, `generate_experiment_ids`, `file_storage_utils` and model-loading validation.
  - `test_backend.py` currently provides a manual smoke test for the API — convert this into pytest-based automated tests.

Troubleshooting
---------------

- If `api_server.py` prints "Warning: Could not load FL modules" at startup, you're in simulation mode. Either install the required ML dependencies or run the simulation for UI testing.
- If model loading fails with `torch` errors, check that model and architecture match. Look at `results/` for saved files; use `torch.load(..., map_location='cpu')` in a Python shell to inspect keys.

Recommended immediate improvements (short roadmap)
-------------------------------------------------
1. Add `exp_id` validation and canonical path checks to all file reads/writes.
2. Harden model loading by validating state_dict keys & shapes before applying them to models.
3. Add a simple API key-based auth and bind to localhost by default.
4. Move experiment execution to an external worker (Celery/RQ) and persist queue metadata in Redis/DB.
5. Add unit tests and CI (GitHub Actions): linting, unit tests, and a simulation-mode smoke test.

Contributing
------------

Contributions are welcome. High-impact areas:
- Improving security (see recommended improvements)
- Adding comprehensive unit tests and CI
- Refactoring duplicate helper code in `api_server.py` into shared utilities
- Adding Dockerfiles for reproducible deployments

Suggested code style and tooling
- Python: use `black` or `ruff` for formatting and linting. Add `mypy` for type checks.
- Frontend: use `npm run lint` and a minimal Prettier config.

Contact and references
----------------------
For research context and examples, the UI links to: https://github.com/MinhajulBhuiyan/fl-security

--------------------------------------------------------------------------------

This README is intended to be a single-stop reference for engineers picking up the repository. If you'd like, I can also:

- Generate a contributor-oriented checklist and add GitHub Actions CI (lint, tests, simulation smoke test).
- Implement the immediate security hardening changes (exp_id validation + safe model loader).
- Add example Postman/curl requests and a short developer tutorial notebook that runs a sample experiment end-to-end.

Choose next steps by replying with one of the above or asking for a custom add-on.
