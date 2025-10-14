# Study of Label Flipping Attack on Federated Learning Systems

![image](https://user-images.githubusercontent.com/53596227/148281148-8f85bab3-d7a3-407f-8ce5-a7e8747e4ac3.png)

# How to Run Instructions

## Prerequisites
- Python 3.8+
- Node.js 20.19+ or 22.12+
- Git

## 1. Setup Environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

## 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.pip
```

## 3. Setup Frontend
```bash
cd frontend
npm install
cd ..
```

## 4. Generate Initial Data and Models (One-time setup)
```bash
python generate_data_distribution.py
python generate_default_models.py
```

## 5. Run Web Interface (Recommended)

### Terminal 1 - Start Backend API Server:
```bash
python api_server.py
```

### Terminal 2 - Start Frontend:
```bash
cd frontend
npm run dev
```

### Access the application:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000

## 6. Run Command Line Experiments (Alternative)
```bash
python label_flipping_attack.py
python attack_timing.py
python malicious_participant_availability.py
python defense.py
```

## File Structure
- `api_server.py` - Main FastAPI backend server
- `server.py` - Core federated learning server functions
- `frontend/` - React web interface
- `results/` - Experiment results and CSV files
- `logs/` - Experiment logs
- `federated_learning/` - Core FL modules

### Experiment Hyperparameters

Recommended default hyperparameters for CIFAR10 (using the provided CNN):
- Batch size: 10
- LR: 0.01
- Number of epochs: 200
- Momentum: 0.5
- Scheduler step size: 50
- Scheduler gamma: 0.5
- Min_lr: 1e-10

Recommended default hyperparameters for Fashion-MNIST (using the provided CNN):
- Batch size: 4
- LR: 0.001
- Number of epochs: 200
- Momentum: 0.9
- Scheduler step size: 10
- Scheduler gamma: 0.1
- Min_lr: 1e-10


