# EV Battery Health Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen)

**Goal:** Predict the State of Health (SOH) of EV batteries (a continuous value) and classify each prediction into health buckets:
- `Healthy (≥85%)`
- `Moderate (70–84%)`
- `End-of-Life (<70%)`

This repository demonstrates a practical MLOps workflow:
data generation → cleaning → schema checks → train/val/test split → model training (RandomForest) → evaluation (regression + derived buckets) → experiment tracking with MLflow → optional API serving with FastAPI → containerization with Docker.

---

## Project Highlights

- Built a complete **MLOps pipeline** for EV battery health prediction using synthetic datasets  
- Implemented **RandomForest regression** to predict SOH and classify results into health buckets  
- Logged parameters, metrics, and models with **MLflow** for experiment tracking and reproducibility  
- Designed an optional **FastAPI service** and **Docker container** to demonstrate real-world deployment  
- Organized codebase with modular structure for **data, models, pipelines, and serving**

---

## Project Structure

```text
EV-Battery-Health-Prediction/
├── artifacts/              # Trained models and feature names
├── configs/                # Config files (params, schema)
├── data/                   # Data folders (raw, interim, processed)
├── notebooks/              # Jupyter notebooks for exploration/demo
├── src/
│   ├── data/               # Data generation, cleaning, checks, splits
│   ├── models/             # Training, evaluation, postprocess, metrics
│   ├── pipelines/          # End-to-end training pipeline
│   ├── serve/              # FastAPI app (for serving predictions)
│   └── utils/              # Utilities (seed, helpers)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Containerization
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Tech Stack

- **Python 3.12**
- **pandas** and **numpy** for data handling and synthetic dataset generation
- **scikit-learn** for machine learning models (RandomForest baseline)
- **MLflow** for experiment tracking, metrics logging, and model registry
- **FastAPI** for serving the model via REST API (optional deployment)
- **Docker** for containerization and reproducible environments
- **Jupyter Notebooks** for data exploration and demo visualizations

---

## Getting Started

### 1) Clone and set up a virtual environment

```bash
git clone https://github.com/<your-username>/ev-battery-health-prediction.git
cd ev-battery-health-prediction

# create & activate venv (macOS/Linux)
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
# python -m venv .venv
# .\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the training pipeline

This step generates synthetic data → cleans → checks → splits → trains → evaluates.

```bash
python -m src.pipelines.train_pipeline
```

### 4) Open MLflow UI

Track experiments, metrics, and artifacts in the browser.

```bash
python -m mlflow ui --backend-store-uri ./mlruns --port 5000
```

Then open: http://127.0.0.1:5000

### 5) (Optional) Quick inference from a saved run

Replace `<RUN_ID>` with the MLflow run ID shown in the UI.

```bash
python scripts/infer_from_run.py <RUN_ID>
```
### MLflow UI Example
![MLflow UI](./assets/images/mlflow_ui_metrics.png)


---

## Example Results

Validation metrics from a RandomForest baseline:

- **Train RMSE:** ~0.79
- **Val RMSE:** ~2.15
- **Test RMSE:** ~2.08
- **Test bucket accuracy** (derived from predicted SOH): ~94.9%

Derived bucket report:

```
              precision    recall  f1-score   support
         EOL      0.979     0.967     0.973
     Healthy      0.943     0.947     0.945
    Moderate      0.903     0.919     0.911
```

---

## Running the API locally (without Docker)

After training, you can serve the model using FastAPI + Uvicorn.

```bash
# make sure you have a trained model
python -m src.pipelines.train_pipeline

# start the API
uvicorn src.serve.app:app --reload
```

Now visit:

- **Root:** http://127.0.0.1:8000
- **Interactive docs:** http://127.0.0.1:8000/docs

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"cycle_count":500,"avg_temperature":30,"charge_rate":1.2,
       "discharge_rate":1.0,"depth_of_discharge":70,"internal_resistance":0.08}'
```

### FastAPI Docs Example Prediction Response
![Prediction Example](./assets/images/fast_api_real_time_prediction.png)

---

## Running with Docker

### Build the image:

```bash
docker build -t ev-battery .
```

### Train locally (creates artifacts/):

```bash
python -m src.pipelines.train_pipeline
```

### Run the container with the trained model mounted:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts:/app/artifacts:ro" \
  ev-battery
```

### Check health:

```bash
curl http://127.0.0.1:8000/health
```

### Call prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"cycle_count":500,"avg_temperature":30,"charge_rate":1.2,
       "discharge_rate":1.0,"depth_of_discharge":70,"internal_resistance":0.08}'
```

---

## Notebooks

- **notebooks/demo.ipynb** – explore the synthetic dataset, visualize battery degradation, and test model predictions interactively.

---

## Future Improvements

- Hyperparameter tuning (e.g., Optuna)
- Explore alternative models (Gradient Boosted Trees, XGBoost)
- Add CI/CD with GitHub Actions for testing and builds
- Deploy to cloud platforms (AWS/GCP/Azure) for serving

---

## Author

**Trina Ghosh**

- **LinkedIn:** https://linkedin.com/in/ghoshtrina
- **Portfolio:** http://ghoshtrina.com# ev-battery-health-prediction
