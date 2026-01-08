Chest Cancer Prediction – End-to-End MLOps Pipeline
===================================================

This repository contains a production-oriented MLOps pipeline for chest-cancer image classification using chest CT scans.
The solution is implemented in TensorFlow/Keras (VGG16 backbone) and is designed to be:

- Configuration-driven (via YAML files and typed config entities)
- Modular and reproducible (explicit components and pipeline stages)
- Observable (evaluation metrics and models logged to MLflow / Dagshub)


Overview
--------

The pipeline automates the following high-level steps:

1. **Data ingestion** – Downloading and unpacking the chest CT scan dataset into a well-defined artifacts structure.
2. **Base model preparation** – Loading VGG16 with ImageNet weights, attaching a classification head, and exporting reusable base models.
3. **Model training** – Fine-tuning the model on the ingested dataset with configurable hyperparameters.
4. **Model evaluation & logging** – Evaluating the trained model on a validation split and logging results and models to MLflow.

The same logic is available both as Python modules under `src/cnnClassifier` and as research notebooks under `research/` for experimentation.


Key Features
------------

- Clear separation of concerns between configuration, components, and pipeline orchestration.
- Typed configuration entities using Pydantic dataclasses for safer configuration management.
- Config-driven hyperparameters (`params.yaml`) for image size, batch size, epochs, augmentation, and learning rate.
- MLflow / Dagshub integration for metric logging and model registry (VGG16-based model).
- Artifacts-first design: each stage reads from and writes to the `artifacts/` directory, enabling reproducibility and downstream automation.


Technology Stack
----------------

- **Language & runtime**: Python
- **Deep learning**: TensorFlow / Keras (VGG16)
- **Configuration**: YAML (`config/config.yaml`, `params.yaml`) + Pydantic dataclasses
- **Experiment tracking**: MLflow, hosted on Dagshub
- **Orchestration**: Python pipeline scripts (`src/cnnClassifier/pipeline`) and `main.py`


Project Structure
-----------------

- `config/`
  - `config.yaml` – High-level configuration for artifacts, data ingestion, base model paths, and training outputs.
- `params.yaml` – Model and training hyperparameters (image size, batch size, epochs, learning rate, etc.).
- `src/cnnClassifier/`
  - `config/` – `configuration.py` builds typed config objects from YAML files.
  - `entity/` – Pydantic dataclasses for configuration entities (data ingestion, base model, training, evaluation).
  - `components/` – Reusable building blocks:
    - `data_ingestion.py`
    - `prepare_base_model.py`
    - `model_trainer.py`
    - `model_evaluation.py` (evaluation + MLflow logging)
  - `pipeline/` – Orchestrated training stages:
    - `stage_01_data_ingestion.py`
    - `stage_02_prepare_base_model.py`
    - `stage_03_model_trainer.py`
    - `stage_04_model_evaluation.py`
- `research/` – Jupyter notebooks that mirror the production pipeline logic for experimentation.
- `artifacts/` – Auto-created directory for downloaded data, intermediate models, and the final trained model.
- `main.py` – Entry point that runs the full pipeline end-to-end.
- `dvc.yaml` – Placeholder for DVC pipeline definitions (optional, for data and pipeline versioning).


Getting Started
---------------

### 1. Clone the repository

Clone this project and switch into the project directory.

### 2. Create and activate a virtual environment

It is recommended to isolate dependencies using a virtual environment (example using `venv`):

- Create and activate a virtual environment in the project root (for example, named `chest-cancer`).

### 3. Install dependencies

From the project root, install the Python dependencies:

- `pip install -r requirements.txt`

### 4. Configure the project

- Update `config/config.yaml` to reflect:
  - Artifact locations (for example, `artifacts/...` paths)
  - Data ingestion source URL and paths
  - Base model and training model output locations
- Update `params.yaml` to tune:
  - `IMAGE_SIZE`
  - `BATCH_SIZE`
  - `EPOCHS`
  - `AUGMENTATION`
  - `LEARNING_RATE`


Running the Pipeline
--------------------

To execute the full training workflow end-to-end, run from the project root:

- `python main.py`

This triggers the following stages in sequence:

1. **Data Ingestion (`stage_01_data_ingestion.py`)**
   - Downloads the chest CT scan dataset.
   - Stores the raw archive and extracted files under `artifacts/data_ingestion`.
2. **Prepare Base Model (`stage_02_prepare_base_model.py`)**
   - Loads VGG16 with ImageNet weights.
   - Attaches the classification head and compiles the model.
   - Saves both the base and updated models under `artifacts/prepare_base_model`.
3. **Model Training (`stage_03_model_trainer.py`)**
   - Loads the updated base model.
   - Builds training and validation generators from the ingested dataset.
   - Trains the model and saves the final weights to `artifacts/training/model.h5`.
4. **Model Evaluation (`stage_04_model_evaluation.py`)**
   - Loads the trained model.
   - Evaluates it on the validation split.
   - Persists evaluation metrics to `scores.json` and can log results to MLflow.

Each stage script can also be executed independently for debugging or development by running the corresponding `stage_0X_*.py` module.


MLflow and Dagshub Integration
------------------------------

- The evaluation configuration (`ConfigurationManager.get_evaluation_config`) sets `mlflow_uri` to the Dagshub MLflow endpoint:
  - `https://dagshub.com/Astrrokid/chest-cancer-prediction.mlflow`
- The `Evaluation.log_into_mlflow()` method in `src/cnnClassifier/components/model_evaluation.py`:
  - Sets the MLflow registry URI.
  - Logs parameters (from `params.yaml`) and evaluation metrics (`loss`, `accuracy`).
  - Logs and, when supported by the backend store, registers the trained Keras model (`VGG16Model`) in the MLflow Model Registry.
- To ensure runs and models appear in the expected MLflow UI:
  - Confirm that `evaluation.log_into_mlflow()` is invoked from `stage_04_model_evaluation.py`.
  - Ensure your `MLFLOW_TRACKING_URI` or configuration matches the Dagshub MLflow URL you are monitoring.


Development Workflow
--------------------

When extending or modifying the project, the recommended workflow is:

1. Update `config/config.yaml` with new artifact paths or data sources as needed.
2. Optionally update secrets configuration for credentials (for example, `secrets.yaml` if used).
3. Update `params.yaml` to reflect new or changed hyperparameters.
4. Update configuration entities in `src/cnnClassifier/entity/` if you add or modify configuration fields.
5. Update the configuration manager in `src/cnnClassifier/config/configuration.py` to construct and expose the revised entities.
6. Implement or adjust components under `src/cnnClassifier/components/` to introduce new behavior.
7. Wire components together in pipeline stages under `src/cnnClassifier/pipeline/`.
8. Update `main.py` if you add, remove, or reorder pipeline stages.
9. Optionally update `dvc.yaml` to version data and pipeline steps with DVC.

This approach keeps the codebase modular, testable, and easy to reason about as requirements evolve.


Notebooks
---------

The `research/` directory contains Jupyter notebooks that support experimentation and rapid prototyping:

- `01_data_ingestion.ipynb` – Prototyping and validating the data ingestion step.
- `02_prepare_base_model.ipynb` – Exploring base model creation and fine-tuning strategies.
- `03_model_trainer.ipynb` – Prototyping training, data loaders, and training loops.
- `04_model_evaluation_with_mlflow.ipynb` – Prototyping evaluation and MLflow / Dagshub integration.

These notebooks closely mirror the production pipeline logic and are useful for debugging, exploratory analysis, and communicating design choices before formalizing changes into the `src/` codebase.

