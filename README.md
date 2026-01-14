# End-to-End-Chest-Cancer-Prediction-using-MLflow-DVC

This repository implements an end-to-end MLOps pipeline for classifying chest CT scan images into cancer / non-cancer classes using TensorFlow/Keras (VGG16), MLflow for experiment tracking, DVC for data & pipeline versioning, and AWS + GitHub Actions for CI/CD and container deployment.


## Workflows

1. Update `config/config.yaml` (artifacts, data paths, MLflow settings).
2. [Optional] Update secrets configuration (for example `secrets.yaml` if you maintain credentials in a file).
3. Update `params.yaml` (model and training hyperparameters).
4. Update the config entities in `src/cnnClassifier/entity/`.
5. Update the configuration manager in `src/cnnClassifier/config/configuration.py`.
6. Update or create components in `src/cnnClassifier/components/`.
7. Update pipeline stages in `src/cnnClassifier/pipeline/`.
8. Update `main.py` if you add, remove, or reorder stages.
9. Update `dvc.yaml` to reflect the pipeline stages and dependencies.


## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow tutorial](https://youtube.com/playlist?list=PLkz_y24mlSJZrqiZ4_cLUiP0CBN5wFmTb&si=zEp_C8zLHt1DzWKK)

##### cmd
- `mlflow ui` (runs the local UI over the `mlruns/` directory)


### Dagshub

[Dagshub project](https://dagshub.com/Astrrokid/chest-cancer-prediction)  
Remote MLflow tracking URI: `https://dagshub.com/Astrrokid/chest-cancer-prediction.mlflow`

Run training with remote MLflow tracking:

```bash
MLFLOW_TRACKING_URI=https://dagshub.com/Astrrokid/chest-cancer-prediction.mlflow \
MLFLOW_TRACKING_USERNAME=<your_dagshub_username> \
MLFLOW_TRACKING_PASSWORD=<your_dagshub_token> \
python main.py
```

Export these as environment variables (Linux/macOS):

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/Astrrokid/chest-cancer-prediction.mlflow
export MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
export MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
```

On Windows PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/Astrrokid/chest-cancer-prediction.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<your_dagshub_username>"
$env:MLFLOW_TRACKING_PASSWORD="<your_dagshub_token>"
python main.py
```


### DVC cmd

1. `dvc init` – initialize DVC in the repository (run once).
2. `dvc repro` – reproduce the full pipeline defined in `dvc.yaml`.
3. `dvc dag` – visualize the pipeline: `data_ingestion → prepare_base_model → training → evaluation`.


## About MLflow & DVC

MLflow

- Production-grade experiment tracker and model registry.
- Tracks all chest-cancer experiments, parameters, metrics, and artifacts.
- Logs and (optionally) registers the VGG16-based classification model.


DVC 

- Lightweight experiment and data versioning; ideal for this POC and future extensions.
- Tracks dataset and model artifacts under `artifacts/`.
- Orchestrates pipeline stages via `dvc.yaml` (data ingestion, base model preparation, training, evaluation).


# AWS-CICD-Deployment-with-Github-Actions

This project ships a Dockerized FastAPI app (`app.py`) that exposes training and prediction endpoints and is deployed to AWS using GitHub Actions, Amazon ECR, and an EC2-based self-hosted runner.


## 1. Login to AWS console

Use an AWS account with permissions to manage IAM, ECR, and EC2.


## 2. Create IAM user for deployment

With specific access:

1. EC2 access – to manage the virtual machine used as a self-hosted runner / app host.
2. ECR – Elastic Container Registry access to push and pull the Docker image for this project.

Recommended policies:

1. `AmazonEC2ContainerRegistryFullAccess`
2. `AmazonEC2FullAccess`


## 3. Create ECR repo to store/save Docker image

Create an ECR repository dedicated to this project (for example `chest-cancer-prediction`) and save the URI, e.g.:

- `<AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/chest-cancer-prediction`


## 4. Create EC2 machine (Ubuntu)

Provision an Ubuntu EC2 instance that will:

- Host the Dockerized chest-cancer prediction app.
- Act as a self-hosted GitHub Actions runner for the deployment job.


## 5. Open EC2 and install Docker in EC2 machine

Optional:

```bash
sudo apt-get update -y
sudo apt-get upgrade -y
```

Required:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```


## 6. Configure EC2 as self-hosted runner

In your GitHub repository:

- Go to `Settings > Actions > Runners > New self-hosted runner`.
- Choose OS (Linux) and architecture.
- Run the provided commands on the EC2 instance.

The deployment job in `.github/workflows/main.yaml` uses `runs-on: self-hosted`.


## 7. Setup GitHub secrets

Configure the following secrets under `Settings > Secrets and variables > Actions`:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (e.g. `us-east-1`)
- `AWS_ECR_LOGIN_URI` (e.g. `<AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com`)
- `ECR_REPOSITORY_NAME` (e.g. `chest-cancer-prediction`)

The GitHub Actions workflow will:

- Build the Docker image from this repository (using `Dockerfile`, which runs `app.py` on port 8000).
- Push the image to the configured ECR repository.
- On the self-hosted EC2 runner, pull the latest image from ECR and run the container as `cnncls` with `-p 8000:8000`.

