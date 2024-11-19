## Inspired by

Learn how to combine machine learning with software engineering to design, develop, deploy and iterate on production-grade ML applications.

- Lessons: https://madewithml.com/
- Code: [GokuMohandas/Made-With-ML](https://github.com/GokuMohandas/Made-With-ML)

<a href="https://madewithml.com/#course">
  <img src="https://madewithml.com/static/images/lessons.png" alt="lessons">
</a>

## Overview

In this course, we'll go from experimentation (design + development) to production (deployment + iteration). We'll do this iteratively by motivating the components that will enable us to build a *reliable* production system.

<br>

- **💡 First principles**: before we jump straight into the code, we develop a first principles understanding for every machine learning concept.
- **💻 Best practices**: implement software engineering best practices as we develop and deploy our machine learning models.
- **📈 Scale**: easily scale ML workloads (data, train, tune, serve) in Python without having to learn completely new languages.
- **⚙️ MLOps**: connect MLOps components (tracking, testing, serving, orchestration, etc.) as we build an end-to-end machine learning system.
- **🚀 Dev to Prod**: learn how to quickly and reliably go from development to production without any changes to our code or infra management.
- **🐙 CI/CD**: learn how to create mature CI/CD workflows to continuously train and deploy better models in a modular way that integrates with any stack.

## Set up

Be sure to go through the [course](https://madewithml/#course) for a much more detailed walkthrough of the content on this repository.

### Cluster

We'll start by setting up our cluster with the environment and compute configurations.

<details>
  <summary>Local</summary><br>
  Your personal laptop (single machine) will act as the cluster, where one CPU will be the head node and some of the remaining CPU will be the worker nodes. All of the code in this course will work in any personal laptop though it will be slower than executing the same workloads on a larger cluster.
</details>


### Git setup

Create a repository by following these instructions: [Create a new repository](https://github.com/new) → name it `Made-With-ML` → Toggle `Add a README file` (**very important** as this creates a `main` branch) → Click `Create repository` (scroll down)

Now we're ready to clone the repository that has all of our code:

```bash
git clone https://github.com/GokuMohandas/Made-With-ML.git .
```

### Credentials

```bash
touch .env
```
```bash
# Inside .env
GITHUB_USERNAME="CHANGE_THIS_TO_YOUR_USERNAME"  # ← CHANGE THIS
```
```bash
source .env
```

### Virtual environment

<details>
  <summary>Local</summary><br>

  ```bash
  export PYTHONPATH=$PYTHONPATH:$PWD
  python3 -m venv venv  # recommend using Python 3.10
  source venv/bin/activate  # on Windows: venv\Scripts\activate
  python3 -m pip install --upgrade pip setuptools wheel
  python3 -m pip install -r requirements.txt
  pre-commit install
  pre-commit autoupdate
  ```

  > Highly recommend using Python `3.10` and using [pyenv](https://github.com/pyenv/pyenv) (mac) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) (windows).

</details>

## Notebook

Start by exploring the [jupyter notebook](notebooks/madewithml.ipynb) to interactively walkthrough the core machine learning workloads.

<div align="center">
  <img src="https://madewithml.com/static/images/mlops/systems-design/workloads.png">
</div>

<details>
  <summary>Local</summary><br>

  ```bash
  # Start notebook
  jupyter lab notebooks/madewithml.ipynb
```

</details>


## Scripts

Now we'll execute the same workloads using the clean Python scripts following software engineering best practices (testing, documentation, logging, serving, versioning, etc.) The code we've implemented in our notebook will be refactored into the following scripts:

```bash
madewithml
├── config.py
├── data.py
├── evaluate.py
├── models.py
├── predict.py
├── serve.py
├── train.py
├── tune.py
└── utils.py
```

**Note**: Change the `--num-workers`, `--cpu-per-worker`, and `--gpu-per-worker` input argument values below based on your system's resources. For example, if you're on a local laptop, a reasonable configuration would be `--num-workers 6 --cpu-per-worker 1 --gpu-per-worker 0`.

### Training
```bash
export EXPERIMENT_NAME="llm"
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
python madewithml/train.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "$DATASET_LOC" \
    --train-loop-config "$TRAIN_LOOP_CONFIG" \
    --num-workers 1 \
    --cpu-per-worker 3 \
    --gpu-per-worker 1 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp results/training_results.json
```

### Tuning
```bash
export EXPERIMENT_NAME="llm"
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
export INITIAL_PARAMS="[{\"train_loop_config\": $TRAIN_LOOP_CONFIG}]"
python madewithml/tune.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "$DATASET_LOC" \
    --initial-params "$INITIAL_PARAMS" \
    --num-runs 2 \
    --num-workers 1 \
    --cpu-per-worker 3 \
    --gpu-per-worker 1 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp results/tuning_results.json
```

### Experiment tracking

We'll use [MLflow](https://mlflow.org/) to track our experiments and store our models and the [MLflow Tracking UI](https://www.mlflow.org/docs/latest/tracking.html#tracking-ui) to view our experiments. We have been saving our experiments to a local directory but note that in an actual production setting, we would have a central location to store all of our experiments. It's easy/inexpensive to spin up your own MLflow server for all of your team members to track their experiments on or use a managed solution like [Weights & Biases](https://wandb.ai/site), [Comet](https://www.comet.ml/), etc.

```bash
export MODEL_REGISTRY=$(python -c "from madewithml import config; print(config.MODEL_REGISTRY)")
mlflow server -h 0.0.0.0 -p 8080 --backend-store-uri $MODEL_REGISTRY
```

<details>
  <summary>Local</summary><br>

  If you're running this notebook on your local laptop then head on over to <a href="http://localhost:8080/" target="_blank">http://localhost:8080/</a> to view your MLflow dashboard.

</details>


### Evaluation
```bash
export EXPERIMENT_NAME="llm"
export RUN_ID=$(python madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
export HOLDOUT_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/holdout.csv"
python madewithml/evaluate.py \
    --run-id $RUN_ID \
    --dataset-loc $HOLDOUT_LOC \
    --results-fp results/evaluation_results.json
```
```json
{
  "timestamp": "June 09, 2023 09:26:18 AM",
  "run_id": "6149e3fec8d24f1492d4a4cabd5c06f6",
  "overall": {
    "precision": 0.9076136428670714,
    "recall": 0.9057591623036649,
    "f1": 0.9046792827719773,
    "num_samples": 191.0
  },
...
```

### Inference
```bash
export EXPERIMENT_NAME="llm"
export RUN_ID=$(python madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
python madewithml/predict.py predict \
    --run-id $RUN_ID \
    --title "Transfer learning with transformers" \
    --description "Using transformers for transfer learning on text classification tasks."
```
```json
[{
  "prediction": [
    "natural-language-processing"
  ],
  "probabilities": {
    "computer-vision": 0.0009767753,
    "mlops": 0.0008223939,
    "natural-language-processing": 0.99762577,
    "other": 0.000575123
  }
}]
```

### Serving

<details>
  <summary>Local</summary><br>

  ```bash
  # Start
  ray start --head
  ```

  ```bash
  # Set up
  export EXPERIMENT_NAME="llm"
  export RUN_ID=$(python madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
  python madewithml/serve.py --run_id $RUN_ID
  ```

  Once the application is running, we can use it via cURL, Python, etc.:

  ```python
  # via Python
  import json
  import requests
  title = "Transfer learning with transformers"
  description = "Using transformers for transfer learning on text classification tasks."
  json_data = json.dumps({"title": title, "description": description})
  requests.post("http://127.0.0.1:8000/predict", data=json_data).json()
  ```

  ```bash
  ray stop  # shutdown
  ```

</details>


### Testing
```bash
# Code
python3 -m pytest tests/code --verbose --disable-warnings

# Data
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings

# Model
export EXPERIMENT_NAME="llm"
export RUN_ID=$(python madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
pytest --run-id=$RUN_ID tests/model --verbose --disable-warnings

# Coverage
python3 -m pytest tests/code --cov madewithml --cov-report html --disable-warnings  # html report
python3 -m pytest tests/code --cov madewithml --cov-report term --disable-warnings  # terminal report
```

## Production

From this point onwards, in order to deploy our application into production, we'll need to either be on Anyscale or on a [cloud VM](https://docs.ray.io/en/latest/cluster/vms/index.html#cloud-vm-index) / [on-prem](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html#on-prem) cluster you manage yourself (w/ Ray). If not on Anyscale, the commands will be [slightly different](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html) but the concepts will be the same.

<div align="center">
  <img src="https://madewithml.com/static/images/mlops/jobs_and_services/manual.png">
</div>

### Cluster environment

The cluster environment determines **where** our workloads will be executed (OS, dependencies, etc.) We've already created this [cluster environment](./deploy/cluster_env.yaml) for us but this is how we can create/update one ourselves.

```bash
export CLUSTER_ENV_NAME="madewithml-cluster-env"
anyscale cluster-env build deploy/cluster_env.yaml --name $CLUSTER_ENV_NAME
```

### Compute configuration

The compute configuration determines **what** resources our workloads will be executes on. We've already created this [compute configuration](./deploy/cluster_compute.yaml) for us but this is how we can create it ourselves.

```bash
export CLUSTER_COMPUTE_NAME="madewithml-cluster-compute-g5.4xlarge"
anyscale cluster-compute create deploy/cluster_compute.yaml --name $CLUSTER_COMPUTE_NAME
```

### CI/CD

We're not going to manually deploy our application every time we make a change. Instead, we'll automate this process using GitHub Actions!

1. Create a new github branch to save our changes to and execute CI/CD workloads:
```bash
git remote set-url origin https://github.com/$GITHUB_USERNAME/Made-With-ML.git  # <-- CHANGE THIS to your username
git checkout -b dev
```

2. We'll start by adding the necessary credentials to the [`/settings/secrets/actions`](https://github.com/GokuMohandas/Made-With-ML/settings/secrets/actions) page of our GitHub repository.

``` bash
export ANYSCALE_HOST=https://console.anyscale.com
export ANYSCALE_CLI_TOKEN=$YOUR_CLI_TOKEN  # retrieved from https://console.anyscale.com/o/madewithml/credentials
```

3. Now we can make changes to our code (not on `main` branch) and push them to GitHub. But in order to push our code to GitHub, we'll need to first authenticate with our credentials before pushing to our repository:

```bash
git config --global user.name $GITHUB_USERNAME  # <-- CHANGE THIS to your username
git config --global user.email you@example.com  # <-- CHANGE THIS to your email
git add .
git commit -m ""  # <-- CHANGE THIS to your message
git push origin dev
```

Now you will be prompted to enter your username and password (personal access token). Follow these steps to get personal access token: [New GitHub personal access token](https://github.com/settings/tokens/new) → Add a name → Toggle `repo` and `workflow` → Click `Generate token` (scroll down) → Copy the token and paste it when prompted for your password.

4. Now we can start a PR from this branch to our `main` branch and this will trigger the [workloads workflow](/.github/workflows/workloads.yaml). If the workflow (Anyscale Jobs) succeeds, this will produce comments with the training and evaluation results directly on the PR.

<div align="center">
  <img src="https://madewithml.com/static/images/mlops/cicd/comments.png">
</div>

5. If we like the results, we can merge the PR into the `main` branch. This will trigger the [serve workflow](/.github/workflows/serve.yaml) which will rollout our new service to production!

### Continual learning

With our CI/CD workflow in place to deploy our application, we can now focus on continually improving our model. It becomes really easy to extend on this foundation to connect to scheduled runs (cron), [data pipelines](https://madewithml.com/courses/mlops/data-engineering/), drift detected through [monitoring](https://madewithml.com/courses/mlops/monitoring/), [online evaluation](https://madewithml.com/courses/mlops/evaluation/#online-evaluation), etc. And we can easily add additional context such as comparing any experiment with what's currently in production (directly in the PR even), etc.

<div align="center">
  <img src="https://madewithml.com/static/images/mlops/cicd/continual.png">
</div>
