# IE7374 Lab 4 — CI/CD with GitHub Actions and Google Cloud Platform

## Overview

This lab builds a complete CI/CD pipeline for a machine learning project. Every
push to main triggers a sequence that runs automated tests, trains a model, saves
it to Google Cloud Storage with an auto-incrementing version number, builds a Docker
container that packages the training environment, verifies the container runs
correctly, and pushes it to Google Cloud Artifact Registry with both a version tag
and a latest tag. The result is a fully reproducible, containerized ML workflow
where every deployment is traceable to a specific model version.

## Changes Made to the Lab

**Source code (src/train_and_save_model.py)**

The dataset was changed from Iris to the Wine dataset, which contains 178 samples
across 13 numeric features and 3 target classes. The model was changed from
RandomForestClassifier to GradientBoostingClassifier with n_estimators=100 and
learning_rate=0.1. A dedicated evaluate_model function was extracted from main()
to compute and return both accuracy and weighted precision as a dictionary. This
makes the evaluation step independently testable and reusable.

**Tests (test/test_pytest.py)**

test_download_data was updated to assert the Wine dataset shape (178 samples,
13 features) and the expected three-class target set. test_train_model was updated
to use Wine feature names and assert that the returned model is a
GradientBoostingClassifier with the correct hyperparameters. A new test,
test_evaluate_model, was added to run an end-to-end train/evaluate cycle on the
real Wine data and assert that both metric keys are present, both values fall
between 0 and 1, and accuracy is at least 0.85 given that GradientBoosting is
a strong learner on this dataset. All mock tests for GCP interactions were
preserved unchanged.

**Pipeline (main.yml)**

The original workflow built and pushed the Docker image in a single step with no
validation between the two operations. The build and push steps were separated, and
a new "Verify Docker image" step was added between them. This step starts the
container twice: once to confirm Python is available and reports its version, and
once to import all runtime dependencies and confirm their versions are resolvable
inside the container. The pipeline only proceeds to push if both checks pass,
which prevents a broken image from reaching the registry.

## Prerequisites

The following must be available before running locally.

Python 3.10 or later. Download from https://www.python.org/downloads and
verify with: python3 --version

Docker Desktop for Mac. Download from https://www.docker.com/products/docker-desktop
and verify with: docker --version

A Google Cloud Platform account with:
a project named ie7374-lab4 (or equivalent),
Cloud Storage API and Artifact Registry API both enabled,
a service account with the roles Storage Admin, Storage Object Admin, and
Artifact Registry Administrator,
a GCS bucket to store model files,
an Artifact Registry repository named my-repo in region us-east4.

The following GitHub repository secrets must be set before the pipeline runs:
GCP_SA_KEY (full contents of the service account JSON key),
GCP_PROJECT_ID (your GCP project ID),
GCS_BUCKET_NAME (name of your GCS bucket),
VERSION_FILE_NAME (name of the version tracking file, for example model_version.txt).

## Security Note

The file ie7374-lab4-2a5ed4ca0a5e.json contains GCP service account credentials.
It is excluded by the .gitignore rule *.json. Before running git add, always run
git status and confirm this file does not appear in the staged list. Never commit
it under any name or path.

## Environment Setup

Clone the repository.

```
git clone https://github.com/lequo-neu/IE7374-Github-Lab4.git
cd IE7374-Github-Lab4
```

Create and activate a virtual environment.

```
python3 -m venv venv
source venv/bin/activate
```

Install all dependencies. All versions are pinned in requirements.txt.

```
pip install -r requirements.txt
```

Create a .env file in the root of the repo with your GCS configuration. This
file is excluded by .gitignore and must never be committed.

```
GCS_BUCKET_NAME=your-bucket-name
VERSION_FILE_NAME=model_version.txt
```

Export the path to your service account key so the script can authenticate with GCP.

```
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/ie7374-lab4-2a5ed4ca0a5e.json"
```

## Running the Code Locally

Run the test suite first to confirm everything passes before touching GCP.

```
pytest test/test_pytest.py -v
```

If all tests pass, run the training script. This reads your .env file,
trains the model, saves it to GCS, and increments the version counter.

```
python src/train_and_save_model.py
```

To test the Docker build locally:

```
docker build -t lab4-local-test .
docker run --rm lab4-local-test python --version
docker run --rm lab4-local-test python -c "import sklearn, joblib, pandas; print('OK')"
```

## What a Successful Run Looks Like

After running pytest, every test should be green.

```
test/test_pytest.py::test_download_data          PASSED
test/test_pytest.py::test_preprocess_data        PASSED
test/test_pytest.py::test_train_model            PASSED
test/test_pytest.py::test_evaluate_model         PASSED
test/test_pytest.py::test_get_model_version      PASSED
test/test_pytest.py::test_update_model_version   PASSED
test/test_pytest.py::test_ensure_folder_exists   PASSED
test/test_pytest.py::test_save_model_to_gcs      PASSED

8 passed in X.XXs
```

After running the training script, the terminal should show:

```
Accuracy:           0.9722
Precision weighted: 0.9740
Model saved to gs://your-bucket/trained_models/model_v2_TIMESTAMP.joblib
Model version updated to 2
MODEL_VERSION_OUTPUT: 2
```

On GitHub, after any push to main, the Actions tab will show the pipeline running
through all steps. The "Verify Docker image" step will print each library's version
inside the container before proceeding to the push. Once the pipeline completes,
navigate to Artifact Registry in GCP Console and open my-repo. You will find the
model-image with two tags: the version number from that run and latest.

## Project Structure

```
IE7374-Github-Lab4/
    .github/
        workflows/
            main.yml
    src/
        __init__.py
        train_and_save_model.py
    test/
        __init__.py
        test_pytest.py
    .gitignore
    Dockerfile
    Mock.md
    README.md
    requirements.txt
```

## CI/CD Pipeline Summary

The main.yml workflow triggers on every push to main. It sets up Python 3.10,
caches pip dependencies, runs the full pytest suite including all mock tests for
GCP interactions, authenticates with GCP, trains the model and saves it to GCS,
reads back the current model version from GCS, builds the Docker image, verifies
that the container's Python environment and all runtime dependencies resolve
correctly, and only then pushes the image to Artifact Registry with both the
version tag and latest.
