import os

from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
VERSION_FILE_NAME = os.getenv('VERSION_FILE_NAME')

import joblib
import pandas as pd
from datetime import datetime
from google.cloud import storage
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split


def download_data():
    from sklearn.datasets import load_wine
    wine = load_wine()
    features = pd.DataFrame(wine.data, columns=wine.feature_names)
    target = pd.Series(wine.target)
    return features, target


def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_weighted": round(
            precision_score(y_test, y_pred, average="weighted"), 4
        ),
    }


def get_model_version(bucket_name, version_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(version_file_name)
    if blob.exists():
        version_as_string = blob.download_as_text()
        version = int(version_as_string)
    else:
        version = 0
    return version


def update_model_version(bucket_name, version_file_name, version):
    if not isinstance(version, int):
        raise ValueError("Version must be an integer")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(version_file_name)
        blob.upload_from_string(str(version))
        return True
    except Exception as e:
        print(f"Error updating model version: {e}")
        return False


def ensure_folder_exists(bucket, folder_name):
    blob = bucket.blob(f"{folder_name}/")
    if not blob.exists():
        blob.upload_from_string('')
        print(f"Created folder: {folder_name}")


def save_model_to_gcs(model, bucket_name, blob_name):
    joblib.dump(model, "model.joblib")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    ensure_folder_exists(bucket, "trained_models")
    blob = bucket.blob(blob_name)
    blob.upload_from_filename('model.joblib')


def main():
    current_version = get_model_version(BUCKET_NAME, VERSION_FILE_NAME)
    new_version = current_version + 1

    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    print(f"Accuracy:           {metrics['accuracy']}")
    print(f"Precision weighted: {metrics['precision_weighted']}")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    blob_name = f"trained_models/model_v{new_version}_{timestamp}.joblib"
    save_model_to_gcs(model, BUCKET_NAME, blob_name)
    print(f"Model saved to gs://{BUCKET_NAME}/{blob_name}")

    if update_model_version(BUCKET_NAME, VERSION_FILE_NAME, new_version):
        print(f"Model version updated to {new_version}")
        print(f"MODEL_VERSION_OUTPUT: {new_version}")
    else:
        print("Failed to update model version")


if __name__ == "__main__":
    main()
