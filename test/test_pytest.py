import pytest
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from unittest.mock import patch, MagicMock
from src.train_and_save_model import download_data, preprocess_data, train_model
from src.train_and_save_model import get_model_version, update_model_version
from src.train_and_save_model import ensure_folder_exists, save_model_to_gcs
from src.train_and_save_model import evaluate_model


# ----------------- Test Download ----------------- #
def test_download_data():
    X, y = download_data()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert not X.empty
    assert not y.empty
    assert X.shape[0] == y.shape[0]
    assert X.shape == (178, 13)
    assert set(y.unique()) == {0, 1, 2}


# ----------------- Test Preprocess ----------------- #
def test_preprocess_data():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]
    assert X_train.shape[1] == X.shape[1]


# ----------------- Test Train model ----------------- #
def test_train_model():
    X = pd.DataFrame({
        'alcohol': [13.2, 12.4, 11.8, 14.1, 13.7],
        'malic_acid': [1.78, 2.14, 2.36, 1.95, 1.62],
        'ash': [2.14, 2.09, 2.28, 2.50, 2.17],
        'alcalinity_of_ash': [11.2, 19.5, 20.0, 16.8, 14.0],
    })
    y = pd.Series([0, 1, 2, 0, 1])

    model = train_model(X, y)

    assert isinstance(model, GradientBoostingClassifier)
    assert hasattr(model, 'predict')
    assert model.n_estimators == 100
    assert model.learning_rate == 0.1


# ----------------- Test Evaluate model ----------------- #
def test_evaluate_model():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    assert "accuracy" in metrics
    assert "precision_weighted" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision_weighted"] <= 1.0
    assert metrics["accuracy"] >= 0.85


# ----------------- Test Model versioning ----------------- #
def test_get_model_version():
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        bucket_name = "bucket-test"
        version_file_name = "version.txt"

        mock_blob.exists.return_value = True
        version = get_model_version(bucket_name, version_file_name)
        mock_blob.download_as_text.return_value = '1'

        assert version == 1
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.download_as_text.assert_called_once()

        mock_storage_client.reset_mock()
        mock_bucket.reset_mock()
        mock_blob.reset_mock()

        mock_blob.exists.return_value = False
        version = get_model_version(bucket_name, version_file_name)
        mock_blob.download_as_text.return_value = '0'

        assert version == 0
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.download_as_text.assert_not_called()


# ----------------- Test Update Model version ----------------- #
def test_update_model_version():
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        bucket_name = 'bucket-test'
        version_file_name = 'version.txt'
        new_version = 2

        result = update_model_version(bucket_name, version_file_name, new_version)
        assert result is True
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.upload_from_string.assert_called_once_with(str(new_version))

        mock_storage_client.reset_mock()
        mock_bucket.reset_mock()
        mock_blob.reset_mock()

        with pytest.raises(ValueError):
            update_model_version(bucket_name, version_file_name, 'invalid_version')

        mock_blob.upload_from_string.side_effect = Exception("Upload failed")
        result = update_model_version(bucket_name, version_file_name, new_version)
        assert result is False


# ----------------- Test Ensure Folder Exists ----------------- #
def test_ensure_folder_exists():
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        folder_name = "trained_models"

        mock_blob.exists.return_value = False
        ensure_folder_exists(mock_bucket, folder_name)
        mock_bucket.blob.assert_called_with(f"{folder_name}/")
        mock_blob.upload_from_string.assert_called_once_with('')

        mock_blob.reset_mock()

        mock_blob.exists.return_value = True
        ensure_folder_exists(mock_bucket, folder_name)
        mock_bucket.blob.assert_called_with(f"{folder_name}/")
        mock_blob.upload_from_string.assert_not_called()


# ----------------- Test Save model to GCS ----------------- #
def test_save_model_to_gcs():
    model = GradientBoostingClassifier()

    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = False

        save_model_to_gcs(model, 'bucket-test', 'blob-test')

        mock_storage_client.assert_called_once()
        mock_storage_client.return_value.bucket.assert_called_once_with('bucket-test')
        assert mock_bucket.blob.call_count == 2
        mock_bucket.blob.assert_any_call('trained_models/')
        mock_bucket.blob.assert_any_call('blob-test')
        mock_blob.upload_from_filename.assert_called_once_with('model.joblib')
