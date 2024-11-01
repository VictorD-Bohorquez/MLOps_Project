
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../stages')))

import pytest
from unittest.mock import MagicMock, mock_open, patch
import pandas as pd
import numpy as np
from stages.feature_engineering import process
from stages.classes.feature_manager import Feature_Manager

@pytest.fixture
def mock_feature_manager(monkeypatch):
    # Create a mock Feature_Manager
    mock_manager = MagicMock()
    
    # Mock the PCA method to set pca_components based on variance
    def mock_get_PCA_components():
        mock_manager.pca_components = 5  # Example component count explaining 90% of variance
    
    mock_manager.get_PCA_components = mock_get_PCA_components
    mock_manager.process_features = MagicMock(return_value=pd.DataFrame({'feature1': [1], 'feature2': [2]}))
    mock_manager.sync_columns = MagicMock(return_value=pd.DataFrame({'feature1': [1], 'feature2': [2]}))
    
    # Replace Feature_Manager in feature_engineering.py with our mock
    monkeypatch.setattr('stages.feature_engineering.Feature_Manager', lambda path: mock_manager)
    return mock_manager

@patch("builtins.open", new_callable=mock_open, read_data="config: dummy")
@patch("yaml.safe_load", return_value={
    "datasets": {
        "path_xtrain": "dummy_xtrain.csv",
        "path_xtest": "dummy_xtest.csv"
    },
    "processed": {
        "path_xtrain": "processed_xtrain.csv",
        "path_xtest": "processed_xtest.csv"
    }
})
@patch("pandas.read_csv", return_value=pd.DataFrame({
    'col1': np.random.rand(100),
    'col2': np.random.rand(100)
}))
@patch("pandas.DataFrame.to_csv")  # Mocking to_csv to avoid actual file writing
def test_process(mock_to_csv, mock_read_csv, mock_yaml, mock_file, mock_feature_manager):
    # Run process function with a dummy config path
    process(path='dummy_config.yaml')
    
    # Check if process_features and sync_columns were called
    assert mock_feature_manager.process_features.called, "process_features should be called"
    assert mock_feature_manager.sync_columns.called, "sync_columns should be called"
    
    # Verify that to_csv is called twice (once for Xtrain, once for Xtest)
    assert mock_to_csv.call_count == 2, "to_csv should be called twice"
    
    # Assert that PCA components were set correctly
    mock_feature_manager.get_PCA_components()
    assert mock_feature_manager.pca_components == 5, "PCA components should be set to 5"