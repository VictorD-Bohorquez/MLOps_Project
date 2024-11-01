import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../stages')))

import pytest
from unittest.mock import MagicMock, mock_open, patch

from stages.data_load import load_data
from stages.classes.data_manager import Data_Manager



@pytest.fixture
def mock_data_manager(monkeypatch):
    # Mock the Data_Manager's load_data method to create a dataset with 100 rows
    mock_manager = MagicMock()
    mock_manager.data = range(100)  # Simulating 100 rows
    
    # Mock methods within Data_Manager
    mock_manager.load_data = MagicMock()
    mock_manager.split_data = MagicMock()

    # Replace Data_Manager in data_load.py with our mock
    monkeypatch.setattr('stages.data_load.Data_Manager', lambda path: mock_manager)
    return mock_manager

@patch("builtins.open", new_callable=mock_open, read_data="config: dummy")
@patch("yaml.safe_load", return_value={"config": "dummy"})
def test_load_data_with_100_rows(mock_yaml, mock_file, mock_data_manager):
    # Assume path is provided correctly; you may pass any string as it’s mocked
    load_data(path='dummy_config.yaml')
    
    # Check if data has at least 100 rows
    assert len(mock_data_manager.data) >= 100, "Data should contain at least 100 rows."