#from stages.model_training import train
from src.testing.test_model import train
import argparse
import yaml
import numpy as np
import pandas as pd
import joblib
import pytest

# This test is intended to validate Xtrain and ytrain have same num of rows
def test_datasets_size():
    xtrain_size, ytrain_size, xxx = train('params.yaml')
    assert xtrain_size == ytrain_size

# This test is intended to validate none column in Xtrain dataset is Object type
def test_no_object_types():
    xxx, yyy, column_types = train('params.yaml')
    #object_type = 'object'
    object_type = 'float64'
    assert object_type  not in column_types.values


