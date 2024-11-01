#from stages.model_training import train
from src.testing.test_model import train
import argparse
import yaml
import numpy as np
import pandas as pd
import joblib
import pytest

class TestClass:

    # This test is intended to validate Xtrain and ytrain have same num of rows
    def test_datasets_size(self):
        xtrain_size, ytrain_size, tmp = train('params.yaml')
        assert xtrain_size == ytrain_size

    # This test is intended to validate none column in Xtrain dataset is Object type
    def test_no_object_types(self):
        tmp1, tmp2, column_types = train('params.yaml')
        object_type = 'object'
        assert object_type  not in column_types.values


