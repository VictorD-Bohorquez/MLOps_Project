#from stages.model_training import train
from src.stages.classes.models import Models
import argparse
import yaml
import numpy as np
import pandas as pd
import joblib
import pytest

def train(path):
    with open(path) as con:
        config = yaml.safe_load(con)
    models = Models(path)
    file = config['processed']['path_xtrain']
    Xtrain = pd.read_csv(file, sep=';')
    file = config['datasets']['path_ytrain']
    ytrain = pd.read_csv(file, sep=';')
    column_types = Xtrain.dtypes
    return (Xtrain.shape[0],ytrain.shape[0],column_types)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameters tunning and training processing step.')
    parser.add_argument('--config', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    train(path=args.config)