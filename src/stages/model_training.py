from classes.models import Models
import argparse
import yaml
import numpy as np
import pandas as pd

def train(path):
    with open(path) as con:
        config = yaml.safe_load(con)
    models = Models(path)
    file = config['processed']['path_xtrain']
    Xtrain = pd.read_csv(file, sep=';')
    file = config['datasets']['path_ytrain']
    ytrain = pd.read_csv(file, sep=';')
    models.finetune_parameters(Xtrain, ytrain)
    models.validate_models(Xtrain, ytrain)
    models.train_models(Xtrain, ytrain)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameters tunning and training processing step.')
    parser.add_argument('--config', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    train(path=args.config)