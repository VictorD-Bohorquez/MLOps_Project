from classes.feature_manager import Feature_Manager
import argparse
import yaml
import numpy as np
import pandas as pd

def process(path):
    with open(path) as con:
        config = yaml.safe_load(con)
    features = Feature_Manager(path)
    features.get_PCA_components()
    file = config['datasets']['path_xtrain']
    Xtrain = pd.read_csv(file, sep=';')
    Xtrain = features.process_features(Xtrain)
    Xtrain.to_csv(config['processed']['path_xtrain'], sep=';', index=False)
    file = config['datasets']['path_xtest']
    Xtest = pd.read_csv(file, sep=';')
    Xtest = features.process_features(Xtest)
    Xtest = features.sync_columns(Xtrain, Xtest)
    Xtest.to_csv(config['processed']['path_xtest'], sep=';', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data feature processing step.')
    parser.add_argument('--config', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    process(path=args.config)