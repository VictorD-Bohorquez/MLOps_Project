from classes.models import Models
import argparse
import yaml
import numpy as np
import pandas as pd
import joblib 

def evaluation(path):
    with open(path) as con:
        config = yaml.safe_load(con)
    evaluator = Models()
    models = joblib.load(config['models']['path_models'])
    file = config['processed']['path_xtest']
    Xtest = pd.read_csv(file, sep=';')
    file = config['datasets']['path_ytest']
    ytest = pd.read_csv(file, sep=';')
    models.evaluate(Xtest, ytest, models)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Models evaluation step.')
    parser.add_argument('--config', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    train(path=args.config)