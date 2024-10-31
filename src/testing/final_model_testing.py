import pytest
import yaml
import numpy as np
import pandas as pd
import joblib 
from sklearn.metrics import accuracy_score

def load_models(path):
    with open(path) as con:
        config = yaml.safe_load(con)
        models =  joblib.load(config['models']['path_models']) 
    file = config['processed']['path_xtest']
    Xtest = pd.read_csv(file, sep=';')
    file = config['datasets']['path_ytest']
    ytest = pd.read_csv(file, sep=';')
    return models, Xtest, ytest

models, x, y = load_models("params.yaml")
inputs = []
for model in models:
    inputs.append((model,x,y))

@pytest.mark.parametrize("model,x,y,", inputs)
def test_accuracy(model, x, y):
    predict = model.predict(x)
    accuracy = accuracy_score(y, predict)
    assert accuracy >= 0.6

