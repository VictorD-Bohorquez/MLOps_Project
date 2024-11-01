import pytest
import yaml
import numpy as np
import pandas as pd
import joblib 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

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
    assert accuracy >= 0.6, "Models accuracy should be at least 60%"

@pytest.mark.parametrize("model,x,y,", inputs)
def test_avg_precision(model, x, y):
    predict = model.predict(x)
    report = classification_report(y, predict, output_dict=True)
    avg_scores = report['macro avg']
    precision = avg_scores['precision']
    assert precision >= 0.5, f"Average precision should be at least 50%"

@pytest.mark.parametrize("model,x,y,", inputs)
def test_avg_recall(model, x, y):
    predict = model.predict(x)
    report = classification_report(y, predict, output_dict=True)
    avg_scores = report['macro avg']
    recall = avg_scores['recall']
    assert recall >= 0.5, f"Average recall should be at least 50%"

@pytest.mark.parametrize("model,x,y,", inputs)
def test_weighted_precision(model, x, y):
    predict = model.predict(x)
    report = classification_report(y, predict, output_dict=True)
    avg_scores = report['weighted avg']
    precision = avg_scores['precision']
    assert precision >= 0.6, f"Weighted precision should be at least 60%"

@pytest.mark.parametrize("model,x,y,", inputs)
def test_weighted_recall(model, x, y):
    predict = model.predict(x)
    report = classification_report(y, predict, output_dict=True)
    avg_scores = report['weighted avg']
    recall = avg_scores['recall']
    assert recall >= 0.6, f"Weighted recall should be at least 60%"