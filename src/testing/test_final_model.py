import pytest
import yaml
import numpy as np
import pandas as pd
import joblib 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

def load_config(path):
    with open(path) as con:
        config = yaml.safe_load(con)
    return config

def load_models(path):
    config = load_config(path)
    models =  joblib.load(config['models']['path_models']) 
    file = config['processed']['path_xtest']
    Xtest = pd.read_csv(file, sep=';')
    file = config['datasets']['path_ytest']
    ytest = pd.read_csv(file, sep=';')
    return models, Xtest, ytest

metrics = load_config("./src/testing/final_model_metrics.yaml")
models, x, y = load_models("params.yaml")
inputs = []
for model in models:
    inputs.append((model,x,y))

@pytest.mark.parametrize("model,x,y,", inputs)
def test_accuracy(model, x, y):
    limit = float(metrics['metrics']['accuracy'])
    predict = model.predict(x)
    accuracy = accuracy_score(y, predict)
    assert accuracy >= limit, f"Models accuracy should be at least {limit*100}%"

@pytest.mark.parametrize("model,x,y,", inputs)
def test_avg_precision(model, x, y):
    limit = float(metrics['metrics']['avg_presicion'])
    predict = model.predict(x)
    report = classification_report(y, predict, output_dict=True)
    avg_scores = report['macro avg']
    precision = avg_scores['precision']
    assert precision >= limit, f"Average precision should be at least {limit*100}%"

@pytest.mark.parametrize("model,x,y,", inputs)
def test_avg_recall(model, x, y):
    limit = float(metrics['metrics']['avg_recall'])
    predict = model.predict(x)
    report = classification_report(y, predict, output_dict=True)
    avg_scores = report['macro avg']
    recall = avg_scores['recall']
    assert recall >= limit, f"Average recall should be at least {limit*100}%"

@pytest.mark.parametrize("model,x,y,", inputs)
def test_weighted_precision(model, x, y):
    limit = float(metrics['metrics']['weighted_presicion'])
    predict = model.predict(x)
    report = classification_report(y, predict, output_dict=True)
    avg_scores = report['weighted avg']
    precision = avg_scores['precision']
    assert precision >= limit, f"Weighted precision should be at least {limit*100}%"

@pytest.mark.parametrize("model,x,y,", inputs)
def test_weighted_recall(model, x, y):
    limit = float(metrics['metrics']['weighted_recall'])
    predict = model.predict(x)
    report = classification_report(y, predict, output_dict=True)
    avg_scores = report['weighted avg']
    recall = avg_scores['recall']
    assert recall >= limit, f"Weighted recall should be at least {limit*100}%"