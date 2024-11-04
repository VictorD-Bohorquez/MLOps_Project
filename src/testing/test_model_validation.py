import pytest
import yaml
import numpy as np
import pandas as pd
import joblib 
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
#import sys
#import os
 
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../stages')))
from src.stages.classes.models import Models

def load_config(path):
    with open(path) as con:
        config = yaml.safe_load(con)
    return config

def load_data(path):
    config = load_config(path)
    file = config['processed']['path_xtrain']
    Xtrain = pd.read_csv(file, sep=';')
    file = config['datasets']['path_ytrain']
    ytrain = pd.read_csv(file, sep=';')
    return  Xtrain, ytrain

def create_models(path):
    Xtrain, ytrain = load_data(path)
    models = Models(path)
    models.finetune_parameters(Xtrain, ytrain)
    return models

models = create_models(path="src/testing/train_val.yaml") # create a dedicated yaml file to avoid using params.yaml
X, y = load_data(path="src/testing/train_val.yaml")

inputs = []
for model in models.models:
    inputs.append((model,X,y))

def cross_validation(model, X_train, y_train):
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=42)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv)
    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)
    return mean_accuracy, std_accuracy


def evaluate_model(model, x, y): 
    y_pred = model.predict(x) 
    accuracy = accuracy_score(y, y_pred) 
    f1 = f1_score(y, y_pred, average='weighted') 
    cm = confusion_matrix(y, y_pred) 
    print(f"Accuracy: {accuracy:.2f}") 
    print(f"F1 Score: {f1:.2f}") 
    print("Confusion Matrix:") 
    print(cm) 
    return accuracy, f1, cm

@pytest.mark.parametrize("model, X, y", inputs)
def test_mean_accuracy(model, X, y): 
    mean_accuracy, std_accuracy = cross_validation(model,X, y)
    assert mean_accuracy > 0.6

@pytest.mark.parametrize("model, X, y", inputs)
def test_std_accuracy(model, X, y): 
    mean_accuracy, std_accuracy = cross_validation(model,X, y)
    assert std_accuracy < 0.05
