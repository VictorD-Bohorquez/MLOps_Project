# src/testing/utils.py
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_path):
    return pd.read_csv(data_path)

def load_model(model_path):
    return joblib.load(model_path)

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted']), annot=True, fmt='d')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

