import joblib 
import yaml
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from warnings import filterwarnings
from mlflow.sklearn import load_model
import pandas as pd
from dotenv import load_dotenv
import os
filterwarnings('ignore')

load_dotenv()

class Model_Loader():
    def __init__(self, path):
        self.path = path
        self.names = self.load_names(self.path)
        self.models = self.load_models()
        self.X = None
        self.y = None

    def load_config(self, path):
        with open(path) as con:
            config = yaml.safe_load(con)
        return config
    
    def load_names(self, path):
        config = self.load_config(path)
        names = config['models']['names']
        return names

    def load_models(self):
        models = []
        dir_path = os.path.dirname(os.path.realpath(__file__))
        for name in self.names:
            model_version = int(os.getenv('MODELS_VERSION'))
            model = load_model(
                model_uri=f"models:/{name}/{model_version}"
            )
            models.append(model)
        return models
    
    def load_data(self, environment):
        config = self.load_config(self.path)
        file = config['processed']['path_x'+environment]
        X = pd.read_csv(file, sep=';')
        file = config['datasets']['path_y'+environment]
        y  = pd.read_csv(file, sep=';')
        self.X = X
        self.y = y
    
    def evaluate_metrics(self, name):
        response = {}
        if name == None:
            for i in range(len(self.models)):
                predict = self.models[i].predict(self.X)
                report = classification_report(self.y, predict, output_dict=True)
                metrics = {'accuracy': report.pop("accuracy")}
                for class_or_avg, metrics_dict in report.items():
                    for metric, value in metrics_dict.items():
                        metrics[class_or_avg + '_' + metric] = value
                response[self.names[i]] = metrics
        else:
            if name.upper() not in self.names:
                return None
            i = self.names.index(name.upper())
            response = {}
            predict = self.models[i].predict(self.X)
            report = classification_report(self.y, predict, output_dict=True)
            metrics = {'accuracy': report.pop("accuracy")}
            for class_or_avg, metrics_dict in report.items():
                for metric, value in metrics_dict.items():
                    metrics[class_or_avg + '_' + metric] = value
            response[self.names[i]] = metrics
        return response
    
    def predict(self, data:pd.DataFrame, model:str):
        response = {}
        if model == None:
            for i in range(len(self.models)):
                columns = list(self.models[i].feature_names_in_)
                data = data[columns]
                pred = self.models[i].predict(data)
                name = self.names[i]
                model_pred = {}
                for i in range(len(pred)):
                    model_pred[str(i)] = {'prediction':pred[i]}
                response[name] = model_pred
        else:
            if model.upper() not in self.names:
                return None
            i = self.names.index(model.upper())
            columns = list(self.models[i].feature_names_in_)
            data = data[columns]
            pred = self.models[i].predict(data)
            model_pred = {}
            for x in range(len(pred)):
                model_pred[str(x)] = {'prediction':pred[x]}
            response[self.names[i]] = model_pred
        return response
            
