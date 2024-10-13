from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import joblib 
import yaml
from warnings import filterwarnings
filterwarnings('ignore')

class Models():
    def __init__(self, path):
        with open(path) as config:
            self.config = yaml.safe_load(config)
        self.models, self.names = self.generate_models()
        self.params = self.config['tuning']['parameters']
    
    def generate_models(self):
        models, names = list(), list()
        models.append(LogisticRegression(max_iter=1000))
        names.append('LR')
        models.append(KNeighborsClassifier())
        names.append('KNN')
        models.append(DecisionTreeClassifier())
        names.append('DTree')
        models.append(RandomForestClassifier(n_jobs=-1))
        names.append('RF')
        models.append(MLPClassifier( hidden_layer_sizes=(20,20),max_iter=10000))
        names.append('MLP')
        models.append(SVC())
        names.append('SVC')
        return models, names
    
    def finetune_parameters(self, data, y):
        best_parameters = self.get_hyperparameters(data, y)
        for i in range(len(self.models)):
            self.models[i].set_params(**best_parameters[i])

    def get_hyperparameters(self, data, y):
        parameters = []
        for i in range(len(self.models)):
            result = self.model_grid_search(self.models[i], self.params[self.names[i]], data, y)
            parameters.append(result)
            print(f'Best parameters found for {self.names[i]} : {result}')
        return parameters

    def model_grid_search(self, model, params, X_train, y_train):
        grid_search = GridSearchCV(estimator = model, param_grid = params, n_jobs = -1, cv=3)
        grid_search.fit(X_train, y_train.values.ravel())
        best_mod = grid_search.best_params_
        return  best_mod
    
    def validate_models(self, data, y):
        for i in range(len(self.models)):
            pipeline = Pipeline(steps=[('m',self.models[i])])
            cv = RepeatedStratifiedKFold(n_splits=5,
                                            n_repeats=15,
                                            random_state=5
                                            )
            metrics = self.config['evaluation']['metrics']
            scores = cross_validate(pipeline,
                                    data,
                                    np.ravel(y),
                                    scoring=metrics,
                                    cv=cv,
                                    return_train_score=True,
                                    error_score = 0,
                                    )

            print('>> %s' % self.names[i])
            for j,k in enumerate(list(scores.keys())):
                if j>1:
                    print('\t %s %.3f (%.3f)' % (k, np.mean(scores[k]),np.std(scores[k])))
    
    def train_models(self, Xtrain, ytrain):
        for model in self.models:
            model.fit(Xtrain,ytrain)
        file = self.config['models']['path_models']
        joblib.dump(self.models, file)

    def evaluate(self, data, ytest, models):
        Xtest = data
        for i in range(len(models)):
            y_pred= models[i].predict(Xtest)
            print(f"\n>>Reporte final Test de {self.names[i]}:")
            print(classification_report(ytest, y_pred))