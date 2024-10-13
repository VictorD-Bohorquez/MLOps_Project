import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

class Data_Manager():
    def __init__(self, path):
        with open(path) as config:
            self.config = yaml.safe_load(config)
        self.data = self.load_data()
    
    def load_data(self):
        path = self.config['data_load']['dataset']
        data = pd.read_csv(path, sep=';')
        self.data = data
    
    def split_data(self):
        target = self.config['data_load']['target']
        test = self.config['data_load']['test']
        X = self.data.drop([target], axis=1)
        y = self.data[target]
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test, stratify=y)
        print("La dimension del conjunto de entrenamiento es: ",Xtrain.shape)
        print("La dimension del conjunto de prueba es: ",Xtest.shape)
        Xtrain.to_csv(self.config['datasets']['path_xtrain'], sep=';', index=False)
        Xtest.to_csv(self.config['datasets']['path_xtest'], sep=';', index=False)
        ytrain.to_csv(self.config['datasets']['path_ytrain'], sep=';', index=False)
        ytest.to_csv(self.config['datasets']['path_ytest'], sep=';', index=False)