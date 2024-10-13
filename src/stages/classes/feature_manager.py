import yaml
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from warnings import filterwarnings
filterwarnings('ignore')

class Feature_Manager():
    def __init__(self, path):
        with open(path) as config:
            self.config = yaml.safe_load(config)
        self.pca = PCA()
        self.pca_components = None
        self.log_pipe = Pipeline([('Log', FunctionTransformer(np.log1p, feature_names_out='one-to-one'))] )
        self.log_pipe_nombres = self.config['features']['log']
        self.scaler_pipe = Pipeline([('scaler', StandardScaler())] )
        self.scaler_pipe_nombres = self.config['features']['scaler']
        self.catOHE_pipeline = Pipeline( [('OneHot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))] )
        self.catOHE_pipeline_nombres = list(set(self.get_allcolumns()) - set(self.log_pipe_nombres + self.scaler_pipe_nombres))
        self.ct_numericas = ColumnTransformer( transformers=[('log_transformer', self.log_pipe, self.log_pipe_nombres),('standard_Scaler', self.scaler_pipe, self.scaler_pipe_nombres)])
        self.ct_categoricas = ColumnTransformer( transformers=[('cat', self.catOHE_pipeline, self.catOHE_pipeline_nombres)])
        self.all_categories = self.get_total_categories()
    
    def get_allcolumns(self):
        path = self.config['datasets']['path_xtrain']
        data = pd.read_csv(path, sep=';')
        return data.columns.values

    def get_total_categories(self):
        path = self.config['data_load']['dataset']
        data = pd.read_csv(path, sep=';')
        all_cat = self.ct_categoricas.fit_transform(data)
        total_categories = self.ct_categoricas.named_transformers_['cat'].get_feature_names_out()
        return total_categories
    
    def sync_columns(self, reference, data):
        data = data[reference.columns]
        return data
    
    def complete_features(self, data):
        columns = data.columns
        missig_categories = list(set(self.all_categories)-set(columns))
        for missing in missig_categories:
            data[missing]=0
        return data

    def get_PCA_components(self):
        path = self.config['datasets']['path_xtrain']
        data = pd.read_csv(path, sep=';')
        processed = self.ct_numericas.fit_transform(data)
        x_projected = self.pca.fit_transform(processed)
        va = np.cumsum(self.pca.explained_variance_ratio_)
        components = None
        for i in range(len(va)):
            if va[i] > 0.9:
                print(f'El número de componentes que explican el 90% de la varianza son: {i+1}')
                components = i+1
                break
        self.pca_components = components

    def process_features(self, data):
        if self.pca_components:
            data_nums_processed = self.ct_numericas.fit_transform(data)
            x_projected = self.pca.fit_transform(data_nums_processed)
            x_projected = pd.DataFrame(x_projected)
            data_cat_processed= self.ct_categoricas.fit_transform(data)
            onehot_columns = self.ct_categoricas.named_transformers_['cat'].get_feature_names_out()
            data_cat_processed_df = pd.DataFrame(data_cat_processed, columns=onehot_columns)
            data_cat_processed_df = self.complete_features(data_cat_processed_df)
            componentes = x_projected.iloc[:,0:self.pca_components]
            componentes.reset_index(drop=True, inplace=True)
            data_cat_processed_df.reset_index(drop=True, inplace=True)
            data_final = pd.concat([componentes,data_cat_processed_df], axis=1)
            data_final.columns = data_final.columns.astype(str)
            return data_final
        else:
            raise Exception("PCA components has not been calculated, please first run the calculation to process the features.")