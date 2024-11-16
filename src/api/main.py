from fastapi import FastAPI, UploadFile, File, HTTPException, status
from src.api.classes.model_loader import Model_Loader
from src.api.classes.feature_manager import Feature_Manager
import pandas as pd
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()

@app.get('/model_metrics', status_code=status.HTTP_200_OK)
async def get_metrics(environment:str = 'Test', model:str = None):
    env = environment.lower()
    models = Model_Loader(path='params.yaml')
    if env != 'test' and env != 'train':
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='The selected environment is not available')
    else:
        models.load_data(environment=env)
        metrics = models.evaluate_metrics(model)
        if metrics:
            response = {env+'_metrics':metrics}
            return response
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='The provided model name does not exist')

@app.post('/predict', status_code=status.HTTP_200_OK)
async def predict(file: UploadFile, model:str = None):
    if not validate_file(file):
        raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail='Only CSV files are accepted')
    df = pd.read_csv(file.file,sep=';')
    file.file.close()
    columns = list(df.columns)
    if not validate_columns(columns):
        raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail='The CSV does not contain the proper information')
    features = Feature_Manager(path='params.yaml')
    features.get_PCA_components()
    X = features.process_features(df)
    models = Model_Loader(path='params.yaml')
    prediction = models.predict(data=X, model=model)
    if prediction:
        return prediction
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='The provided model name does not exist')

def validate_file(file:UploadFile):
    if not file.filename.endswith('csv'):
        return False
    return True

def validate_columns(columns:list[str]):
    c = str(os.getenv('COLUMNS_NEEDED'))
    columns_needed = c.split(",")
    for i in range(len(columns)):
        columns[i] = columns[i].replace("\t","")
    if sorted(columns) == sorted(columns_needed):
        return True
    return False