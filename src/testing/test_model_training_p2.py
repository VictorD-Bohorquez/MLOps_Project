import pytest
import pandas as pd
from src.stages.classes.models import Models
test_model_path="src/testing/train_val.yaml"
 
@pytest.fixture
def model_generator():
    return Models(path=test_model_path)

def test_generate_models(model_generator):
    models, names = model_generator.generate_models()

# Verifica que los modelos se entrenen sin errores
    Xtrain = pd.read_csv('data/processed/xtrain.csv', sep=';')
    ytrain = pd.read_csv('data/raw/ytrain.csv')

    try:
        model_generator.train_models(Xtrain, ytrain)
    except Exception as e:
        pytest.fail(f"El entrenamiento falló con un error: {e}")