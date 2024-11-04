import pytest
from src.stages.classes.models import Models
test_model_path="src/testing/train_val.yaml"
 
@pytest.fixture
def model_generator():
    return Models(path=test_model_path)
#This test is to validate at least 1 model is being trained
def test_atleast_1_model(model_generator):
    models, names = model_generator.generate_models()
    assert len(models) > 0
#This test is to validate expected models have been trained
def test_expected_models(model_generator):
    models, names = model_generator.generate_models()

    expected_model_types = [
        'LogisticRegression',
        'KNeighborsClassifier',
        'DecisionTreeClassifier',
        'RandomForestClassifier',
        'SVC']

    for model, expected_name in zip(models, expected_model_types):
        assert type(model).__name__ == expected_name, f"Se esperaba un modelo de tipo {expected_name}"
    
