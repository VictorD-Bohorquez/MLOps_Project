# src/testing/test_model_performance.py
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from utils import load_data, load_model, plot_confusion_matrix

# Definir rutas
DATA_PATH = "/content/drive/MyDrive/ColabNotebooks/MLOps_Project/data/raw/data.csv"
MODEL_PATH = "/content/MLOps_Project/mlruns/119121465528204301/c6485ba31735424cacdaab42b687f574/artifacts/models/model.pkl"
REPORT_PATH = "src/testing/reports/performance_report.txt"
GRAPH_PATH = "src/testing/reports/performance_graphs/confusion_matrix.png"

# Cargar datos y modelo
data = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

# Dividir datos en conjuntos de entrenamiento y validación
X = data.drop(columns=["target"])  # Asegurarse de que 'target' es el nombre de la columna de etiquetas
y = data["target"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluar en el conjunto de entrenamiento
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred, average="weighted")
train_precision = precision_score(y_train, y_train_pred, average="weighted")
train_recall = recall_score(y_train, y_train_pred, average="weighted")

# Evaluar en el conjunto de validación
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average="weighted")
val_precision = precision_score(y_val, y_val_pred, average="weighted")
val_recall = recall_score(y_val, y_val_pred, average="weighted")

# Generar reporte
with open(REPORT_PATH, "w") as f:
    f.write("MÉTRICAS DE ENTRENAMIENTO:\n")
    f.write(f"Exactitud: {train_accuracy}\n")
    f.write(f"F1 Score: {train_f1}\n")
    f.write(f"Precisión: {train_precision}\n")
    f.write(f"Recall: {train_recall}\n\n")
    
    f.write("MÉTRICAS DE VALIDACIÓN:\n")
    f.write(f"Exactitud: {val_accuracy}\n")
    f.write(f"F1 Score: {val_f1}\n")
    f.write(f"Precisión: {val_precision}\n")
    f.write(f"Recall: {val_recall}\n\n")
    
    f.write("REPORTE DE CLASIFICACIÓN (Validación):\n")
    f.write(classification_report(y_val, y_val_pred))

# Generar matriz de confusión
plot_confusion_matrix(y_val, y_val_pred, "Confusion Matrix - Validation Set", GRAPH_PATH)

print("Pruebas completadas. Consulta la carpeta src/testing/reports para ver los resultados.")

