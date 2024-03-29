import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import wandb
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, confusion_matrix

# Cargar datos
wbcd = load_breast_cancer()
feature_names = wbcd.feature_names
labels = wbcd.target_names

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(wbcd.data, wbcd.target, test_size=0.2)

# Entrenar modelo y obtener predicciones
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_probas = model.predict_proba(X_test)

# Configuración
nombre_proyecto = "MLOps-mod-classification-2024"

# Inicializar la corrida en W&B
run = wandb.init(project=nombre_proyecto, name="classification")

# Visualizar proporciones de clase
wandb.sklearn.plot_class_proportions(y_train, y_test, labels)

# Visualizar curva de aprendizaje
wandb.sklearn.plot_learning_curve(model, X_train, y_train)

# Imprimir información sobre y_test
print("y_test shape:", y_test.shape)
print("y_test values:", y_test)

# Convertir valores continuos de y_test a etiquetas binarias
threshold = 0.5
y_test_binary = (y_test > threshold).astype(int)

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test_binary, y_probas[:, 1])  # Probabilidad de la clase positiva

# Registrar la curva ROC en Weights & Biases
roc_chart = wandb.sklearn.plot_roc(y_test_binary, y_probas[:, 1], labels=labels)
wandb.log({"ROC_Curve": roc_chart})

# Imprimir información para depuración
print("Curva ROC - fpr:", fpr.tolist())
print("Curva ROC - tpr:", tpr.tolist())
print("Curva ROC - thresholds:", thresholds.tolist())

# Guardar los datos de la curva ROC
roc_curve_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "roc_auc": auc(fpr, tpr)}

# Registrar los datos en Weights & Biases
wandb.log({"roc_curve": roc_curve_data})

# Calcular la curva Precisión-Recall usando scikit-learn
precision, recall, thresholds_pr = precision_recall_curve(y_test_binary, y_probas[:, 1])

# Imprimir información para depuración
print("Curva Precisión-Recall - precision:", precision.tolist())
print("Curva Precisión-Recall - recall:", recall.tolist())
print("Curva Precisión-Recall - thresholds:", thresholds_pr.tolist())

# Registrar datos de la curva Precisión-Recall en Weights & Biases
pr_data = [
    {"precision": p, "recall": r, "thresholds": th} for p, r, th in zip(precision, recall, thresholds_pr)
]
wandb.log({"average_precision": average_precision_score(y_test_binary, y_probas[:, 1]), "precision_recall_curve": pr_data})

# Visualizar importancia de características
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
wandb.sklearn.plot_feature_importances(model, feature_names=feature_names)

y_pred = (y_probas[:, 1] > 0.5).astype(int)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test_binary, y_pred)

# Imprimir información para depuración
print("Matriz de Confusión:")
print(conf_matrix)

# Registrar la matriz de confusión en Weights & Biases
wandb.log({"Confusion_Matrix": conf_matrix})

# Utilizar wandb.sklearn.plot_roc para la curva ROC
roc_chart = wandb.sklearn.plot_roc(y_test_binary, y_probas[:, 1], labels=labels)
wandb.log({"Curva ROC": roc_chart})

# Visualizar evaluación del clasificador
wandb.sklearn.plot_classifier(model,
                              X_train, X_test,
                              y_train, y_test,
                              y_pred, y_probas[:, 1],  # Agregar y_pred y y_probas aquí
                              model_name='RandomForest',
                              labels=labels,
                              is_binary=True)

# Finalizar corrida en W&B
wandb.finish()
