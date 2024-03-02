import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import wandb
from sklearn.metrics import roc_curve, auc, average_precision_score
import matplotlib.pyplot as plt

# Cargar datos
wbcd = load_breast_cancer()

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(wbcd.data, wbcd.target, test_size=0.2)

# Entrenar modelo y obtener predicciones
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_probas = model.predict_proba(X_test)[:, 1]  # Probabilidad de la clase positiva

# Calcular la curva ROC
fpr, tpr, _ = roc_curve(y_test, y_probas)
roc_auc = auc(fpr, tpr)

# Configuración
nombre_proyecto = "MLOps-mod-classification-2024"

# Inicializar la corrida en W&B
run = wandb.init(project=nombre_proyecto, name="classification")

# Guardar los datos de la curva ROC
roc_curve_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "roc_auc": roc_auc}

# Registrar los datos en Weights & Biases
wandb.log({"roc_curve": roc_curve_data})

# Visualizar la curva ROC con matplotlib
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Guardar la imagen en W&B
wandb.log({"roc_curve_plot": plt})

# Finalizar corrida en W&B
wandb.finish()
