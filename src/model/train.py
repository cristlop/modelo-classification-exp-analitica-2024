import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import wandb
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data
wbcd = load_breast_cancer()
feature_names = wbcd.feature_names
labels = wbcd.target_names

X_train, X_test, y_train, y_test = train_test_split(wbcd.data, wbcd.target, test_size=0.2)

# Train model, get predictions
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_probas = model.predict_proba(X_test)[:, 1]  # Probability of positive class

# Configuration
project_name = "MLOps-mod-classification-2024"

# Initialize W&B run
run = wandb.init(project=project_name, name="classification")

# Plot class proportions
wandb.sklearn.plot_class_proportions(y_train, y_test, labels)

# Plot learning curve
wandb.sklearn.plot_learning_curve(model, X_train, y_train)

# Calculate ROC curve using scikit-learn
fpr, tpr, thresholds = roc_curve(y_test, y_probas)

# Register ROC curve in Weights & Biases
roc_data = [
    {"fpr": f, "tpr": t, "thresholds": th} for f, t, th in zip(fpr, tpr, thresholds)
]
wandb.log({"roc_curve": roc_data})

# Plot Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_probas)
average_precision = average_precision_score(y_test, y_probas)
wandb.log({"average_precision": average_precision, "precision_recall_curve": wandb.plot.precision_recall_curve(y_test, y_probas)})

# Plot feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
wandb.sklearn.plot_feature_importances(model, feature_names=feature_names)

# Plot classifier evaluation
wandb.sklearn.plot_classifier(model,
                              X_train, X_test,
                              y_train, y_test,
                              model_name='RandomForest',
                              labels=labels,
                              is_binary=True)

# Finish W&B run
wandb.finish()
