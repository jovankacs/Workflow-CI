import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# ============================================================
# ARGUMENT PARSER
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=200)
parser.add_argument('--max_depth', type=int, default=5)
parser.add_argument('--min_samples_split', type=int, default=2)
args = parser.parse_args()

# ============================================================
# LOAD DATA
# ============================================================
train_df = pd.read_csv('heart_preprocessing_train.csv')
test_df = pd.read_csv('heart_preprocessing_test.csv')

X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

print("Data berhasil dimuat!")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ============================================================
# TRAINING (tanpa mlflow.start_run karena sudah dihandle MLProject)
# ============================================================
model = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    min_samples_split=args.min_samples_split,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

# Log parameters
mlflow.log_param("n_estimators", args.n_estimators)
mlflow.log_param("max_depth", args.max_depth)
mlflow.log_param("min_samples_split", args.min_samples_split)
mlflow.log_param("random_state", 42)

# Log metrics
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("precision", prec)
mlflow.log_metric("recall", rec)
mlflow.log_metric("f1_score", f1)
mlflow.log_metric("roc_auc", roc_auc)

# Log model
mlflow.sklearn.log_model(model, "model")

# Artefak: Confusion Matrix
os.makedirs("artifacts", exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Tidak Sakit', 'Sakit'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix - CI')
plt.tight_layout()
plt.savefig("artifacts/confusion_matrix.png")
plt.close()
mlflow.log_artifact("artifacts/confusion_matrix.png")

# Artefak: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - CI')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("artifacts/roc_curve.png")
plt.close()
mlflow.log_artifact("artifacts/roc_curve.png")

print("\n===== HASIL TRAINING CI =====")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")
print("=============================")
print("Model berhasil disimpan!")