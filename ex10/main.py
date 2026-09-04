import numpy as np
import pandas as p
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
iris = load_iris()
X = iris.data
y = iris.target
print("Feature names:")
print(iris.feature_names)
print("\nTarget names:")
print(iris.target_names)
print("\nDataset Shape:")
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(
    max_iter=200,
    random_state=42
)
kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=kf,
    scoring="accuracy"
)
print("\nK-Fold Cross Validation Results")
for i, score in enumerate(scores, 1):
    print(f"Fold {i} Accuracy: {score:.4f}")
print("\nMean Accuracy:",
      round(scores.mean(), 4))
print("Standard Deviation:",
      round(scores.std(), 4))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)
recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)
f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)
print("\nFinal Test Set Performance")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)
