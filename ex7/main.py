import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
np.random.seed(42)
n = 1000
study_hours = np.random.uniform(0, 10, n)
attd = np.random.uniform(50, 100, n)
assign_marks = np.random.uniform(0, 100, n)
internal_marks = (
    0.4 * study_hours * 10 +
    0.4 * attd +
    0.3 * assign_marks +
    np.random.normal(0, 5, n)
)
internal_marks = np.clip(internal_marks, 0, 100)
result = np.where(internal_marks >= 50, "Pass", "Fail")
df = pd.DataFrame({
    "study_hours": study_hours,
    "attd": attd,
    "assign_marks": assign_marks,
    "internal_marks": internal_marks,
    "result": result
})
# Convert target to numeric (Best Practice)
df["result"] = df["result"].map({"Fail": 0, "Pass": 1})
print("Sample Data:\n", df.head())


X = df.drop(["result", "internal_marks"], axis=1)  # removed leakage feature
y = df["result"]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n--- Decision Tree (No Restriction) ---")
print("Training Accuracy :", train_accuracy)
print("Testing Accuracy  :", test_accuracy)

if train_accuracy > test_accuracy:
    print("Model is Overfitting")
else:
    print("Model is Not Overfitting")
print("Accuracy :", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall   :", recall_score(y_test, y_test_pred))
print("F1 Score :", f1_score(y_test, y_test_pred))

cm_full = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:\n", cm_full)

print("Classification Report:\n")
print(classification_report(y_test, y_test_pred))

dt_pruned = DecisionTreeClassifier(max_leaf_nodes=3,criterion="entropy", random_state=42)
dt_pruned.fit(X_train, y_train)

y_train_pruned = dt_pruned.predict(X_train)
y_test_pruned = dt_pruned.predict(X_test)

train_acc_pruned = accuracy_score(y_train, y_train_pruned)
test_acc_pruned = accuracy_score(y_test, y_test_pruned)

print("\n--- Pre-Pruned Decision Tree  ---")
print("Training Accuracy :", train_acc_pruned)
print("Testing Accuracy  :", test_acc_pruned)

print("Accuracy :", accuracy_score(y_test, y_test_pruned))
print("Precision:", precision_score(y_test, y_test_pruned))
print("Recall   :", recall_score(y_test, y_test_pruned))
print("F1 Score :", f1_score(y_test, y_test_pruned))

cm_pruned = confusion_matrix(y_test, y_test_pruned)
print("Confusion Matrix:\n", cm_pruned)

print("Classification Report:\n")
print(classification_report(y_test, y_test_pruned))