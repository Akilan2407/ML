
from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
# Fetch dataset
breast_cancer = fetch_ucirepo(id=14)
# Data
X = breast_cancer.data.features
y = breast_cancer.data.targets.squeeze()
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
# Encode target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
class_counts = pd.Series(y).value_counts()
print("Class Counts Before SMOTE:")
print(class_counts)
plt.figure(figsize=(6,4))
plt.bar(["No Recurrence", "Recurrence"], class_counts.values,
        color=["skyblue", "orange"])
plt.title("Class Count Before SMOTE")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
for i, v in enumerate(class_counts.values):
    plt.text(i, v+2, str(v), ha="center")
plt.show()
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
class_counts_after = pd.Series(y_smote).value_counts()
print("\nClass Counts After SMOTE:")
print(class_counts_after)
plt.figure(figsize=(6,4))
plt.bar(["No Recurrence", "Recurrence"], class_counts_after.values,
        color=["skyblue", "orange"])
plt.title("Class Count After SMOTE")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
for i, v in enumerate(class_counts_after.values):
    plt.text(i, v+2, str(v), ha="center")
plt.show()

