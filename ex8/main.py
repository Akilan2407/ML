import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df=sns.load_dataset("titanic")
print(df.head(10))    
df = df.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'], axis=1)

# 3. Handle missing values (NO inplace)
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# 4. Drop remaining nulls
df = df.dropna()

# 5. IQR Outlier Removal (fare)
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['fare'] >= lower) & (df['fare'] <= upper)]

# 6. Encode categorical
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df['sex'] = le_sex.fit_transform(df['sex'])
df['embarked'] = le_embarked.fit_transform(df['embarked'])

# 7. Scaling
scaler = StandardScaler()
num_cols = ['age', 'fare', 'sibsp', 'parch']
df[num_cols] = scaler.fit_transform(df[num_cols])

# 8. Features & Target
X = df.drop('survived', axis=1)
y = df['survived']
print("Final Features:", X.columns.tolist())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
print("Model Accuracy:", model.score(X_test,y_test))
coefficients = model.coef_[0]  
intercept = model.intercept_[0]
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Display Logistic Regression Equation
feature_names = X.columns
equation = f"z = {intercept:.4f}"
for feature, coef in zip(feature_names, coefficients):
    if coef >= 0:
        equation += f" + ({coef:.4f} * {feature})"
    else:
        equation += f" - ({abs(coef):.4f} * {feature})"
print("\nLogistic Regression Equation:")
print(equation)
print("\nProbability Equation:")
print("P(Survived=1) = 1 / (1 + e^(-z))")

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=True)

# Plotting
plt.figure(figsize=(8, 5))
# Color positive impacts green and negative impacts red
colors = ['firebrick' if c < 0 else 'forestgreen' for c in importance_df['Coefficient']]
plt.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors, edgecolor='black')
plt.axvline(0, color='black', linestyle='--', alpha=0.7)
plt.title('Titanic Logistic Regression: Feature Importance', fontsize=14, pad=15)
plt.xlabel('Coefficient Value (Impact on Survival Log-Odds)', fontsize=11)
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()