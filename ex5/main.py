# TITANIC - FINAL CORRECT CODE (ALL FIXED)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Load dataset
df = sns.load_dataset('titanic')

print("Original Shape:", df.shape)

# 2. Drop unnecessary columns
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

# 9. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 10. Train model
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

print("Model trained successfully!")

# 11. Display tree
plt.figure(figsize=(20,10))
plot_tree(model,
          feature_names=X.columns,
          class_names=['Not Survived','Survived'],
          filled=True)
plt.show()


print("\nEnter Passenger Details:")

pclass = int(input("Pclass (1/2/3): "))

sex_map = {'male': 'male', 'm': 'male', '1': 'male',
           'female': 'female', 'f': 'female', '0': 'female'}
while True:
    sex_raw = input("Sex (male/female): ").strip().lower()
    if sex_raw in sex_map:
        sex = sex_map[sex_raw]
        break
    print("  Invalid input. Please enter 'male' or 'female'.")

age = float(input("Age: "))
sibsp = int(input("Siblings/Spouses aboard: "))
parch = int(input("Parents/Children aboard: "))
fare = float(input("Fare: "))

# Only accept ports the encoder was actually trained on (C/Q/S)
valid_ports = set(le_embarked.classes_)
while True:
    embarked = input("Embarked (C/Q/S): ").strip().upper()
    if embarked in valid_ports:
        break
    print(f"  Invalid input. Please enter one of: {sorted(valid_ports)}")

# Encode categorical
sex_encoded = le_sex.transform([sex])[0]
embarked_encoded = le_embarked.transform([embarked])[0]

# Apply IQR clipping
fare = max(lower, min(fare, upper))

# Proper DataFrame for scaling (NO warning)
input_df = pd.DataFrame({
    'age': [age],
    'fare': [fare],
    'sibsp': [sibsp],
    'parch': [parch]
})

scaled = scaler.transform(input_df)

# Final input (MATCHES TRAINING EXACTLY - 7 features, same order as X)
new_data = pd.DataFrame({
    'pclass': [pclass],
    'sex': [sex_encoded],
    'age': [scaled[0][0]],
    'sibsp': [scaled[0][2]],
    'parch': [scaled[0][3]],
    'fare': [scaled[0][1]],
    'embarked': [embarked_encoded]
})

# Prediction
prediction = model.predict(new_data)

print("\nPrediction Result:")
if prediction[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger Did NOT Survive")