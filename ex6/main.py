import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset
data = p.read_csv("Walmart.csv")
# Keep only required columns
data = data[['Temperature', 'Weekly_Sales']]
# Check nulls
print("Null values:\n", data.isnull().sum())
# Drop duplicates
data = data.drop_duplicates()
# Define X and y
X = data[['Temperature']]
y = data['Weekly_Sales']
# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Model
model = LinearRegression()
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
# Slope & Intercept
slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope (m): {slope}")
print(f"Intercept (c): {intercept}")
# Performance Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R2 Score: {r2}")
# Best Fit Line Equation
print(f"\nBest Fitting Line Equation:")
print(f"Weekly_Sales = {slope:.2f} * Temperature + {intercept:.2f}")