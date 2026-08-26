import pandas as pd
from sklearn.linear_model import Perceptron
data = pd.read_csv("lineardata.csv")
data = data.dropna()
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
p1 = Perceptron(eta0=0.5,max_iter=10,fit_intercept=False)
p1.fit(X, y)
predictions = p1.predict(X)
print("Weights:", p1.coef_)
print("Predictions:", predictions)
print("Actual:", y)