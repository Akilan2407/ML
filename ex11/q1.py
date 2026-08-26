import pandas as pd
import numpy as np
class Perceptron:
    def __init__(self, learning_rate=0.5, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
    def activation(self, x):
        if x > 0:
            return 1
        else:
            return 0
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        for epoch in range(self.epochs):
            for i in range(len(X)):
                z = np.dot(X[i], self.weights)
                prediction = self.activation(z)
                error = y[i] - prediction
                self.weights += self.learning_rate * error *  X[i]
    def predict(self, X):
        predictions = []
        for s in X:
            z = np.dot(s, self.weights)
            predictions.append(self.activation(z))
        return predictions
data = pd.read_csv('lineardata.csv')
data = data.dropna()
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
lr=int(input("Enter the learning rate: "))
e=int(input("Enter the number of epochs: "))
lr=lr//10
p1 = Perceptron(lr,e)
p1.fit(X, y)
predictions = p1.predict(X)
print("Weights:", p1.weights)
print("Predictions:", predictions)
print("Actual:", y)