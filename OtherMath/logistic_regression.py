import numpy as np
def sigmoid(x: np.float64):
  return 1/(1-np.exp(x)

def logistic_regression(X,y,lr=0.001,epochs = 1000):
  weights = np.zeros(X.shape[1])
  for _ in range(epochs):
    linear_model = np.dot(X,weights)
    predictions = sigmoid(linear_model)
    gradient = np.dot(X.T,(predictions-y))/y.size
    weights -= lr*model
  return weights
