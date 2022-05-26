import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pprint import pprint

data     = pd.read_csv("train.csv")

data.head()

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_test = data[0:8399].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.

data_train = data[8399:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def Init():
  W1 = np.random.rand(10, 784) - 0.5
  b1 = np.random.rand(10, 1) - 0.5
  W2 = np.random.rand(10, 10) - 0.5
  b2 = np.random.rand(10, 1) - 0.5
  return W1, b1, W2, b2

def Activation_ReLU(Z):
  result = np.maximum(Z, 0)
  return result

def Activation_softmax(Z):
  result = np.exp(Z) / sum(np.exp(Z))
  return result

def Forward(W1, b1, W2, b2, X):
  Z1 = W1.dot(X) + b1
  A1 = Activation_ReLU(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = Activation_softmax(Z2)
  return Z1, A1, Z2, A2

def Activation_ReLU_deriv(Z):
  result = Z > 0
  return result

def OneHot(Y):
  OneHot_Y = np.zeros((Y.size, Y.max() + 1))
  OneHot_Y[np.arange(Y.size), Y] = 1
  OneHot_Y = OneHot_Y.T
  return OneHot_Y

def Backward(Z1, A1, Z2, A2, W1, W2, X, Y):
  OneHot_Y = OneHot(Y)
  dZ2 = A2 - OneHot_Y
  dW2 = 1 / m * dZ2.dot(A1.T)
  db2 = 1 / m * np.sum(dZ2)
  dZ1 = W2.T.dot(dZ2) * Activation_ReLU_deriv(Z1)
  dW1 = 1 / m * dZ1.dot(X.T)
  db1 = 1 / m * np.sum(dZ1)
  return dW1, db1, dW2, db2

def Update(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
  W1 = W1 - alpha * dW1
  b1 = b1 - alpha * db1    
  W2 = W2 - alpha * dW2  
  b2 = b2 - alpha * db2    
  return W1, b1, W2, b2

def Predictions(A2):
  result = np.argmax(A2, 0)
  return result

def Accuracy(predictions, Y):
  print(predictions, Y)
  result = np.sum(predictions == Y) / Y.size
  return result

def GradientDescent(X, Y, alpha, iterations):
  W1, b1, W2, b2 = Init()
  for i in range(iterations):
      Z1, A1, Z2, A2 = Forward(W1, b1, W2, b2, X)
      dW1, db1, dW2, db2 = Backward(Z1, A1, Z2, A2, W1, W2, X, Y)
      W1, b1, W2, b2 = Update(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
      if i == (iterations-10):
        print("Number of iteration: ", i)
        predictions = Predictions(A2)
        print(Accuracy(predictions, Y))
  return W1, b1, W2, b2

def MakePredictions(X, W1, b1, W2, b2):
  _, _, _, A2 = Forward(W1, b1, W2, b2, X)
  predictions = Predictions(A2)
  return predictions

def IndividualPrediction(index, W1, b1, W2, b2, X, Y):
  image = X[:, index, None]
  prediction = MakePredictions(X[:, index, None], W1, b1, W2, b2)
  label = Y[index]
  print("Prediction: ", prediction)
  print("Label: ", label)
    
  image = image.reshape((28, 28)) * 255
  plt.gray()
  plt.imshow(image, interpolation='nearest')
  plt.show()

W1, b1, W2, b2 = GradientDescent(X_train, Y_train, 0.5, 600)

IndividualPrediction(10, W1, b1, W2, b2, X_test, Y_test)
IndividualPrediction(300, W1, b1, W2, b2, X_test, Y_test)
IndividualPrediction(9, W1, b1, W2, b2, X_test, Y_test)
IndividualPrediction(10, W1, b1, W2, b2, X_test, Y_test)