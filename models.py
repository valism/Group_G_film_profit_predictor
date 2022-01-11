# Import Packages
import pandas as pd
import numpy as np

from fomlads.model.classification import fisher_linear_discriminant_projection
from helper_functions import *


class FisherLinearDiscriminant:
    """
    This class builds Fisher Linear Discriminant Model.
    """
    def __init__(self):
        self.weights = None

    def fit(self, x_train, y_train):
        self.weights = fisher_linear_discriminant_projection(x_train, y_train)

    def predict_proba(self, x_test):
        m = np.mean(x_test, axis=0)
        return np.dot(x_test - m, self.weights)

class LogisticRegression:
    """
    This class builds Logistic Regression Model.
    """
    def __init__(self, lr=0.01, iter=10000):
        self.weights = None
        self.lr = lr
        self.iter = iter

    def sigmoid(self, a):
        return 1.0 / (1 + np.exp(-a))

    def fit(self, X, y):
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        bias = np.ones((X.shape[0], 1))
        np.concatenate((bias, X), axis=1)

        # weights initialization
        self.weights = np.zeros(X.shape[1])

        for i in range(self.iter):
            z = np.dot(X, self.weights)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.weights -= self.lr * gradient

    def predict_proba(self, X):
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        bias = np.ones((X.shape[0], 1))
        np.concatenate((bias, X), axis=1)

        return self.sigmoid(np.dot(X, self.weights))

    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)
