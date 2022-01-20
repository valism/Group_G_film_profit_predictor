# Import Packages
import pandas as pd
import numpy as np
from fomlads.model.classification import project_data
from fomlads.plot.exploratory import plot_class_histograms

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

    def predict(self, x_test, threshold=0):

        m = np.mean(x_test, axis=0)
        predictions = []

        for x in x_test:
            y = np.matmul(self.weights.T, x - m)  # y = w.T * (x - m)

            if y > threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions

    def get_roc_curve(self, x_train, y_train, x_test, y_test):
        self.fit(x_train, y_train)
        thresholds = np.linspace(-5, 5, 100)
        fpr = []
        tpr = []
        for threshold in thresholds:
            y_pred = self.predict(x_test, threshold=threshold)
            tp, fp, tn, fn = evaluate_predictions(y_pred, y_test)

            tpr.append(tp / (tp + fn))
            fpr.append(fp / (tn + fp))

        return fpr, tpr

    @staticmethod
    def plot_histogram(inputs, targets):
        """
        Plots the projected histogram using Fishers model
        """
        w = fisher_linear_discriminant_projection(inputs, targets)
        projected_inputs = project_data(inputs, w)
        ax = plot_class_histograms(projected_inputs, targets)
        return ax

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

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_roc_curve(self, x_train, y_train, x_test, y_test):
        self.fit(x_train, y_train)
        thresholds = np.linspace(0, 1, 100)
        fpr = []
        tpr = []
        for threshold in thresholds:
            y_pred = self.predict(x_test, threshold=threshold)
            tp, fp, tn, fn = evaluate_predictions(y_pred, y_test)

            tpr.append(tp / (tp + fn))
            fpr.append(fp / (tn + fp))

        return fpr, tpr
