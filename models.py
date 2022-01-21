# Import Packages
import pandas as pd
import numpy as np
from fomlads.model.classification import project_data
from fomlads.plot.exploratory import plot_class_histograms

from fomlads.model.classification import fisher_linear_discriminant_projection
from helper_functions import *


class FishersLinearDiscriminant:
    """
    A class to represent Fisher's Linear Discriminant model
    Attributes:
        weights (numpy array): Contains the projection vector that maximises the Fisher criterion.

    Methods:
        fit: Fits the data to the model by calculating the projection weights.

        predict: Predict the classes given the test data using a threshold for y = w.T * (x - m)
                 where w.T is the transpose of the weights vector, x is the input and m is the mean vector.
                 Default value of the threshold is 0.

        get_roc_curve: Creates an ROC curve by calculating the false positive rates and true positive rates
                       by varying the threshold.

        plot_histogram: Plots a histogram of the projected points. Note that this does not fit
                        the data to the class instance nor make any predictions. It simply returns a plot.
    """

    def __init__(self):
        self.weights = None

    def fit(self, x_train, y_train):
        """
        Fits the data to the model by calculating the projection weights.

        Args:
            x_train (numpy array): Inputs for training
            y_train (numpy array): Targets for training

        """
        self.weights = fisher_linear_discriminant_projection(x_train, y_train)

    def predict(self, x_test, threshold=0):
        """
        Predict the classes given the test data using a threshold for y = w.T * (x - m)
        where w.T is the transpose of the weights vector, x is the input and m is the mean vector.
        If y > threshold, class 1 is predicted, otherwise class 0 is predicted.
        Default value of the threshold is 0.

        Args:
            x_test (numpy array): Input data for making predictions.
            threshold: Threshold for y.

        Returns:
            predictions(List) = Predicted classes for the input data.

        """

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
        """
        Fits the model to training data and creates an ROC curve.
        Args:
            x_train (numpy array): Inputs for training
            y_train (numpy array): Targets for training
            x_test (numpy array): Inputs for testing
            y_test (numpy array): Targets for testing

        Returns:
            fpr (List): Contains false positive rates for ROC curve.
            tpr (List): Contains true positive rates for ROC curve.

        """
        self.fit(x_train, y_train)
        thresholds = np.linspace(-5, 5, 100)
        fpr = []
        tpr = []
        for threshold in thresholds:
            y_pred = self.predict(x_test, threshold=threshold)
            tp, fp, tn, fn = evaluate_predictions(y_pred, y_test)

            tpr.append(tp / (tp + fn))  # true positive rate = ture positives / ( true positives + false negatives)
            fpr.append(fp / (tn + fp))  # false positive rate = false positives / (true negatives + false positives)

        return fpr, tpr

    @staticmethod
    def plot_histogram(inputs, targets):
        """
        Creates a histogram of the projected data points using Fishers criterion.
        Args:
            inputs (numpy array): Contains the inputs of the data.
            targets (numpy array): Contains the targets the data.

        Returns:
            ax (matplotlib axes): Contains a matplotlib histogram of the projected data.

        """
        w = fisher_linear_discriminant_projection(inputs, targets)
        projected_inputs = project_data(inputs, w)
        ax = plot_class_histograms(projected_inputs, targets)
        return ax


class LogisticRegression:
    """
    This class builds a Logistic Regression Model with customized learning rate and number of iterations.
    Attributes:
        weights (numpy array): The weights of the model.
        lr (numeric value): The learning rate of the model.
        iter(integer): The number of train epochs.
    Methods:
        sigmoid: Sigmoid function
        fit: Fit the logistic regression model with X and y.
        predict_proba: Predict the probability of each data.
        predict: Predict the classes with customized threshold.
        get_roc_curve: Creates an ROC curve by calculating the false positive rates and true positive rates
                       by varying the threshold.
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
