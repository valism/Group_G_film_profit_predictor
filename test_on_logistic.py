import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from fomlads.model import classification

movie_data = pd.read_csv('profit_x_y.csv')
drop_list = ['Unnamed: 0', 'title_x', 'title_y', 'profit_x', 'profit_y', 'worlwide_gross_income_x',
             'worlwide_gross_income_y']
y_truth_col = 'profit_xy'


class LogisticRegression:
    def __init__(self, lr=0.01, iter=10000):
        self.weights = None
        self.lr = lr
        self.iter = iter

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def fit(self, X, y):
        bias = np.ones((X.shape[0], 1))
        np.concatenate((bias, X), axis=1)

        # weights initialization
        self.weights = np.zeros(X.shape[1])

        for i in range(self.iter):
            z = np.dot(X, self.weights)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.weights -= self.lr * gradient

    def predict_prob(self, X):
        bias = np.ones((X.shape[0], 1))
        np.concatenate((bias, X), axis=1)

        return self.sigmoid(np.dot(X, self.weights))

    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)


def convert_df_into_np(raw, drop, output):
    df_feature = raw.drop(labels=drop, axis=1)
    df_output = raw[output]
    return np.array(df_feature), np.array(df_output)


def min_max_scaler(data):
    Range = np.max(data) - np.min(data)
    return (data - np.min(data)) / Range


def train_test_split(x, y, test_size, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)

    permutation = list(np.random.permutation(len(x)))
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    test_border = int(test_size * len(x))
    x_train = shuffled_x[test_border:]
    x_train = min_max_scaler(x_train)
    y_train = shuffled_y[test_border:]
    x_test = shuffled_x[:test_border]
    x_test = min_max_scaler(x_test)
    y_test = shuffled_y[:test_border]

    return x_train, y_train, x_test, y_test


def train_and_test(x_train, y_train, x_test, y_test):
    # weights = classification.logistic_regression_fit(x_train, y_train)
    # y_predict = classification.logistic_regression_prediction_probs(x_test, weights)
    logit_model = LogisticRegression(lr=0.01, iter=10000)
    logit_model.fit(x_train, y_train)
    y_predict = logit_model.predict_prob(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_predict)
    roc_area = 1 * np.trapz(tpr, fpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"Logistic ( Area = {round(roc_area, 3)} )")

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    plt.legend()
    plt.show()


raw_x, raw_y = convert_df_into_np(movie_data, drop_list, y_truth_col)
x_train, y_train, x_test, y_test = train_test_split(raw_x, raw_y, test_size=0.25, random_seed=42)
train_and_test(x_train, y_train, x_test, y_test)
