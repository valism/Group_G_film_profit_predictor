import pandas as pd
import random
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
import matplotlib.axes as axp
from models import *
from helper_functions import *

# Initiate the 4 models in a dictionary
models_dict = {'Fishers Linear Discriminant': FisherLinearDiscriminant(),
               'Naive Bayes': GaussianNB(),
               'Random Forest': RandomForestClassifier(),
               'Logistic Regression': LogisticRegression(lr=0.1, iter=1000)
               }


# Initialize the dataframe df
df = pd.read_csv("profit_x_y.csv")
df = df.drop(["Unnamed: 0", "title_x", "title_y"], axis=1)


def cross_validate_model(x_data, y_data, model, num_folds):
    """
    The function will take x, y, model and the number of folds of cross-validation as input.
    The function will reset the indices of train and validation set, fit the model in train set,
    and get fold_f1_score for each fold by comparing the true y and predicted y.
    The function will return the list 'scores' which record the fold_f1_score for folds.
    """

    # Keeps track of the scores for each iteration of cross validation.
    scores = []

    tf_indices, vf_indices = get_fold_indices(x_data, num_folds)

    # Performs cross validation
    for i in range(len(tf_indices)):
        # Training and validation data for the current iteration/
        train_fold_x = x_data.iloc[tf_indices[i]].to_numpy()
        train_fold_y = y_data.iloc[tf_indices[i]].to_numpy()

        # X and y data for validation fold
        validation_fold_x = x_data.iloc[vf_indices[i]].to_numpy()
        validation_fold_y = y_data.iloc[vf_indices[i]].to_numpy()

        model.fit(train_fold_x, train_fold_y)
        y_predictions = model.predict(validation_fold_x)

        # Calculates F1 score for current iteration and appends to list
        fold_f1_score = get_f1_score(y_predictions, validation_fold_y)

        scores.append(round(fold_f1_score, 4))

    return scores


def test_models(data, classifiers):
    """
    The function will dataframe and type of classifier as input.
    The function will split train and validation set, use cross-validation to check if the model works well for different folds,
    and then draw the ROC and get the AUC for each model on test set.
    """

    #  Split the data into train set and test set
    df_train, df_test = get_train_test_split(data, train_size=0.8)

    log_standardisation(df_train)
    log_standardisation(df_test)

    # Re-calculates the indices for train and test set
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    # X and y data for training set
    x_train = df_train.drop("profit_xy", axis=1)
    y_train = df_train["profit_xy"]

    # X and y data for test set
    x_test = df_test.drop("profit_xy", axis=1)
    y_test = df_test["profit_xy"]

    cross_validation_scores = {}
    for name, model in classifiers.items():
        model_scores = cross_validate_model(x_train, y_train, model, num_folds=5)
        cross_validation_scores[name] = model_scores

    print("--------------- CROSS VALIDATION SCORES (F1 Scores): ---------------")
    print(cross_validation_scores)
    print()  # Prints empty line
    print(" Mean cross validation scores:")
    for classifier in cross_validation_scores.keys():
        mean_score = sum(cross_validation_scores[classifier]) / len(cross_validation_scores[classifier])
        print(f"\t {classifier} --> {round(mean_score, 4)}")

    # Getting roc curves:
    roc_dict = {}

    for name, model in classifiers.items():
        if name == "Fishers Linear Discriminant" or name == "Logistic Regression":
            fpr, tpr = model.get_roc_curve(x_train.to_numpy(),
                                           y_train.to_numpy(),
                                           x_test.to_numpy(),
                                           y_test.to_numpy())

            auc = round(abs(np.trapz(tpr, fpr)), 4)
            roc_dict[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}
        else:
            model.fit(x_train.to_numpy(), y_train.to_numpy())
            model.fit(x_train, y_train)
            y_score = model.predict_proba(x_test)
            try:
                fpr, tpr, threshold = roc_curve(y_test, y_score[:, 1])
            except:
                fpr, tpr, threshold = roc_curve(y_test, y_score)

            auc = round(abs(np.trapz(tpr, fpr)), 4)
            roc_dict[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}

    # Plot ROC curves for the model
    for m in roc_dict.keys():
        plt.plot(roc_dict[m]['fpr'], roc_dict[m]['tpr'],
                 label=f"{m} ( Area = {round(roc_dict[m]['auc'], 3)} )")

    plt.title("ROC curve for Four Models")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    plt.legend()
    plt.show()


test_models(df, models_dict)
