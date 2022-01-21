import textwrap

import pandas as pd
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
# import matplotlib.axes as axp
from models import *
from helper_functions import *
from pprint import pprint
import argparse


def cross_validate_model(x_data, y_data, model, num_folds):
    """
    Perform k-fold cross validation using the model and given data.
    Return the F1 scores for each iteration in a list.
    Args:
        x_data (DataFrame): Inputs of the model.
        y_data (DataFrame): Targets of the model.
        model (Object): Class instance of the model.
        num_folds (int): Number of folds used for cross validation

    Returns:
        scores (list): F1 scores for each iteration.

    """
    # Keeps track of cross validation scores for each iteration.
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


def get_roc_data(x_train, y_train, x_test, y_test, classifiers):
    """
    Gets the false positive rates (fpr), true positive rates (tpr) and area below ROC curve (auc)
    for each model in classifiers.

    Args:
        x_train (DataFrame): Inputs for training.
        y_train (DataFrame): Targets for training.
        x_test (DataFrame): Inputs for testing.
        y_test (DataFrame): Outputs for testing.
        classifiers (Dictionary): The models to be used, keys of the dictionary are model names and
                                  values are class instances.

    Returns:
        roc_dict (dictionary): Dictionary containing model names as keys and fpr list, tpr list and auc as values.

    """

    roc_dict = {}

    for name, model in classifiers.items():
        # Lab models
        if name == "Fishers Linear Discriminant" or name == "Logistic Regression":
            fpr, tpr = model.get_roc_curve(x_train.to_numpy(),
                                           y_train.to_numpy(),
                                           x_test.to_numpy(),
                                           y_test.to_numpy())

            auc = round(abs(np.trapz(tpr, fpr)), 4)
            roc_dict[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}
        # External (sklearn) models.
        else:
            model.fit(x_train, y_train)
            y_score = model.predict_proba(x_test)
            try:
                fpr, tpr, threshold = roc_curve(y_test, y_score[:, 1])
            except:
                fpr, tpr, threshold = roc_curve(y_test, y_score)

            auc = round(abs(np.trapz(tpr, fpr)), 4)
            roc_dict[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}

    return roc_dict


def print_cross_validation_results(cross_validation_scores):
    """
    Neatly prints the results from cross validation of the models.

    Args:
        cross_validation_scores (Dictionary): Contains the name (as keys) and a list
        of cross validation scores ( as values) for each model.

    """

    print("\n --------------- CROSS VALIDATION SCORES (F1 Scores): ---------------")
    pprint(cross_validation_scores)

    print("\n Mean cross validation scores:")
    for classifier in cross_validation_scores.keys():
        mean_score = sum(cross_validation_scores[classifier]) / len(cross_validation_scores[classifier])
        print(f"\t {classifier} --> {round(mean_score, 4)}")
    print("-" * 70)


def test_models(data, classifiers):
    """
    Performs cross validation on classifiers and plots their ROC curves.
    Args:
        data (dataframe): The pre-processed movie data (located in the csv file: "profit_x_y.csv")
        classifiers (Dictionary): Contains the name (as key) and class instance (as value) of each classifier.

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

    print_cross_validation_results(cross_validation_scores)

    # Getting roc curves:
    roc_dict = get_roc_data(x_train, y_train, x_test, y_test, classifiers)

    print("\n Area below ROC curve:")

    # Plot ROC curves for the model
    for model_name in roc_dict.keys():
        print(f"\t {model_name} --> {round(roc_dict[model_name]['auc'], 3)}")

        plt.plot(roc_dict[model_name]['fpr'],
                 roc_dict[model_name]['tpr'],
                 label=f"{model_name} ( Area = {round(roc_dict[model_name]['auc'], 3)} )")

    plt.title("ROC curve for classifiers")

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    plt.legend()

    plt.savefig("ROC Curves for Classifiers.png")

    plt.show()


if __name__ == "__main__":
    """
    Run file from the pycharm terminal using the command "python evaluate_classifiers.py profit_x_y.csv"
    If you want to use Windows' command prompt, you need to change the directory (using the "cd" command)
    to the project directory.
    Then it is recommended that you create a virtual environment if you haven't already.
    Activate the virtual environment and run
    the file using the command "python evaluate_classifiers.py profit_x_y.csv".
    An optional argument called "model_names" exists if you only want to compare selected models
    For example, use "python evaluate_classifiers.py profit_x_y.csv --model_names rf nb"
    to compare Random Forest and Naive Bayes only.
    Note that only the initials are used for model_names.
        fld: Fishers Linear Discriminant
        nb: Naive Bayes
        rf: Random Forest
        lr: Logistic Regression
    Separate the names by just a single space
    """

    # Parser code adapted from Dhaval Patel's YouTube's channel "codebasics", last accessed on 21/1/22.
    # Available at : https://www.youtube.com/watch?v=XYUXFR5FSxI
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Name of the csv file containing the data: profit_x_y.csv")

    # Code adapted from users Seth M Morton and Martin Thoma on stackoverflow.com, last accessed on 21/1/22:
    # Available at:
    #   https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
    parser.add_argument('--model_names',
                        help="Initials of the names of the models you want to use, "
                             "separated by a single space between them."
                             " \"fld\" -> Fishers Linear Discriminant,"
                             "\"nb\" -> Naive Bayes,"
                             "\"rf\" -> Random Forest,"
                             " and \"lr\"-> Logistic Regression."
                             " For example, \"nb rf\" means Naive Bayes and Random Forest ",
                        choices=["fld", "nb", "rf", "lr"],
                        nargs="+",

                        type=str)

    args = parser.parse_args()

    # Initiate the 4 models in a dictionary
    default_models_dict = {'Fishers Linear Discriminant': FishersLinearDiscriminant(),
                           'Naive Bayes': GaussianNB(),
                           'Random Forest': RandomForestClassifier(),
                           'Logistic Regression': LogisticRegression(lr=0.1, iter=1000)
                           }

    models_dict = {}
    if args.model_names is None:
        models_dict = default_models_dict
    else:
        models_list = [item.lower() for item in args.model_names]
        for name in models_list:
            if name == "fld":
                models_dict["Fishers Linear Discriminant"] = FishersLinearDiscriminant()
            elif name == "nb":
                models_dict["Naive Bayes"] = GaussianNB()
            elif name == "rf":
                models_dict["Random Forest"] = RandomForestClassifier()
            elif name == "lr":
                models_dict["Logistic Regression"] = LogisticRegression(lr=0.1, iter=1000)

    print("\n Models selected:")
    for name in models_dict.keys():
        print("\t" + name)

    # Initialize the dataframe df
    df = pd.read_csv(args.filename)
    df = df.drop(["Unnamed: 0", "title_x", "title_y"], axis=1)
    test_models(df, models_dict)
