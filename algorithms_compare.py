# Import Packages
import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import matplotlib.axes as axp
from models import *
from helper_functions import *

# Initialize four models as a dictionary, will be input in the later function
models = {'Fishers Linear Discriminant': FishersLinearDiscriminant(),
          'Naive Bayes': GaussianNB(),
          'Random Forest': RandomForestClassifier(),
          'Logistic Regression': LogisticRegression(lr=0.1, iter=1000)
          }

# Initialize the dataframe df
df = pd.read_csv("profit_x_y.csv")
df = df.drop(["Unnamed: 0", "title_x", "title_y"], axis=1)


def test_on_models(data=df, model_dict=models):
    """
    The inputs are the initiated dataset df and the initiated model list which includes four models.
    The function fits the four models in the model list on the input dataset,
    make prediction and save the statistics of fpr, tpr, auc_value for each model in model_stats dict.
    Finally, generate the AUC and draw the ROC.
    """

    #  Split the data into train set and test set
    df_train, df_test = get_train_test_split(data, train_size=0.8)

    # Logs of numerical variables to reduce skewness
    for item in (df_train, df_test):
        log_standardisation(item)

    # fig, axs = plt.subplots(2)
    # plt.subplots_adjust(hspace=1)
    # axs[0].hist(df_train["duration_x_log"], density = True)
    # axs[1].hist(df_test["budget_x_log"], density = True)
    # axs[0].set(xlabel="Film duration (mins)", ylabel="Density", title="Log and standardised film duration")
    # axs[1].set(xlabel="Budget ($USD)", ylabel="Density", title="Log and standardised budget")

    # plt.savefig("scaled_budget_duration_plots.jpg")

    # Re-calculates the indices for train and test set
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    # Create an empty dictionary to store the statistics of models
    model_stats = {}

    # X and y data for training set
    x_train = df_train.drop("profit_xy", axis=1).to_numpy()
    y_train = df_train["profit_xy"].to_numpy()

    # X and y data for test set
    x_test = df_test.drop("profit_xy", axis=1).to_numpy()
    y_test = df_test["profit_xy"].to_numpy()

    # Get fpr, tpr, auc_value for model and save them in model_stats dict
    for name, model in model_dict.items():
        fpr, tpr, auc_value = experiment_on_model(model, x_train, y_train, x_test, y_test)
        model_stats[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_value}

    # Plot ROC and get AUC
    for m in model_dict.keys():
        plt.plot(model_stats[m]['fpr'], model_stats[m]['tpr'],
                 label=f"{m} ( Area = {round(model_stats[m]['auc'], 3)} )")

    plt.title("ROC curve for Four Models")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    plt.legend()
    plt.show()


def experiment_on_model(model, x_train, y_train, x_test, y_test):
    """
    The inputs are the model, the x, y in train set and the x, y in test set.
    The function will fit the model in the model list on train set,
    get the predicted probability, and return the statistics of
    fpr, tpr, auc_value for the input model.
    """
    classifier = model
    classifier.fit(x_train, y_train)
    y_score = classifier.predict_proba(x_test)
    try:
        fpr, tpr, threshold = roc_curve(y_test, y_score[:, 1])
    except:
        fpr, tpr, threshold = roc_curve(y_test, y_score)
    auc_value = auc(fpr, tpr)
    return fpr, tpr, auc_value


test_on_models()
