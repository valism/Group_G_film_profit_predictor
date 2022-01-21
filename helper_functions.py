import pandas as pd
import numpy as np


def evaluate_predictions(y_pred, y_true):
    """
    Calculates the number of true positives, false positives, true negatives, and false negatives
    in the predicted classes.

    Args:
        y_pred (numpy array): Contains the predicted class labels.
        y_true (numpy array): Contains the actual class labels.

    Returns:
        tp (int): Number of true positives.
        fp (int): Number of false positives.
        tn (int): Number of true negatives.
        fn (int): Number of false negatives.

    """

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for j in range(len(y_true)):
        p = y_pred[j]
        t = y_true[j]

        if t == 1 and p == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 1 and p == 0:
            fn += 1

    return tp, fp, tn, fn


def get_f1_score(y_pred, y_true):
    """
    Calculates the F1-score of the predictions.

    Args:
        y_pred (numpy array): Contains the predicted class labels.
        y_true (numpy array): Contains the true class labels.

    Returns:
        f1_score (float): The F1-score calculated from the predictions.


    """

    tp, fp, tn, fn = evaluate_predictions(y_pred, y_true)
    f1_score = tp / (tp + 0.5 * (fp + fn))

    return f1_score


def get_train_test_split(df, train_size=0.7):
    """
    Randomly samples and splits the data into a training set and a testing set based on a given split.

    Args:
        df (DataFrame): Contains the data that will be split.
        train_size (float): Proportion of the data that will go into the training set.
                            The remaining data will go into the testing set.

    Returns:
        train (DataFrame): The training set of the data.
        test (DataFrame): The testing set of the data.

    """

    # Code for sampling adapted from: https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas

    train = df.sample(frac=train_size, random_state=100)  # Use random_state to fix a seed value
    test = df.drop(train.index)

    return train, test


def log_standardisation(df):
    """
    Log standardises columns in the data with large numbers, so they don't end up dominating the outcome.
    Applied to the "budget_x", "budget_y", "duration_x", and "duration_y" columns.
    Note that for year_x and year_y the log is not taken because it was causing issues with some samples.
    The model would work one some runs but not on others. Not taking the log fixed this issue.
    We are not sure how taking the log was only causing problems on some runs.

    Args:
        df (DataFrame): Contains the data to be used.

    Returns:
        df (DataFrame): Same as the input "df" but with
        some columns replaced by their log standardised counterparts.

    """
    # Logs of budget and duration columns to help reduce skew, particularly relevant for duration.
    # Budget is highly non-normal so not a huge effect
    df['budget_x_log'] = np.log(df['budget_x'])
    df['budget_y_log'] = np.log(df['budget_y'])
    df['duration_x_log'] = np.log(df['duration_x'])
    df['duration_y_log'] = np.log(df['duration_y'])

    # Standardisation of budget and duration log data
    df_num = df[["duration_x_log", "budget_x_log", "duration_y_log", "budget_y_log", "year_x", "year_y"]]
    df_num = (df_num - df_num.mean()) / (df_num.std())

    # Replace columns in original dataset with standardised data
    df['duration_x_log'] = df_num['duration_x_log']
    df['budget_x_log'] = df_num['budget_x_log']
    df['duration_y_log'] = df_num['duration_y_log']
    df['budget_y_log'] = df_num['budget_y_log']
    df['year_x'] = df_num['year_x']
    df['year_y'] = df_num['year_y']

    df.drop(["budget_x", "budget_y", "duration_x", "duration_y"], axis=1)

    return df


# Code adapted from: https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def split(input_list, n):
    """
    Splits an array (or list) into n approximately equal parts.
    If the list can't be split evenly, one of the parts will be a different length than the others (off by 1).
    Args:
        input_list (Array or List): The Array/List being split.
        n (int): The desired number of parts.

    Returns:
        A 2D list, which has the same items as "a" but split into sub-lists.

    """
    if n == 0:
        raise Exception(" Cannot split a list into 0 parts, enter a valid input for n")
    elif n == 1:
        return input_list
    else:
        k, m = divmod(len(input_list), n)
        return list(input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def get_fold_indices(df, num_folds=5):
    """
    Returns the indices for each iteration of cross validation.
    For example, if input is has 10 rows and needs 5 folds:
    the first iteration has indices [0, 1] for testing and [2, 3, 4, 5, 6, 7, 8, 9] for training
    the second iteration has indices [2, 3] for testing and [0, 1, 4, 5, 6, 7, 8, 9] for training
    and so on.

    train_indices[0] will give you the row numbers of the training data for the first iteration
    val_indices[0] will give you the row numbers of the validation data for the first iteration

    Args:
        df (DataFrame): The data for which cross validation will be performed
        num_folds (int): The number of folds for cross validation.

    Returns:
        train_indices (2D List): Contains the row numbers (indices) of the training data for each iteration.
        val_indices (2D List): Contains the row numbers (indices) of the validation data for each iteration.

    """

    indices = df.index.values.tolist()
    N = len(indices)

    train_indices = []
    val_indices = []

    if num_folds == 0 or num_folds == 1:
        raise Exception(" Cannot cross validate with less than 2 folds")
    else:

        splits = split(indices, num_folds)

        for i in range(num_folds):
            v = splits[i]

            t = splits[:i] + splits[i + 1:]
            t = [item for sublist in t for item in sublist]

            train_indices.append(t)
            val_indices.append(v)

        return train_indices, val_indices
