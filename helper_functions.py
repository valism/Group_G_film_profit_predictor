import pandas as pd
import numpy as np

def get_accuracy(predictions, targets):
    """
    Calculates what proportion of the guesses are correct.
    For example, if 80% of the values match, returns 0.8.
    """

    scores = []
    for i in range(len(targets)):
        if predictions[i] == targets[i]:
            scores.append(1)
        else:
            scores.append(0)
        
    accuracy = sum(scores)/len(targets)
    return accuracy


def evaluate_predictions(y_pred, y_true):
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
    # Number of true positives (tp), false positives (fp), true negatives (tn) and false negatives (fn)
   
    tp, fp, tn, fn = evaluate_predictions(y_pred, y_true)
    f1_score = tp/(tp + 0.5*(fp + fn))
    
    return f1_score


def get_train_test_split(df, train_size = 0.7):
    """
    Splits the dataframe into a training set and a testing set.
    train_size determiens how much of the data goes into the trainig set (0.7 means 70%).
    Returns the training set and testing set.
    """

    # Code for sampling adapted from: https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
    
    train = df.sample(frac = train_size, random_state=100) # Use random_state to fix a seed value
    test = df.drop(train.index)  

    
    return train, test

def log_standardisation(df):
    # Logs of budget and duration columns to help reduce skew, particularly relevant for duration.
    # Budget is highly non-normal so not a huge effect
    df['budget_x_log'] = np.log(df['budget_x'])
    df['budget_y_log'] = np.log(df['budget_y'])
    df['duration_x_log'] = np.log(df['duration_x'])
    df['duration_y_log'] = np.log(df['duration_y'])

    # Standardisation of budget and duration log data
    df_num = df[["duration_x_log", "budget_x_log", "duration_y_log", "budget_y_log","year_x","year_y"]]
    df_num = (df_num - df_num.mean())/(df_num.std())

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
def split(a, n):
    """
    Splits a list into n parts of equal length (or approximiately equal length, if it can't be splt evenly).
    """
    if n == 0:
        raise Exception(" Cannot split a list into 0 parts, enter a valid input for n")
    elif n == 1:
        return a
    else:    
        k, m = divmod(len(a), n)
        return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    

def get_fold_indices(df, num_folds = 10):
    """
    Retruns the indices for each iteration of cross validation.
    For example, if input is has 10 rows and needs 5 folds:
    the first iteration has indices [0, 1] for testing and [2, 3, 4, 5, 6, 7, 8, 9] for training
    the second iteration has indices [2, 3] for testing and [0, 1, 4, 5, 6, 7, 8, 9] for training 
    and so on.
    
    train_indices[0] will give you the row numbers of the training data for the first iteration  
    val_indices[0] will give you the row numbers of the validation data for the first iteration
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

            t = splits[:i] + splits[i+1:]        
            t =  [item for sublist in t for item in sublist]

            train_indices.append(t)
            val_indices.append(v)


        return train_indices, val_indices

        