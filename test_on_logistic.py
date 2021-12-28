import pandas as pd
import numpy as np

movie_data = pd.read_csv('profit_x_y.csv')
drop_list = ['Unnamed: 0', 'title_x', 'title_y', 'profit_x', 'profit_y']
y_truth_col = 'profit_xy'


def convert_df_into_np(raw, drop, output):
    df_feature = raw.drop(labels=drop, axis=1)
    df_output = raw[output]
    return np.array(df_feature), np.array(df_output)


def train_test_split(x, y, test_size, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)

    permutation = list(np.random.permutation(len(x)))
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    test_border = int(test_size*len(x))
    x_train = x[test_border:]
    y_train = y[test_border:]
    x_test = x[:test_border]
    y_test = y[:test_border]
    return x_train, y_train, x_test, y_test
