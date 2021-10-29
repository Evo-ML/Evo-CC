from genericpath import exists
from math import e
from os import environ, listdir, makedirs, rmdir, path
from os.path import dirname, expanduser, isdir, join, splitext, basename
from typing import Tuple
import pandas as pd
import shutil
from scipy.sparse import data
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path

from .. import datasets


def _set_data_home(data_home=None):
    if data_home is None:
        data_home = join('~', 'evoml_framework_data')
    environ["EVOML_FRAMEWORK_DATA"] = data_home


def _load(train_file, test_file, name):
    if not exists(train_file) or \
            (test_file is not None and not exists(test_file)):
        raise IOError("Dataset missing! %s" % name)

    train_dataset = np.genfromtxt(train_file, delimiter=',', dtype=np.int32)
    test_dataset = np.genfromtxt(test_file, delimiter=',', dtype=np.int32)

    return train_dataset, test_dataset


def _toense(data):
    X_train, y_train, X_test, y_test = data
    X_train = X_train.toarray()
    if X_test is not None:
        X_test = X_test.toarray()
    return X_train, y_train, X_test, y_test

def _read_experiment_details(data_file, column_names, prefix, offset):
    # Delimiter
    data_file_delimiter = ','

    # The max column count a line in the file could have
    largest_column_count = 0

    # Loop the data lines
    with open(data_file, 'r') as temp_f:
        # Read the lines
        lines = temp_f.readlines()

        for l in lines:
            # Count the column count for the current line
            column_count = len(l.split(data_file_delimiter)) + 1
            
            # Set the new most column count
            largest_column_count = column_count if largest_column_count < column_count else largest_column_count

    # Generate column names (will be 0, 1, 2, ..., largest_column_count - 1)

    cl_names = column_names + [prefix + str(i) for i in range(0, largest_column_count-offset)]

    # Read csv
    return pd.read_csv(data_file, header=None, delimiter=data_file_delimiter, names=cl_names, dtype=object)[1:]


# def _get_df_with_different_columns(data_file):

#     # Delimiter
#     data_file_delimiter = ','

#     # The max column count a line in the file could have
#     largest_column_count = 0

#     # Loop the data lines
#     with open(data_file, 'r') as temp_f:
#         # Read the lines
#         lines = temp_f.readlines()

#         for l in lines:
#             # Count the column count for the current line
#             column_count = len(l.split(data_file_delimiter)) + 1
            
#             # Set the new most column count
#             largest_column_count = column_count if largest_column_count < column_count else largest_column_count

#     # Generate column names (will be 0, 1, 2, ..., largest_column_count - 1)

#     column_names = ['Dataset', 'Optimizer', 'objfname', 'k'] + ['label' + str(i) for i in range(0, largest_column_count-4)]

#     # Read csv
#     return pd.read_csv(data_file, header=None, delimiter=data_file_delimiter, names=column_names, dtype=object)[1:]


def load_dataset(name):
    data_home = get_data_home()
    train_file = join(data_home, name, "train.csv")
    test_file = join(data_home, name, "test.csv")
    return _toense(_load(train_file, test_file, name))


def get_data_home(data_home=None) -> str:
    """Return the path of the evoml-framework data dir.
    """
    if data_home is None:
        data_home = environ.get('EVOML_FRAMEWORK_DATA',
                                join('~', 'evoml_framework_data'))
    # update new path
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def split_dataset(src, dst=None, ratio=0.3, cluster=False):
    """ divide a dataset into train and test sets
    """
    if (dst is None):
        dst = join(dirname(src), splitext(basename(src))[0])
        shutil.rmtree(dst, ignore_errors=True)
        makedirs(dst, exist_ok=True)

    df = pd.read_csv(src)
    X = df.iloc[:, 0:(len(df.columns)-1)]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio)

    pd.concat([X_train, y_train], axis=1).to_csv(
        join(dst, 'train.csv'), header=False, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(
        join(dst, 'test.csv'), header=False, index=False)

    if (cluster is True):
        pd.concat([X_train, y_train], axis=1).to_csv(join(Path(dst).parent, basename(src)), header=False, index=False)


def get_dataset(dataset_folder, data_file):

    # dataset_folder = path.join(datasets.get_data_home(), dataset_folder)
    # print("folder"+ dataset_folder)

    return np.genfromtxt(join(
        dataset_folder, data_file), delimiter=',', dtype=np.int32)


def get_data_frame_frome_experiment_details_by_dataset(evo_folder, dataset):

    experiment_details_Labels_file = join(Path(evo_folder), "experiment_details_Labels.csv")
    experiment_details_file = join(Path(evo_folder), "experiment_details.csv")

    iterations = _read_experiment_details(experiment_details_file, ["Dataset","Optimizer","objfname","k","ExecutionTime","SSE","Purity","Entropy","HS","CS","VM","AMI","ARI","Fmeasure","TWCV","SC","Accuracy","DI","DB","STDev"], "Iter", 20)
    iterations = iterations.loc[iterations['Dataset'] == dataset]
    iterations = iterations.iloc[:,20:]
    
    df = _read_experiment_details(experiment_details_Labels_file, ['Dataset', 'Optimizer', 'objfname', 'k'],"label", 4)

    df = df.loc[df['Dataset'] == dataset]

    df = df.dropna(axis=1, how='all')
    iterations = iterations.dropna(axis=1, how='all')

    df0 = df.iloc[:,1:3]
    
    df1 = df.iloc[:, 4:]
    
    df2 = df.iloc[:, 3:4]

    _re_index(df0)

    _re_index(df1)

    _re_index(df2)

    _re_index(iterations)

    print(iterations)
    
    return df0, df1, df2, iterations

def _re_index(df):
    # re-index df (1, 2, 3...)
    df_index =[]

    for i in range(0, len(df)):
        df_index.append(i+1)

    df.index = (df_index)



