from genericpath import exists
from os import environ, listdir, makedirs, rmdir
from os.path import dirname, expanduser, isdir, join, splitext, basename
from typing import Tuple
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

def _set_data_home(data_home=None):
    if data_home is None:
        data_home = join('~', 'evoml_framework_data')
    environ["EVOML_FRAMEWORK_DATA"] = data_home

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

def split_dataset(src, dst=None, ratio=0.3, cluster=True):
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
        X_train.to_csv(join(dst, 'cluster.csv'), header=False, index=False)

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

def load_dataset(name):
    data_home = get_data_home()
    train_file = join(data_home, name,"train.csv")
    test_file = join(data_home, name, "test.csv")
    return _toense(_load(train_file, test_file, name))