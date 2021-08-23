from os import environ, listdir, makedirs
from os.path import dirname, expanduser, isdir, join, splitext, basename
from typing import Tuple
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split


def get_data_home(data_home=None) -> str:
    """Return the path of the evoml-framework data dir.
    """
    if data_home is None:
        data_home = environ.get('EVOML_FRAMEWORK_DATA',
                                join('~', 'evoml_framework_data'))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def split_dataset(src, dst=None, ratio=0.3, cluster=True):
    """ split a dataset
    """
    if (dst is None):
        dst = join(dirname(src), splitext(basename(src))[0]) 
        makedirs(dst, exist_ok=True)

    df = pd.read_csv(src)
    X = df.iloc[:, 0:(len(df.columns)-1)]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=ratio)
    
    pd.concat([X_train, y_train], axis=1).to_csv(join(dst, 'train.csv') , header = False, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(join(dst, 'test.csv') , header = False, index=False)

    if (cluster is True):
        X_train.to_csv(join(dst, 'cluster.csv') , header = False, index=False)