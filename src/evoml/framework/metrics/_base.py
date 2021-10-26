from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from scipy.stats.mstats import gmean
import pandas as pd
import numpy as np


def get_confusion_matrix_values(y_test, y_pred):
    cm = multilabel_confusion_matrix(y_test, y_pred)
    return cm.tolist()
    # return [[pd.DataFrame(cm[0, 0]).transpose().iloc[0].tolist(), pd.DataFrame(cm[0, 1]).transpose().iloc[0].tolist()],
    #         [pd.DataFrame(cm[1, 0]).transpose().iloc[0].tolist(), pd.DataFrame(cm[1, 1]).transpose().iloc[0].tolist()]]


def evaluate_the_results(y_test, y_pred):

    # TP, FP, FN, TN = get_confusion_matrix_values(y_test, y_pred)
    # precision = TP / (TP+FP)
    # recall = TP / (TP+FN)
    # f1_score = 2 * precision * recall / (precision + recall)
    # return f1_score, precision, recall

    metrics = score(y_test, y_pred)
    precision = pd.DataFrame(metrics[0]).transpose().iloc[0].tolist()
    recall = pd.DataFrame(metrics[1]).transpose().iloc[0].tolist()
    f1_score = pd.DataFrame(metrics[2]).transpose().iloc[0].tolist()

    return f1_score, precision, recall
