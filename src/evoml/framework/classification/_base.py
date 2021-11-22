# Author: Anh Dang
from typing import Any
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier

# names = [
#     "SVM",
#     "Linear SVM",
#     "SGDClassifier"
#     "Nearest Neighbors",
#     "Naive Bayes",
#     "DecisionTreeClassifier"
# ]


def get_classifer(classifier, kwargs: Any):
    if "LinearSVC" == classifier:
        return svm.LinearSVC().set_params(**kwargs)
    elif "SVM" == classifier:
        return svm.SVC().set_params(**kwargs)
    elif "MLPClassifier" == classifier:
        return MLPClassifier()
    elif "LinearRegression" == classifier:
        return linear_model.LinearRegression().set_params(**kwargs)
    elif "KNeighborsClassifier" == classifier:
        return KNeighborsClassifier().set_params(**kwargs)
    elif "GaussianProcessClassifier" == classifier:
        return GaussianProcessClassifier().set_params(**kwargs)
    elif "DecisionTreeClassifier" == classifier:
        return DecisionTreeClassifier().set_params(**kwargs)
    elif "GaussianNB" == classifier:
        return GaussianNB()
    elif "LogisticRegression" == classifier:
        return linear_model.LogisticRegression().set_params(**kwargs)
    elif "AdaBoostClassifier" == classifier:
        return AdaBoostClassifier().set_params(**kwargs)
    elif "SGDClassifier" == classifier:
        return SGDClassifier().set_params(**kwargs)
