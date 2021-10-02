# Author: Anh Dang
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model

def get_classifer(classifier, params):
    if "SVC" == classifier:
        # Linear Kernel
        return svm.SVC(C=params['C'], degree=params['degree'], gamma=params['gamma'])
    elif "MLPClassifier" == classifier:
        return MLPClassifier()
    elif "LinearRegression" == classifier:
        # return linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
        return linear_model.LinearRegression()
