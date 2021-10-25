# Author: Anh Dang
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
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
    elif "KNeighborsClassifier" == classifier:
        return KNeighborsClassifier(3)
    elif "GaussianProcessClassifier" == classifier:
        return GaussianProcessClassifier(1.0 * RBF(1.0))
    elif "LogisticRegression" == classifier:
        return linear_model.LogisticRegression(C=10, penalty='l1',
                                  solver='saga',
                                  multi_class='multinomial',
                                  max_iter=10000)
