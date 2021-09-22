# Author: Anh Dang
from sklearn import svm
from sklearn.neural_network import MLPClassifier

def get_classifer(classifier, params):
    if "SVC" == classifier:
        # Linear Kernel
        return svm.SVC(C=params['C'], degree=params['degree'], gamma=params['gamma'])
    elif "MLPClassifier" == classifier:
        return MLPClassifier()
