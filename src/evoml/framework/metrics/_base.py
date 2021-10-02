from sklearn.metrics import confusion_matrix
from scipy.stats.mstats import gmean

def get_confusion_matrix_values(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])
    
def evaluate_the_results(y_test, y_pred):
    TP, FP, FN, TN = get_confusion_matrix_values(y_test, y_pred)
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score, precision, recall