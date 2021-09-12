# warnings.simplefilter(action='ignore')

from posixpath import dirname, join
from typing import Any
import pandas as pd
import numpy as np

from . import datasets
from os import path
from sklearn.linear_model import LinearRegression


class EvoCC:

    train_file = 'train.csv'
    test_file = 'test.csv'

    def __init__(self, **kwparameters):  
        '''
        we can apply ** to more than one argument in a function call.
        ''' 
        self.optimizer = kwparameters["evo_params"]["optimizer"]
        self.objective_func = kwparameters["evo_params"]["objective_func"]
        self.dataset_list = kwparameters["evo_params"]["dataset_list"]

        if ('classifier' in kwparameters):
            self.classifier = kwparameters["classifier"]
            
        self.dataset = kwparameters["dataset"]
        self.evo_folder = kwparameters["evo_folder"]

    def run(self):
        
        np_dataset_train = np.genfromtxt(join(
            self.dataset, self.train_file), delimiter=',', dtype=np.int32)

        np_dataset_test = np.genfromtxt(join(
            self.dataset, self.test_file), delimiter=',', dtype=np.int32)

        n_train_instances = len(np_dataset_train)

        header_names = ['Dataset', 'Optimizer', 'objfname',
                        'k'] + ['label' + str(i) for i in range(n_train_instances)]

        filename = join(self.evo_folder, "experiment_details_Labels.csv")

        n_train_instances = 105

        NumOfRuns = 1

        df = pd.read_csv(filename, names=header_names, dtype=object)[1:].iloc[:, 4:]
        df2 = pd.read_csv(filename, names=header_names, dtype=object)[1:].iloc[:, 3]

        #iterate through each experiment result (labels). Ex: PSO alg, SSE measure, fist run

        all_results = [0]*NumOfRuns
        for index, row in df.iterrows():
            k = int(df2.iloc[index-1])
            results = [self.dataset_list[0],
                       self.optimizer[0], self.objective_func[0], k]
            np_labels_train = row.to_numpy().astype(int)
            centroids = np.zeros(k, dtype=object)
            distinations = np.zeros(k, dtype=object)
            print('Experiment ' + str(index))

            #iterate through each cluster generated to get appropriate testing instances for it
            for i in range(k):
                # get indices of specific cluster
                train_indices = np.nonzero(np_labels_train == i)
                # get training instances of specific cluster
                np_train_cluster = np_dataset_train[train_indices]
                # print(train_indices)
                # print(np_train_cluster)

                # centroids for each cluster
                centroids[i] = np.mean(np_train_cluster[:, :-1], axis=0)
                # distances for each instance and a centroid
                distinations[i] = np.linalg.norm(
                    np_dataset_test[:, :-1] - centroids[i], axis=1)

            distinations = np.stack(distinations)  # to provide right shape
            # get the indices of the min distances
            np_labels_test = np.argmin(distinations, axis=0)

            #iterate through each cluster generated to and get the training and testing instances for the classifier to run

            sum_score = 0
            all_y_pred = []
            all_y_true = []

            for i in range(k):
                # get training indices of specific cluster
                train_indices = np.nonzero(np_labels_train == i)
                # get training instances of specific cluster
                np_train_cluster = np_dataset_train[train_indices]
                # get features' values for training instances of specific cluster
                np_X_train = np_train_cluster[:, :-1]
                # get labels' values for training instances of specific cluster
                np_Y_train = np_train_cluster[:, -1]

                # get testing indices of specific cluster
                test_indices = np.nonzero(np_labels_test == i)
                # get testing instances of specific cluster
                np_test_cluster = np_dataset_test[test_indices]
                # get features' values for testing instances of specific cluster
                np_X_test = np_test_cluster[:, :-1]
                # get labels' values for testing instances of specific cluster
                np_Y_test = np_test_cluster[:, -1]

                clf = self.classifier

                if len(np_X_test) == 0:
                    ratio = 0
                    score = 0
                else:
                    pipe = clf.fit(np_X_train, np_Y_train)
                    y_pred = pipe.predict(np_X_test)
                    score = pipe.score(np_X_test, np_Y_test)

                    all_y_pred = np.append(all_y_pred, y_pred).astype('int32')
                    all_y_true = np.append(all_y_true, np_Y_test).astype('int32')

                    ratio = len(np_X_test) / len(np_dataset_test)
                    sum_score += score * len(np_X_test)
                results.append(score)
                results.append(ratio)
                print('Score for cluster ' + str(i) + ' is: ' + str(score))

            average_score = sum_score / len(np_dataset_test)

            print('Aggregate accuracy: ' + str(average_score))

            results.append(all_y_true)
            results.append(all_y_pred)
            results.append(average_score)
            results.append(index)
            all_results[index - 1] = results
