# warnings.simplefilter(action='ignore')

from math import e
import math
import pathlib
from posixpath import dirname, join
from typing import Any
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from os import environ, listdir, makedirs, remove, rmdir
import time

from sklearn import metrics as sklearn_metrics
from ..framework import metrics
from imblearn.metrics import geometric_mean_score



from . import datasets, classification
from os import getcwd, path
from . import utils

from EvoCluster import EvoCluster

import pandas as pd
import csv
import numpy


class EvoCC:

    train_file = 'train.csv'
    test_file = 'test.csv'

    def __init__(self, **kwparameters):
        '''
        we can apply ** to more than one argument in a function call.
        '''

    def __init__(self,
                 num_of_runs,
                 classifiers,
                 cls_params,
                 optimizer,
                 objective_func,
                 dataset_list,
                 evocluster_params,
                 auto_cluster: bool,
                 n_clusters,
                 metric
                 ) -> None:

        self.num_of_runs = num_of_runs
        self.classifiers = classifiers
        self.cls_params = cls_params
        self.dataset_list = dataset_list
        self.optimizer = optimizer
        self.objective_func = objective_func
        self.evocluster_params = evocluster_params
        self.auto_cluster = auto_cluster
        self.n_clusters = n_clusters
        self.metric = metric

    def _run_evo_cluster(self, optimizer, objective_func, dataset_list, evocluster_params, auto_cluster, n_clusters, metric):

        # Choose your preferemces of exporting files
        export_flags = {'Export_avg': True, 'Export_details': True, 'Export_details_labels': True,
                        'Export_convergence': False, 'Export_boxplot': False}

        evo_cluster = EvoCluster(
            optimizer,
            objective_func,
            dataset_list,
            1,
            evocluster_params,
            export_flags=export_flags,
            auto_cluster=auto_cluster,
            n_clusters=n_clusters,
            metric=metric
        )

        # run evocluster for each dataset
        evo_cluster.run(path.join(Path(datasets.get_data_home()).parent, self.evo_folder),
                        path.join(Path(datasets.get_data_home()).parent, self.evo_folder))

    def _prepare_data_for_evo_cluster(self, dataset_list, evo_folder):

        folder_after_split_list = []

        for dataset in dataset_list:

            # prepate to split
            dataset_file = path.join(datasets.get_data_home(), dataset + '.csv')
            folder_after_split = path.join(evo_folder, dataset)
            folder_after_split_list.append(folder_after_split)

            # delete folder if it exists, then create new one
            shutil.rmtree(folder_after_split, ignore_errors=True)
            makedirs(folder_after_split, exist_ok=True)

            # split a dateset into train and test sets
            datasets.split_dataset(dataset_file, folder_after_split, cluster=True)

        return folder_after_split_list

    def run(self):

        # after running, it created a folder that contains results (called evo-folder)
        # self.evo_folder = utils.get_latest_folder(getcwd())
        self.evo_folder = time.strftime("%Y-%m-%d-%H-%M-%S")

        # data preperation

        # self.folder_after_split_list = self._prepare_data_for_evo_cluster(
        #     self.dataset_list, self.evo_folder)

        # run evocluster and get results
        # self._run_evo_cluster(self.optimizer,
        #                       self.objective_func,
        #                       self.dataset_list,
        #                       self.evocluster_params,
        #                       self.auto_cluster,
        #                       self.n_clusters,
        #                       self.metric)

        # run evocc for each dataset
        final_results = []

        for num in range(0, self.num_of_runs):
            print("Run no.: " + str(num))
            # split datasets
            self.folder_after_split_list = self._prepare_data_for_evo_cluster(
                self.dataset_list, self.evo_folder)

            # run evo-cluster
            self._run_evo_cluster(self.optimizer,
                                  self.objective_func,
                                  self.dataset_list,
                                  self.evocluster_params,
                                  self.auto_cluster,
                                  self.n_clusters,
                                  self.metric)

            # run evo-cc

            for id_of_data, dataset in enumerate(self.dataset_list):
                print("=== run evocc for " + dataset + " datasets ===")
                for id_of_cl, classifier in enumerate(self.classifiers):
                    results = self._run_classify(
                        dataset, self.folder_after_split_list[id_of_data], classifier, self.cls_params[id_of_cl])
                    for result in results:
                        final_results.append(result)

            # clear data
            for dataset_folder in self.folder_after_split_list:
                try:
                    shutil.rmtree(dataset_folder)
                    remove(dataset_folder+".csv")
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))
            try:
                remove(path.join(self.evo_folder, "experiment.csv"))
                remove(path.join(self.evo_folder, "experiment_details_Labels.csv"))
                remove(path.join(self.evo_folder, "experiment_details.csv"))
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        utils.write_results_to_csv(final_results, self.evo_folder)
        ev_measures = ['g-mean', 'Accuracy']
        utils.plot_boxplot_to_file(self.evo_folder, self.optimizer,
                                   self.objective_func, self.classifiers, self.dataset_list, ev_measures)
        utils.write_average_to_csv(self.evo_folder, self.classifiers, self.cls_params, self.optimizer,
                                   self.objective_func, self.dataset_list)
        
        utils.plot_convergence_to_file(self.evo_folder, self.optimizer,
                                   self.objective_func, self.classifiers, self.dataset_list, self.evocluster_params["Iterations"])

    def _run_classify(self, dataset, folder_after_split, classifier, cls_param):

        print(classifier, cls_param)

        NumOfRuns = 1

        np_dataset_train = datasets.get_dataset(folder_after_split, self.train_file)
        np_dataset_test = datasets.get_dataset(folder_after_split, self.test_file)

        train_instances = len(np_dataset_train)

        df0, df, df2, iters_list = datasets.get_data_frame_frome_experiment_details_by_dataset(
            evo_folder=self.evo_folder, dataset=dataset)

        df.reset_index(inplace=True)
        df2.reset_index(inplace=True)
        df0.reset_index(inplace=True)
        iters_list.reset_index(inplace=True)
        iters_list = iters_list.iloc[:,1:iters_list.shape[1]]

        df = df.iloc[:, :train_instances]
        print(df)
        print(df2)
        print(df0)
        print(iters_list)

        #iterate through each experiment result (labels). Ex: PSO alg, SSE measure, fist run

        # all_results = [0]*NumOfRuns
        all_results = []

        for index, row in df.iterrows():
            # for index in range(1, len(df)):

            print(index)
            print(row)

            t = time.process_time()

            k = int(df2.iloc[index]["k"])

            iters = iters_list.iloc[index].tolist()
            print(iters)

            objfname = df0.iloc[index]["objfname"]
            optimizer = df0.iloc[index]["Optimizer"]
            print(objfname, optimizer)

            results = [dataset, classifier, cls_param, optimizer, objfname, k]
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
            all_y_test = []

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

                # clf = LinearRegression()

                clf = classification.get_classifer(classifier, cls_param)

                if len(np_X_test) == 0:
                    ratio = 0
                    score = 0

                else:
                    pipe = clf.fit(np_X_train, np_Y_train)
                    y_pred = pipe.predict(np_X_test)

                    print("Accuracy on testing set:")
                    score = pipe.score(np_X_test, np_Y_test)

                    all_y_pred = np.append(all_y_pred, y_pred).astype('int32')
                    all_y_test = np.append(all_y_test, np_Y_test).astype('int32')

                    # print(all_y_test)
                    # print(all_y_pred)

                    ratio = len(np_X_test) / len(np_dataset_test)
                    sum_score += score * len(np_X_test)

                # results.append(score)
                # results.append(ratio)

                # print(metrics.classification_report(all_y_test, all_y_pred))

                print('Score for cluster ' + str(i) + ' is: ' + str(score))

            executionTime = time.process_time()-t
            results.append(executionTime)
            

            average_score = sum_score / len(np_dataset_test)

            print('Aggregate accuracy: ' + str(average_score))

            # results.append(all_y_test)
            # results.append(all_y_pred)

            # confusion matrix
            # TP, FP, FN, TN = metrics.get_confusion_matrix_values(all_y_test, all_y_pred)
            # results.append(TP)
            # results.append(FP)
            # results.append(FN)
            # results.append(TN)
            results.append(metrics.get_confusion_matrix_values(all_y_test, all_y_pred))

            results.append(average_score)

            # g-mean
            # sensitivity = TP/(TP+FN)
            # specificity = TN/(FP+TN)
            g_mean = geometric_mean_score(all_y_test, all_y_pred)
            results.append(g_mean)

            # f1_score, precision and recall
            f1_score, precision, recall = metrics.evaluate_the_results(
                all_y_test, all_y_pred)
            results.append(f1_score)
            results.append(precision)
            results.append(recall)
            results.append(iters)

            all_results.append(results)
            

        return all_results
