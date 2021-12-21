"""
Created on Mon Aug 02 20:00:00 2021
@author: Dang Trung Anh (dangtrunganh@gmail.com)
"""

import glob
from os import getcwd, path
import ast
import csv
from posixpath import join
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import logging
from ast import literal_eval

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.basicConfig(level=logging.DEBUG)


def get_latest_folder(directory):
    latest_file = max(glob.glob(path.join(directory, '*/')), key=path.getmtime)
    return latest_file


def benchmark(func):
    """
    A decorator that prints the function execution duration useful for bench-marking.
    """

    def wrapper(*args, **kwargs):
        t = time.clock()
        res = func(*args, **kwargs)
        print(func.__name__, time.clock()-t)
        return res
    return wrapper


def write_average_to_csv(results_directory, classifiers, classifier_parameters, optimizers, objective_funcs, dataset_list):
    """ The function allows writing average of benchmark results to a csv file.
    """
    details_data_from_file_results = pd.read_csv(results_directory + '/experiment.csv')

    ret_data = []

    n_dataset_list = len(dataset_list)
    n_objective_funcs = len(objective_funcs)
    n_optimizers = len(optimizers)
    n_classifiers = len(classifiers)

    for d in range(n_dataset_list):
        for j in range(0, n_objective_funcs):
            for i in range(n_optimizers):
                for k in range(n_classifiers):
                    objective_name = objective_funcs[j]
                    optimizer_name = optimizers[i]
                    classifier = classifiers[k]
                    classifier_parameter = classifier_parameters[k]
                    detailedData = details_data_from_file_results[(details_data_from_file_results["Dataset"] == dataset_list[d])
                                                                  & (details_data_from_file_results["Optimizer"] == optimizer_name)
                                                                  & (details_data_from_file_results["objfname"] == objective_name)
                                                                  & (details_data_from_file_results["classifier"] == classifier)]
                    n_clusters = detailedData["k"].iloc[0]
                    execution_time_list = np.array(detailedData["ExecutionTime"]).T.tolist()
                    accuracy_list = np.array(detailedData["Accuracy"]).T.tolist()
                    g_mean_list = np.array(detailedData["g-mean"]).T.tolist()
                    f1_score_list = detailedData["f1_score"]
                    precision_list = detailedData["precision"]
                    recall_list = detailedData["recall"]
                    iters_list = detailedData["iters"]
                    # print(average_of_matrix(iters_list))

                    dic_data = [dataset_list[d],
                                classifier,
                                classifier_parameter,
                                optimizer_name,
                                objective_name,
                                n_clusters,
                                average(execution_time_list),
                                average(accuracy_list),
                                average(g_mean_list),
                                average_of_matrix(f1_score_list),
                                average_of_matrix(precision_list),
                                average_of_matrix(recall_list),
                                average_of_matrix(iters_list)
                                ]

                    ret_data.append(dic_data)

    # headers of the output file
    headers_of_csv_file = ["Dataset", "classifier", "classifier_parameter", "Optimizer", "objfname",
                           "k", "ExecutionTime", "Accuracy", "g-mean", "f1_score", "precision", "recall", "iters"]
    average_file = path.join(results_directory, "average.csv")
    write_to_csv_from_list(headers_of_csv_file, ret_data, average_file)


def write_results_to_csv(all_results, parent):
    # experiment_file = 'experiment.csv'

    experiment_details_file = 'experiment.csv'
    file_to_export = path.join(parent, experiment_details_file)
    headers_of_csv_file = np.concatenate([["Dataset", "classifier", "classifier_parameters", "Optimizer", "objfname", "k", "ExecutionTime",
                                           "confusion_matrix", "Accuracy", "g-mean", "f1_score", "precision", "recall", "iters"]])

    write_to_csv_from_list(headers_of_csv_file, all_results, file_to_export)


def plot_boxplot_to_file(results_directory, optimizer, objectivefunc, classifiers, dataset_List, ev_measures):

    plt.ioff()
    fileResultsDetailsData = pd.read_csv(results_directory + '/experiment.csv')
    for d in range(len(dataset_List)):
        for j in range(0, len(objectivefunc)):
            for i in range(len(optimizer)):
                for z in range(0, len(ev_measures)):

                    # Box Plot
                    data = []

                    for k in range(len(classifiers)):
                        objective_name = objectivefunc[j]
                        optimizer_name = optimizer[i]
                        classifier = classifiers[k]
                        detailedData = fileResultsDetailsData[(fileResultsDetailsData["Dataset"] == dataset_List[d])
                                                              & (fileResultsDetailsData["Optimizer"] == optimizer_name)
                                                              & (fileResultsDetailsData["objfname"] == objective_name)
                                                              & (fileResultsDetailsData["classifier"] == classifier)]
                        detailedData = detailedData[ev_measures[z]]
                        detailedData = np.array(detailedData).T.tolist()
                        data.append(detailedData)

                    # , notch=True

                    logging.debug(data)

                    plt.xticks(ha='right', rotation=45)

                    box = plt.boxplot(data, patch_artist=True, labels=classifiers)

                    colors = ['#5c9eb7', '#f77199', '#cf81d2', '#4a5e6a', '#f45b18',
                              '#ffbd35', '#6ba5a1', '#fcd1a1', '#c3ffc1', '#68549d',
                              '#1c8c44', '#a44c40', '#404636']
                    for patch, color in zip(box['boxes'], colors):
                        patch.set_facecolor(color)

                    plt.legend(handles=box['boxes'], labels=classifiers,
                               loc="upper left", bbox_to_anchor=(1.2, 1.02))
                    fig_name = results_directory + "/boxplot-" + \
                        dataset_List[d] + "-" + objective_name + "-" + \
                        optimizer_name + "-"+ev_measures[z] + ".png"
                    plt.savefig(fig_name, bbox_inches='tight')
                    plt.clf()


def plot_convergence_to_file(results_directory, optimizers, objective_funcs, classifiers, dataset_list, iterations):
    """ Plot convergence graph and save to a PNG file
    """
    plt.ioff()
    file_results_data = pd.read_csv(results_directory + '/experiment.csv')
    n_dataset_list = len(dataset_list)
    n_objective_funcs = len(objective_funcs)
    n_optimizers = len(optimizers)
    n_classifiers = len(classifiers)

    for d in range(n_dataset_list):
        for j in range(0, n_objective_funcs):
            objective_name = objective_funcs[j]
            start = 0
            if 'SSA' in optimizers:
                start = 1
            all_generations = [x+1 for x in range(start, iterations)]

            for k in range(n_classifiers):
                classifier_name = classifiers[k]
                temp_lst = []
                for i in range(n_optimizers):
                    optimizer_name = optimizers[i]
                    # print(dataset_list[d], optimizer_name,
                    #       objective_name, classifier_name)
                    file_results_data = file_results_data[[
                        'Dataset', 'Optimizer', 'objfname', 'classifier', 'iters']]
                    rows = file_results_data[(file_results_data["Dataset"] == dataset_list[d])
                                             & (file_results_data["Optimizer"] == optimizer_name)
                                             & (file_results_data["objfname"] == objective_name)
                                             & (file_results_data["classifier"] == classifier_name)
                                             ]
                    logging.debug(rows)
                    rows.reset_index(inplace=True)
                    rows = rows["iters"]

                    list_avg_of_fitness_grb_opt = []

                    for index, row in rows.iteritems():
                        logging.debug(row)
                        list_avg_of_fitness_grb_opt.append(
                            np.array(ast.literal_eval(row), dtype=float))

                    avg = np.matrix(list_avg_of_fitness_grb_opt).mean(0).tolist()[0]
                    if start == 1:
                        avg = np.delete(avg, 0)
                    temp_lst.append(avg)

                logging.debug(temp_lst)

                plt.plot(all_generations, np.matrix(
                    temp_lst).transpose(), label=optimizers)

                plt.xlabel('Iterations')
                plt.ylabel('Fitness')
                plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.02))
                plt.grid()
                fig_name = results_directory + "/convergence-" + \
                    dataset_list[d] + "-" + objective_name + ".png"
                plt.savefig(fig_name, bbox_inches='tight')
                plt.clf()
                break


def write_to_csv_from_list(headers, lst, to__file):
    with open(to__file, 'a', newline='\n') as out_details:
        writer_details = csv.writer(out_details, delimiter=',')
        header_details = np.concatenate([headers])
        writer_details.writerow(header_details)
        for result in lst:
            writer_details.writerow(result)
    out_details.close()


def average(lst):
    return sum(lst) / len(lst)


def average_of_matrix(matrix):

    logging.debug(matrix)

    temp = []

    for value in np.ndenumerate(matrix.values):
        # row = ast.literal_eval(value[1])
        temp.append(ast.literal_eval(value[1]))
    
    logging.debug(temp)

    if (type(temp[0][0]) is str):
        avg = np.array(temp).astype(float).mean(axis=0)
    else:
        avg = np.array(temp).mean(axis=0)

    return avg.tolist()
