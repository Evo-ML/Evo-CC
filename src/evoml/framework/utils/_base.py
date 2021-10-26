import glob
import os
from os import getcwd, path
import pathlib
import csv
from posixpath import join
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_latest_folder(directory):
    latest_file = max(glob.glob(os.path.join(directory, '*/')), key=os.path.getmtime)
    return latest_file


def benchmark(func):
    """
    A decorator that prints the function execution duration useful for bench-marking.
    """
    import time

    def wrapper(*args, **kwargs):
        t = time.clock()
        res = func(*args, **kwargs)
        print(func.__name__, time.clock()-t)
        return res
    return wrapper


def write_average_to_csv(results_directory, optimizer, objectivefunc, classifiers, dataset_List):
    fileResultsDetailsData = pd.read_csv(results_directory + '/experiment.csv')
    ret_data = []
    for d in range(len(dataset_List)):
        for j in range(0, len(objectivefunc)):
            
            for i in range(len(optimizer)):
                for k in range(len(classifiers)):
                    objective_name = objectivefunc[j]
                    optimizer_name = optimizer[i]
                    classifier = classifiers[k]
                    detailedData = fileResultsDetailsData[(fileResultsDetailsData["Dataset"] == dataset_List[d])
                                                            & (fileResultsDetailsData["Optimizer"] == optimizer_name)
                                                            & (fileResultsDetailsData["objfname"] == objective_name)
                                                            & (fileResultsDetailsData["classifier"] == classifier)]
                    detailedData = detailedData["Accuracy"]
                    detailedData = np.array(detailedData).T.tolist()

                    # dic_data = {"Dataset": dataset_List[d],
                    #             "Optimizer": optimizer_name,
                    #             "objfname": objective_name,
                    #             "classifier": classifier,
                    #             "detail": detailedData,
                    #             "average": average(detailedData)}

                    dic_data = [dataset_List[d], optimizer_name, objective_name, classifier, average(detailedData)]
                                
                    ret_data.append(dic_data)

    print(ret_data)
    headers = ["Dataset", "classifier", "classifier_parameters", "Optimizer", "objfname", "average"]
    average_file = path.join(results_directory, "average.csv")
    _write_to_csv_from_list(headers, ret_data, average_file)

def _write_to_csv_from_list(headers, lst, to__file):
    with open(to__file, 'a', newline='\n') as out_details:
        writer_details = csv.writer(out_details, delimiter=',')
        header_details = numpy.concatenate([headers])
        writer_details.writerow(header_details)
        for result in lst:
            writer_details.writerow(result)
    out_details.close()

def average(lst):
    return sum(lst) / len(lst)

def write_results_to_csv(all_results, parent):
    # experiment_file = 'experiment.csv'

    experiment_details_file = 'experiment.csv'
    ExportToFileDetails = path.join(parent, experiment_details_file)

    with open(ExportToFileDetails, 'a', newline='\n') as out_details:
        writer_details = csv.writer(out_details, delimiter=',')
        header_details = numpy.concatenate([["Dataset", "classifier", "classifier_parameters", "Optimizer", "objfname", "k", "ExecutionTime",
                                             "confusion_matrix", "Accuracy", "g-mean", "f1_score", "precision", "recall", "iters"]])
        writer_details.writerow(header_details)
        for results in all_results:
            writer_details.writerow(results)
    out_details.close()


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

                    print(data)

                    box = plt.boxplot(data, patch_artist=True, labels=classifiers)

                    colors = ['#5c9eb7', '#f77199', '#cf81d2', '#4a5e6a', '#f45b18',
                              '#ffbd35', '#6ba5a1', '#fcd1a1', '#c3ffc1', '#68549d',
                              '#1c8c44', '#a44c40', '#404636']
                    for patch, color in zip(box['boxes'], colors):
                        patch.set_facecolor(color)

                    plt.legend(handles=box['boxes'], labels=classifiers,
                               loc="upper right", bbox_to_anchor=(1.2, 1.02))
                    fig_name = results_directory + "/boxplot-" + \
                        dataset_List[d] + "-" + objective_name + "-" + \
                        optimizer_name + "-"+ev_measures[z] + ".png"
                    plt.savefig(fig_name, bbox_inches='tight')
                    plt.clf()
                    # plt.show()


def plot_convergence_to_file(results_directory, optimizer, objectivefunc, classifiers, dataset_List, Iterations):
    plt.ioff()
    fileResultsData = pd.read_csv(results_directory + '/experiment.csv')

    for d in range(len(dataset_List)):
        for j in range (0, len(objectivefunc)):
            objective_name = objectivefunc[j]
            startIteration = 0                
            if 'SSA' in optimizer:
                startIteration = 1             
            allGenerations = [x+1 for x in range(startIteration,Iterations)]   
            for i in range(len(optimizer)):
                optimizer_name = optimizer[i]
                fileResultsData = fileResultsData[['Dataset','Optimizer','objfname','classifier','iters']]
                row = fileResultsData[(fileResultsData["Dataset"] == dataset_List[d]) & (fileResultsData["Optimizer"] == optimizer_name) & (fileResultsData["objfname"] == objective_name)]
                iters = np.array(row.iloc[:, 4:5]["iters"].to_numpy())
                iters[iters == ''] = 0.0
                iters = iters.astype(np.float)
                # iters = iters.astype(np.float)
                print(iters)


                plt.plot(allGenerations, iters, label=optimizer_name)
            plt.xlabel('Iterations')
            plt.ylabel('Fitness')
            plt.legend(loc="upper right", bbox_to_anchor=(1.2,1.02))
            plt.grid()
            fig_name = results_directory + "/convergence-" + dataset_List[d] + "-" + objective_name + ".png"
            plt.savefig(fig_name, bbox_inches='tight')
            plt.clf()
            #plt.show()
