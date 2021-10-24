import glob
import os
from os import getcwd, path
import pathlib
import csv
from posixpath import join
import numpy

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
        print (func.__name__, time.clock()-t)
        return res
    return wrapper

def write_results_to_csv(all_results, parent):
    # experiment_file = 'experiment.csv'
    experiment_details_file = 'experiment_details.csv'
    os.mkdir(join(parent, "evo"))
    ExportToFileDetails = path.join(parent, "evo", experiment_details_file)
    
    with open(ExportToFileDetails, 'a', newline='\n') as out_details:
        writer_details = csv.writer(out_details, delimiter=',')
        header_details = numpy.concatenate([["Dataset", "classifier", "classifier_parameters", "Optimizer", "objfname", "k", "ExecutionTime",
                                                                        "TP", "TN", "FP", "FN", "Accuracy", "g-mean", "f1_score", "precision", "recall"]])
        writer_details.writerow(header_details)
        for results in all_results:
            writer_details.writerow(results)
    out_details.close()

def plot_boxplot_to_file(all_results, parent):
    print("Exporting ....")

