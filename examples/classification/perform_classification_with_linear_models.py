"""
Created on Mon Aug 02 20:00:00 2021
@author: Dang Trung Anh
"""
import os
from posixpath import join
import sys

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from EvoCluster import EvoCluster
from evoml.framework import EvoCC
from evoml.framework import datasets

data_home = datasets.get_data_home(
    "/Volumes/MyWorks/Workplace/Philadelphia University/Evo-CC/data/")

datasets.split_dataset(os.path.join(data_home, "iris.csv"))

optimizer = ["SSA"]
objective_func = ["SSE"]
dataset_list = ["cluster"]
num_of_runs = 1
params = {'PopulationSize': 30, 'Iterations': 50}
export_flags = {'Export_avg': False, 'Export_details': False, 'Export_details_labels': True,
                'Export_convergence': False, 'Export_boxplot': False}

# run EvoCluster with iris dataset
evocluster = EvoCluster(
    optimizer,
    objective_func,
    dataset_list,
    num_of_runs,
    params,
    export_flags,
    auto_cluster=False,
    n_clusters=[2],
    labels_exist=False
)

evocluster_folder = join(data_home, "iris")
evocluster.run(evocluster_folder, evocluster_folder)


dataset_folder = join(data_home, "iris")

evocc = EvoCC(optimizer,
              objective_func,
              evocluster_folder,
              dataset_list,
              dataset=dataset_folder)
evocc.run()