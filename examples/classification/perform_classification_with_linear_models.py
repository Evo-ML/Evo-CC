"""
Created on Mon Aug 02 20:00:00 2021
@author: Dang Trung Anh
"""
import os
from posixpath import join
from sklearn.linear_model import LinearRegression
import sys

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from EvoCluster import EvoCluster
from evoml.framework import EvoCC, evocc
from evoml.framework import datasets

data_home = datasets.get_data_home(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data')))

# export EVOML_FRAMEWORK_DATA='../../data'
# data_home = datasets.get_data_home()

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

# divid dataset into train and test set
# datasets.split_dataset(os.path.join(data_home, "iris.csv"))

dataset_folder = join(data_home, "iris")
output_of_evocluster = join(data_home, "iris")

# run evocluster
# evocluster.run(dataset_folder, output_of_evocluster)

# run evocc
evo_params = {'optimizer': optimizer,
              'objective_func': objective_func,
              'dataset_list': dataset_list}

# Initialize our classifier
clf = LinearRegression()

evocc = EvoCC(evo_params=evo_params,
              evo_folder=output_of_evocluster,
              classifier=clf,
              dataset=dataset_folder
)

evocc.run()
