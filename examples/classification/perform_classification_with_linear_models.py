"""
Created on Mon Aug 02 20:00:00 2021
@author: Dang Trung Anh
"""
import os
from posixpath import join
# from evoml.framework import datasets

# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# from EvoCluster import EvoCluster
# from evoml.framework import EvoCC, evocc
# from evoml.framework import datasets

# data_home = datasets.get_data_home(os.path.abspath(
#     os.path.join(os.path.dirname(__file__), '../../data')))

# # export EVOML_FRAMEWORK_DATA='../../data'
# data_home = datasets.get_data_home()

from evoml.framework import EvoCC

##EvoCluster parameters

#Select optimizers from the list of available ones: "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS".
optimizer = ["SSA", "PSO", "GA", "GWO"]
# optimizer = ["SSA"]


#Select objective function from the list of available ones:"SSE","TWCV","SC","DB","DI".
objective_func = ["SSE", "TWCV"]
# objective_func = ["SSE"]


#Select data sets from the list of available ones
# dataset_list = ["iris"]
# dataset_list = ["aniso"]

dataset_list = ["aggregation", "aniso"]

#Select general parameters for all optimizers (population size, number of iterations)
evocluseter_params = {'PopulationSize': 30, 'Iterations': 50}

#EvoCC parameters

#Select number of runs for the classification.
num_of_runs = 5

classifiers = ['LogisticRegression']

classifiers_parameters = [
    # {'C': 1, 'degree': 3, 'gamma': 1000},
    # {'hidden_layer_sizes': 100, 'max_iter': 200},
    {},
    # {},
]

sol = EvoCC(
    num_of_runs,
    classifiers,
    classifiers_parameters,
    optimizer,
    objective_func,
    dataset_list,
    evocluseter_params,
    auto_cluster=True,  # If False, specify a list of integers for n_clusters.
    # string, or list, default = 'supervised' (don't use supervised)
    n_clusters='supervised',
    metric='euclidean'  # It must be one of the options allowed by scipy.spatial.distance.pdist
)

# sol = EvoCC(
#     num_of_runs,
#     classifiers,
#     classifiers_parameters,
#     optimizer,
#     objective_func,
#     dataset_list,
#     evocluseter_params,
#     num_of_runs,
#     auto_cluster=False,#If False, specify a list of integers for n_clusters.
#     n_clusters=[2,7],#string, or list, default = 'supervised' (don't use supervised)
#     metric='euclidean'#It must be one of the options allowed by scipy.spatial.distance.pdist
# )

sol.run()
