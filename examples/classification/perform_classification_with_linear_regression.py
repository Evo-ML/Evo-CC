"""
Created on Mon Aug 02 20:00:00 2021
@author: Dang Trung Anh
"""
import os
from posixpath import join

from evoml.framework import EvoCC

##EvoCluster parameters

#Select optimizers from the list of available ones: "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS".
# optimizer = ["SSA", "PSO", "GA", "GWO"]
optimizer = ["SSA"]


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
num_of_runs = 2

classifiers = ['LinearRegression']

classifiers_parameters = [
    {'copy_X': True, 'fit_intercept': True, 'normalize': False},
]

sol = EvoCC(
    num_of_runs,
    classifiers,
    classifiers_parameters,
    optimizer,
    objective_func,
    dataset_list,
    evocluseter_params,
    auto_cluster=False,  # If False, specify a list of integers for n_clusters.
    # string, or list, default = 'supervised' (don't use supervised)
    n_clusters=[2, 2],
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
