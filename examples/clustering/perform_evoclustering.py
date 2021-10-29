"""
Created on Mon Aug 02 20:00:00 2021
@author: Dang Trung Anh (dangtrunganh@gmail.com)
"""

# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# create or modify environment variable 
# export EVOML_FRAMEWORK_DATA='your_path_to_data_home_folder'
# set EVOML_FRAMEWORK_DATA='your_path_to_data_home_folder'

from EvoCluster import EvoCluster

##EvoCluster parameters

#Select optimizers from the list of available ones: "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS".
optimizer = ["SSA","PSO"]


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
num_of_runs = 1

export_flags = {'Export_avg': True, 'Export_details': True, 'Export_details_labels': True,
                        'Export_convergence': False, 'Export_boxplot': False}

sol = EvoCluster(
    optimizer,
    objective_func,
    dataset_list,
    1,
    evocluseter_params,
    export_flags,
    auto_cluster=True,  # If False, specify a list of integers for n_clusters.
    # string, or list, default = 'supervised' (don't use supervised)
    n_clusters='supervised',
    metric='euclidean'  # It must be one of the options allowed by scipy.spatial.distance.pdist
)

sol.run()