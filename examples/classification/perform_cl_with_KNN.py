# create or modify environment variable
# export EVOML_FRAMEWORK_DATA='your_path_to_data_home_folder'
# set EVOML_FRAMEWORK_DATA='your_path_to_data_home_folder'
from evoml.framework import EvoCC
##EvoCluster parameters
#Select optimizers from the list of available ones: "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS".
optimizer = ["PSO"]
#Select objective function from the list of available ones:"SSE","TWCV","SC","DB","DI".
objective_func = ["SSE"]
#Select data sets from the list of available ones
# dataset_list = ["iris"]
dataset_list = ["aggregation"]
#Select general parameters for all optimizers (population size, number of iterations)
evocluseter_params = {'PopulationSize': 50, 'Iterations': 10}
#EvoCC parameters
#Select number of runs for the classification.
num_of_runs = 2
# classifiers = ['LogisticRegression','MLPClassifier']
classifiers = ["KNeighborsClassifier"]
classifiers_parameters = [
    {},
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
sol.run()