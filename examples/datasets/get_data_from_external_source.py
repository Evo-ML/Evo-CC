from os import getcwd
from evoml.framework import datasets
from pathlib import Path
from posixpath import dirname, join

evo_folder = join(Path(getcwd(),'2021-09-21-22-28-32'))

df, df2 = datasets.get_data_frame_frome_experiment_details_by_dataset(evo_folder, "aniso")

print(df, df2)