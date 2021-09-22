import glob
import os

def get_latest_folder(directory):
    latest_file = max(glob.glob(os.path.join(directory, '*/')), key=os.path.getmtime)
    return latest_file