"""
This file contains code that will kick off training and testing processes
"""
import os
import json

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData
from sklearn.model_selection import train_test_split

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = "/home/workspace"
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "/home/workspace/out "

if __name__ == "__main__":
    # Get configuration

    
    c = Config()

    # Load data
    print("Loading data...")

     
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)


    # Create test-train-val split
    

    keys = range(len(data))

. 

    split = dict()

    # create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    train, split['test'] = train_test_split(keys, test_size = 0.15)
    split['train'], split['val'] = train_test_split(train, test_size = 0.20)


    # Set up and run experiment
    
    
    exp = UNetExperiment(c, split, data)


    # run training
    exp.run()

    # prep and run testing


    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

