import tensorflow as tf
import numpy as np
from models import *
from utils import load_params
import pickle
import copy
import os
import numpy as np
from metrics import action_acc
from data_pipeline import SingleStateDataPipeline

TRAIN_DIR = os.path.join(os.getcwd(), "Datasets/train")

def main():
    params = load_params("dataset_params.yaml")

    if "train_folder" in params.keys():
        TRAIN_DIR = os.path.join(os.getcwd(), "Datasets", params["train_folder"])
    
        
    train_pipeline = SingleStateDataPipeline(TRAIN_DIR, params, shuffle=True)

    model = PolicyCNN(params)    

    old = 0
    while train_pipeline.num_passed < params["num_epochs"]:
        states, actions, _, _, _ = train_pipeline.get_next_batch()

        labels = np.zeros((actions.shape[0], params["num_actions"]))
        labels[:, actions] = 1
        
        model.train(states, labels)
        if train_pipeline.num_passed > old:
            old = train_pipeline.num_passed
            model.save_weights("weights" + str(old))            

if __name__ == "__main__":
    main()
