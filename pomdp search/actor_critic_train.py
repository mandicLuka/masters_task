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

    model = FullMapCNN(params)   

    old = 0
    while train_pipeline.num_passed < params["num_epochs"]:
        states, actions, rewards, returns, next_states = train_pipeline.get_next_batch()

        policy = model.predict(states)
        action = np.argmax(policy, axis=-1)
        prob = np.max(policy, axis=-1)
        indices_Q = (range(data.shape[0]), next_state["actions"])
        indices_A = (range(data.shape[0]), cache["actions"])
        labels = model.predict(data)
        labels[indices_A] = np.array(rewards, dtype="f") + \
                          params["gamma"] * next_Q[indices_Q]
        
        a = model.fit(data, labels, validation_split=0.1)
        print(model.predict(data[:5]))
        if train_pipeline.num_passed > old:
            old = train_pipeline.num_passed
            model.save_weights("weights" + str(old))            

if __name__ == "__main__":
    main()
