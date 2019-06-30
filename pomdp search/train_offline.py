import tensorflow as tf
import numpy as np
from models import *
from utils import load_params
import pickle
import copy
import os
import numpy as np
from metrics import action_acc
from data_pipeline import DataPipeline, add_coords_to_data

TRAIN_DIR = os.path.join(os.getcwd(), "Datasets/train")

def main():
    params = load_params("dataset_params.yaml")

    if "train_folder" in params.keys():
        TRAIN_DIR = os.path.join(os.getcwd(), "Datasets", params["train_folder"])
    
        
    train_pipeline = DataPipeline(TRAIN_DIR, params)

    model = full_map_cnn(params)
    model.compile(tf.train.AdamOptimizer(learning_rate=0.02), loss='mse', metrics=["mae"])
    

    old = 0
    while train_pipeline.num_passed < params["num_epochs"]:
        cache, next_state = train_pipeline.get_next_batch()
        data = np.concatenate((np.array(cache["object_beliefs"])[..., np.newaxis],  \
                            np.array(cache["obstacle_beliefs"])[..., np.newaxis], \
                            np.array(cache["robots"])[..., np.newaxis], \
                            np.array(cache["other_robots"])[..., np.newaxis], \
                            np.array(cache["goals"])[..., np.newaxis]), axis=-1)
        

        next_data = np.concatenate((np.array(next_state["object_beliefs"])[..., np.newaxis],  \
                            np.array(next_state["obstacle_beliefs"])[..., np.newaxis], \
                            np.array(next_state["robots"])[..., np.newaxis], \
                            np.array(next_state["other_robots"])[..., np.newaxis], \
                            np.array(next_state["goals"])[..., np.newaxis]), axis=-1)

        if bool(params["use_coords"]) == True :        
            data = add_coords_to_data(data, params)
            next_data = add_coords_to_data(next_data, params)

        indices = np.array(range(len(data)))
        np.random.shuffle(indices)
        data = data[indices]
        next_data = next_data[indices]
        
        actions = np.array(cache['actions'])[indices]
        rewards = np.array(cache['rewards'])[indices]
        rewards[rewards == 1] = 0 
        #print(data[:2, :, :, 2])
        #print(next_data[:2, :, :, 2])
        #print(actions[:2])
        #print(rewards[:2])

        next_Q = model.predict(next_data)
        argmax_Q = np.argmax(next_Q, axis = -1)
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
