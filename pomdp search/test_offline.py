import tensorflow as tf
import numpy as np
from models import full_map_cnn
from utils import load_params
import pickle
import copy
import os
import numpy as np
from metrics import action_acc
from pomdp import POMDP
from data_pipeline import DataPipeline, add_coords_to_data

TEST_DIR = os.path.join(os.getcwd(), "Datasets/test")

def main():
    global TEST_DIR
    params = load_params("dataset_params.yaml")
    if "test_folder" in params.keys():
        TEST_DIR = os.path.join(os.getcwd(), "Datasets", params["test_folder"])
    
 
    model = full_map_cnn(params)
    model.compile(tf.train.AdamOptimizer(), loss='mse', metrics=["mae"])
    model.load_weights("weights15")

    pipeline = DataPipeline(TEST_DIR, params)
    while pipeline.num_passed < params["num_epochs"]:
        cache, next_state = pipeline.get_next_batch()
        data = np.concatenate((np.array(cache["object_beliefs"])[..., np.newaxis],  \
                            np.array(cache["obstacle_beliefs"])[..., np.newaxis], \
                            np.array(cache["robots"])[..., np.newaxis], \
                            np.array(cache["other_robots"])[..., np.newaxis], \
                            np.array(cache["goals"])[..., np.newaxis]), axis = -1)
        
        if bool(params["use_coords"]):
            data = add_coords_to_data(data, params)

        pred_actions = np.argmax(model.predict(data), axis=-1)
        print(action_acc(pred_actions, np.array(cache["optimal_actions"])))

if __name__ == "__main__":
    main()
