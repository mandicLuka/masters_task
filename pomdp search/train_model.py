import tensorflow as tf
import numpy as np
from models import full_map_cnn
from utils import load_params
import pickle
import os

TRAIN_DIR = os.path.join(os.getcwd(), "train")
TEST_DIR = os.path.join(os.getcwd(), "test")

def main():
    global DATASET_DIR
    params = load_params("dataset_params.yaml")
    
    if "train_folder" in params.keys():
        TRAIN_DIR = os.path.join(os.getcwd(), params["train_folder"])
    if "test_folder" in params.keys():
        TEST_DIR = os.path.join(os.getcwd(), params["test_folder"])
    
    train_list = os.listdir(TRAIN_DIR)
    test_list = os.listdir(TEST_DIR)
    
    file_batch = params["file_batch"]
    for i in range(params["num_epochs"]):
        count = 0
        while count < len(train_list):
            if count + file_batch >= len(train_list) - 1:
                batch = train_list[count:]
            else:
                batch = train_list[count:count+file_batch]


            for name in batch:
                with open(os.path.join(TRAIN_DIR, name)) as f:
                    trajs, belief = pickle.load(f)
                    for robot in trajs.keys():
                        for position, action,  in trajs[robot]:

            count += file_batch


    
    model = full_map_cnn(params)


if __name__ == "__main__":
    main()