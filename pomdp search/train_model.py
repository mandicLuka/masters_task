import tensorflow as tf
import numpy as np
from models import *
from utils import load_params
import pickle
import copy
import os
import numpy as np
from metrics import action_acc

TRAIN_DIR = os.path.join(os.getcwd(), "Datasets/train")

def main():
    params = load_params("dataset_params.yaml")

    if "train_folder" in params.keys():
        TRAIN_DIR = os.path.join(os.getcwd(), "Datasets", params["train_folder"])
    
        
    train_pipeline = DataPipeline(TRAIN_DIR, params)

    model = full_map_cnn(params)
    model.compile(tf.train.AdamOptimizer(learning_rate=0.0001), loss='categorical_crossentropy', metrics=[action_acc, "mae"])
    
    shape = (params["num_rows"], params["num_cols"])


    if bool(params["use_coords"]) == True:
        i_channel = np.zeros(shape)
        j_channel = np.zeros(shape)
        rangi = np.linspace(0, 1, shape[0])*2 - 1
        rangj = np.linspace(0, 1, shape[1])*2 - 1

        for i in range(shape[1]):
            i_channel[:, i] = rangi
        for j in range(shape[0]):
            j_channel[j] = rangj

    best = 0
    while train_pipeline.num_passed < params["num_epochs"]:
        cache, next_state = train_pipeline.get_next_batch()
        data = np.concatenate((np.array(cache["object_beliefs"])[..., np.newaxis],  \
                            np.array(cache["obstacle_beliefs"])[..., np.newaxis], \
                            np.array(cache["robots"])[..., np.newaxis], \
                            np.array(cache["other_robots"])[..., np.newaxis], \
                            np.array(cache["goals"])[..., np.newaxis]), axis=-1)

        if bool(params["use_coords"]) == True :
            
            tempi = np.vstack([i_channel[np.newaxis, ...]]*data.shape[0])
            tempj = np.vstack([j_channel[np.newaxis, ...]]*data.shape[0])
            data = np.concatenate((data, tempi[..., np.newaxis], \
                                         tempj[..., np.newaxis]), axis=-1)
            del tempi, tempj
        if data.size == 0:
            continue
        indices = np.array(range(len(data)))
        np.random.shuffle(indices)
        data = data[indices]
        #next_data = next_data[indices]
        actions = np.array(cache['actions'])[indices]
        
        #next_Q = model.predict(next_data)
        #argmax_Q = np.argmax(next_Q, axis=-1)
        #argmax_Q = np.argmax(model.predict(data))
        labels = np.zeros((data.shape[0], params["num_actions"]))
        indices = (range(data.shape[0]), actions)
        labels[indices] = 1
        a = model.fit(data, labels, validation_split=0.1)

        model.save_weights('greedy_weights_coords2')
            
class DataPipeline:

    def __init__(self, path, params):
        self.path = path
        self.listdir = os.listdir(path)
        self.params = params
        self.file_batch = self.params["file_batch"]
        self.count = 0
        self.num_passed = 0
        self.reset = False
    def get_next_batch(self):
        batch = []
        if self.count + self.file_batch >= len(self.listdir) - 1:
            batch = self.listdir[self.count:]
            self.num_passed += 1
            self.reset = True
        else:
            batch = self.listdir[self.count:self.count+self.file_batch]


        shape = (self.params["num_rows"], self.params["num_cols"])
        current_state = {"robots": [],
                         "other_robots": [],
                         "object_beliefs": [], 
                         "obstacle_beliefs": [],
                         "goals": [],
                         "actions": [],
                         "rewards": [],
                         "qs": []}
        next_state = copy.deepcopy(current_state)

        for name in batch:     
            with open(os.path.join(self.path, name), "rb") as f:
                trajs, beliefs = pickle.load(f)
                for j in trajs.keys():
                    for step in range(len(trajs[j])):
                        object_belief = beliefs[step]["object_belief"]
                        obstacle_belief = beliefs[step]["obstacle_belief"]
                        state = trajs[j][step]

                        step_robots = np.zeros(shape)
                        step_other_robots = np.zeros(shape)
                        step_goals =  np.zeros(shape)
                        step_objects = np.zeros(shape)
                        step_obstacles = np.zeros(shape)
                        step_actions = np.zeros(1)
                        step_rewards = np.zeros(1)
                        step_q = np.zeros(1)

                        step_robots[state["robot"]] = 1
                        step_actions = state["action"]
                        step_rewards = state["reward"]
                        step_q = state["q"] 
                        step_objects = object_belief
                        step_obstacles = obstacle_belief
                        for g in state["goals"]:
                            step_goals[g] = 1
                        for r in state["other_robots"]:
                            step_other_robots[r] = 1

                        if step < len(trajs[j]) - 1:
                            current_state["robots"].append(step_robots)
                            current_state["other_robots"].append(step_other_robots)
                            current_state["object_beliefs"].append(step_objects)
                            current_state["obstacle_beliefs"].append(step_obstacles)
                            current_state["goals"].append(step_goals)
                            current_state["actions"].append(step_actions)
                            current_state["rewards"].append(step_rewards)
                            current_state["qs"].append(step_q)
                        if step > 0:
                            pass
                            # next_state["robots"].append(step_robots)
                            # next_state["other_robots"].append(step_other_robots)
                            # next_state["object_beliefs"].append(step_objects)
                            # next_state["obstacle_beliefs"].append(step_obstacles)
                            # next_state["goals"].append(step_goals)
                            # next_state["actions"].append(step_actions)
                            # next_state["rewards"].append(step_rewards)
                            # next_state["qs"].append(step_q)

        self.count += self.file_batch
        if self.reset:
            self.reset = False
            self.count = 0
        return current_state, next_state

if __name__ == "__main__":
    main()