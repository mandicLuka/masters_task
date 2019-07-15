import tensorflow as tf
import numpy as np
from models import QCNN
from utils import load_params
import pickle
import copy
import os
from envs.multiagent_env import *
import numpy as np
from metrics import action_acc
from pomdp import POMDP
from data_pipeline import *
import worlds

TEST_DIR = os.path.join(os.getcwd(), "Datasets/test")

def main():
    global TEST_DIR
    params = load_params("dataset_params.yaml")
    if "test_folder" in params.keys():
        TEST_DIR = os.path.join(os.getcwd(), "Datasets", params["test_folder"])
    
 
    model = QCNN(params)
    model.load_model("weights300")
    actions = []
    optim_actions = []
    finished = 0
    counts = []
    for i in range(400):
        goals = [(5, 5)]
        position = goals[0]
        while position in goals:
            position = (random.choice(range(params["num_rows"])),
                        random.choice(range(params["num_cols"])))
        robots = [position]

        grid = worlds.empty_with_robots_and_objects(params, robots, goals)
        env = MultiagentEnv(params=params, grid=grid)
        pomdp = POMDP(env, params)
        count = 0
        env.render()
        robots = [0]
        _, R = pomdp.build_mdp(env) 

        while not env.done and count < env.params["max_iter"]: 
            for robot in robots:
                if params["use_local_data"]:
                    data = get_local_data_as_matrix(robot, env, pomdp, params)
                else:
                    data = get_data_as_matrix(robot, env, pomdp, params)
                data = data[np.newaxis, ...]
                qs = model.predict(data)
                #optim_action, _, _ = pomdp.get_optimal_action_for_robot(robot)
                action = np.argmax(qs[0])
                actions.append(action)
                #optim_actions.append(optim_action)
                state = env.robots[robot].position
                next_state, obs = env.do_action(env.robots[robot], action) 
                reward = R[action, env.ravel_state(state), env.ravel_state(next_state)]
                if env.done:
                    break     
                pomdp.propagate_obs(next_state, action, obs)
                _, R = pomdp.build_mdp(env)           
                #env.render()

            count += 1
        if count < env.params["max_iter"]:
            counts.append(count)
            finished += 1
        #print(np.mean(np.array(actions) == np.array(optim_actions)), finished)
        print(i, finished)
        print(np.mean(np.array(counts)))


if __name__ == "__main__":
    main()
