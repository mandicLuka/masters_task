import tensorflow as tf
import numpy as np
from models import *
from utils import load_params
import pickle
import copy
import os
import worlds
from pomdp import POMDP
import numpy as np
from metrics import action_acc
from data_pipeline import *
from envs.multiagent_env import *

TRAIN_DIR = os.path.join(os.getcwd(), "Datasets/train")

def main():
    params = load_params("dataset_params.yaml")

    model = FullMapCNN(params)
    model.load_model("weights300")

    epsilon = 0.7
    for i in range(100):
        #epsilon *= 0.92
        t = random.choice([0, 1])
        goals = [(5, 5)]
        if t == 0:
            position = (random.choice([0, params["num_rows"]-1]),
                        random.choice(range(params["num_cols"])))
        else:
            position = (random.choice(range(params["num_rows"])),
                        random.choice([0, params["num_cols"]-1]))
        robots = [position]
        grid = worlds.empty_with_robots_and_objects(params, robots, goals)
        env = MultiagentEnv(params, grid)
        robots = range(len(env.robots))
        pomdp = POMDP(env, params)
        count = 0
        env.render()
        _, R = pomdp.build_mdp(env)
        prev_q = None
        prev_state = None
        prev_action = None
        print(epsilon)
        while not env.done and count < env.params["max_iter"]: 
            for robot in robots:
                if params["use_local_data"]:
                    data = get_local_data_as_matrix(robot, env, pomdp, params)
                else:
                    data = get_data_as_matrix(robot, env, pomdp, params)
                data = data[np.newaxis, ...]
                qs = model.predict(data)
                #print(qs)
                if random.random() < epsilon:
                    action = random.choice(range(4))
                else:
                    action, _, _ = pomdp.get_optimal_action_for_robot(robot)
                
                if prev_q is not None:
                    next_max_q = np.max(qs)
                    prev_q[0, prev_action] = reward + params["gamma"] * next_max_q
                    model.train(prev_state, prev_q)
                
                prev_state = copy.deepcopy(data)
                prev_q = copy.deepcopy(qs)
                prev_action = action
                next_state, obs = env.do_action(env.robots[robot], action) 
                reward = R[action, env.ravel_state(env.robots[robot].position), env.ravel_state(next_state)]
                _, R = pomdp.build_mdp(env)

                if env.done:
                    break                
                pomdp.propagate_obs(next_state, action, obs)
            
            count += 1
            env.render()

    model.save_model("nakon_online")

if __name__ == "__main__":
    main()
