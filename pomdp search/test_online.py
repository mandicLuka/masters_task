import tensorflow as tf
import numpy as np
from models import full_map_cnn
from utils import load_params
import pickle
import copy
import os
import numpy as np
from metrics import action_acc
import gym
from pomdp import POMDP
from train_model import DataPipeline, add_coords_to_data

TEST_DIR = os.path.join(os.getcwd(), "Datasets/test")

def main():
    global TEST_DIR
    params = load_params("dataset_params.yaml")
    if "test_folder" in params.keys():
        TEST_DIR = os.path.join(os.getcwd(), "Datasets", params["test_folder"])
    
 
    model = full_map_cnn(params)
    model.compile(tf.train.AdamOptimizer(), loss='mse', metrics=["mae"])
    model.load_weights("weights15")

    for i in range(100):
        env = gym.make("search_env:multiagent_env-v0", params=params)
        robots = range(len(env.robots))
        pomdp = POMDP(env, env.params)
        count = 0
        env.render()
        while not env.done and count < env.params["max_iter"]: 
            object_belief = pomdp.object_belief.reshape(pomdp.shape)
            obstacle_belief = pomdp.obstacle_belief.reshape(pomdp.shape)

            for robot in robots:
                action, reward, robot_env = pomdp.get_optimal_action_for_robot(robot) # robot_env has only 1 robot and 1 goal
                robot_position = np.zeros_like(object_belief)
                other_robots = np.zeros_like(object_belief)
                goals = np.zeros_like(object_belief)
                robot_position[robot_env.robots[0].position] = 1
                for r in env.get_all_robots_positions_excluding(robot):
                    other_robots[r] = 1
                for g in [o.position for o in pomdp.env.objects]:
                    goals[g] = 1
                data = np.concatenate((object_belief[..., np.newaxis],  \
                                      obstacle_belief[..., np.newaxis], \
                                      robot_position[..., np.newaxis], \
                                      other_robots[..., np.newaxis], \
                                      goals[..., np.newaxis]), axis = -1)[np.newaxis, ...]
                qs = model.predict(data)
                pred_action = np.argmax(qs)
                next_state, obs = env.do_action(env.robots[robot], pred_action) 
                if env.done:
                    break                
                pomdp.propagate_obs(next_state, action, obs)
            
            count += 1
            env.render()

if __name__ == "__main__":
    main()
