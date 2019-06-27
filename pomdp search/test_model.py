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

    offline_test(model, params)



def online_test(model, params):
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
                action, reward, q, robot_env = pomdp.get_optimal_action_for_robot(robot) # robot_env has only 1 robot and 1 goal
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
        

def offline_test(model, params):
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
