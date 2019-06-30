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
from data_pipeline import DataPipeline, add_coords_to_data
from envs.multiagent_env import *

TRAIN_DIR = os.path.join(os.getcwd(), "Datasets/train")

def main():
    params = load_params("dataset_params.yaml")

    model = full_map_cnn(params)
    model.compile(tf.train.AdamOptimizer(learning_rate=0.001), loss='mse', metrics=["mae"])

    epsilon = 0.9
    for i in range(100):
        epsilon *= 0.9
        position = (random.choice(range(params["num_rows"])),
                    random.choice(range(params["num_cols"])))
        robots = [position]
        goals = [(8, 8)]
        grid = worlds.empty_with_robots_and_objects(params, robots, goals)
        env = MultiagentEnv(params, grid)
        robots = range(len(env.robots))
        pomdp = POMDP(env, params)
        count = 0
        env.render()
        states = []
        actions = []
        rewards = []
        _, R = pomdp.build_mdp(env)
        prev_q = None
        prev_state = None
        prev_action = None
        while not env.done and count < env.params["max_iter"]: 
            object_belief = pomdp.object_belief.reshape(pomdp.shape)
            obstacle_belief = pomdp.obstacle_belief.reshape(pomdp.shape)

            for robot in robots:
                robot_position = np.zeros_like(object_belief)
                other_robots = np.zeros_like(object_belief)
                goals = np.zeros_like(object_belief)
                robot_position[env.robots[robot].position] = 1
                for r in env.get_all_robots_positions_excluding(robot):
                    other_robots[r] = 1
                for g in [o.position for o in pomdp.env.objects]:
                    goals[g] = 1
                data = np.concatenate((object_belief[..., np.newaxis],  \
                                      obstacle_belief[..., np.newaxis], \
                                      robot_position[..., np.newaxis], \
                                      other_robots[..., np.newaxis], \
                                      goals[..., np.newaxis]), axis = -1)[np.newaxis, ...]
                if bool(params["use_coords"]) == True:
                    data = add_coords_to_data(data, params)

                qs = model.predict(data)
                print(qs)
                print(epsilon)
                if random.random() < epsilon:
                    action = random.choice(range(4))
                elif epsilon < 0.05:
                    action = np.argmax(qs[0])
                else:
                    action, _, _ = pomdp.get_optimal_action_for_robot(robot)
                
                if prev_q is not None:
                    next_max_q = np.max(qs)
                    prev_q[0, prev_action] = reward + params["gamma"] * next_max_q
                    model.train_on_batch(prev_state, prev_q)
                
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

if __name__ == "__main__":
    main()
