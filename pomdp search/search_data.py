from envs.multiagent_env import MultiagentEnv
import numpy as np
from argparse import ArgumentParser
import sys, os
import random
from pomdp import POMDP
from utils import load_params
import pickle
import worlds

PARAMS_FILE_PATH = "dataset_params.yaml"

def main():

    params = load_params(PARAMS_FILE_PATH)

    dataset_folder = os.path.join(os.getcwd(), params["dataset_folder"])
    if not os.path.isdir(dataset_folder): 
        os.mkdir(dataset_folder)

    # training data
    create_dataset(os.path.join(dataset_folder, 'train_greedy'), params["train_envs"], params=params)
    # test data
    create_dataset(os.path.join(dataset_folder, 'test_greedy'), params["test_envs"], params=params)

    params["obstacle_prob"] = 0

    # training data
    create_dataset(os.path.join(dataset_folder, 'train_empty'), params["train_envs"], params=params)
    # test data
    create_dataset(os.path.join(dataset_folder, 'test_empty'), params["test_envs"], params=params)



def create_dataset(dataset_folder, num_envs, params):

    # randomize seeds, set to previous value to determinize random numbers
    np.random.seed()
    random.seed()
    env_i = 0
    while env_i < num_envs:
        print ("Generating %d. environment"%(env_i + 1))
        # grid domain object
        env = MultiagentEnv(params=params)
        trajs, belief = generate_trajectories(env)
	
        if not os.path.isdir(dataset_folder): 
            os.mkdir(dataset_folder)
        if len(trajs) > 0 and len(belief) > 0:
            with open(os.path.join(dataset_folder,"env"+str(env_i+1)), "wb") as f:
                pickle.dump((trajs, belief), f)
                env_i += 1
    print ("Done.")

def generate_trajectories(env):
    robots = range(len(env.robots))
    pomdp = POMDP(env, env.params)
    
    env.render()
    count = 0
    trajs = dict(zip(list(robots), [[] for i in robots]))
    belief = []
    reward = 0
    while not env.done and count < env.params["max_iter"]: 
        b = dict()
        b["object_belief"] = pomdp.object_belief.reshape(pomdp.shape)
        b["obstacle_belief"] = pomdp.obstacle_belief.reshape(pomdp.shape)
        belief.append(b)

        for robot in robots:
            action, reward, robot_env = pomdp.get_optimal_action_for_robot(robot) # robot_env has only 1 robot and 1 goal
            step = dict()
            step["robot"] = robot_env.robots[0].position
            step["other_robots"] = env.get_all_robots_positions_excluding(robot)
            step["reward"] = reward
            step["optimal_action"] = action
            step["goals"] = [o.position for o in pomdp.env.objects]
            step["action"] = action
            next_state, obs = env.do_action(env.robots[robot], action) 
            step["obs"] = env.map_obs(obs)
            trajs[robot].append(step)
            if env.done:
                break                
            pomdp.propagate_obs(next_state, action, obs)
        
        
        env.render()
        count += 1

    if count >= env.params["max_iter"]:
        return [], []
    return trajs, belief

if __name__ == "__main__":
    main()
