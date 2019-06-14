import gym
import numpy as np
from argparse import ArgumentParser
import sys, os
import gym
import random
from pomdp import POMDP
from utils import load_params
import pickle

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
    num_robots = params["num_robots"]
    num_objects = params["num_objects"]
    for env_i in range(num_envs):
        print ("Generating %d. environment"%(env_i))
        # grid domain object
        if isinstance(num_robots, (list, )):
            params["num_robots"] = random.choice(range(num_robots[0], num_robots[1]+1))
        if isinstance(num_objects, (list, )):
            params["num_objects"] = random.choice(range(num_objects[0], num_objects[1]+1))
        env = gym.make("search_env:multiagent_env-v0", params=params)
        trajs, belief = generate_trajectories(env)

        if not os.path.isdir(dataset_folder): 
            os.mkdir(dataset_folder)
        with open(os.path.join(dataset_folder,"env"+str(env_i+1)), "wb") as f:
            pickle.dump((trajs, belief), f)
        
    print ("Done.")

def generate_trajectories(env):
    robots = range(len(env.robots))
    pomdp = POMDP(env, env.params)
    
    # self.render()
    count = 0
    trajs = dict(zip(list(robots), [[] for i in robots]))
    belief = []
    while not env.done and count < env.params["max_iter"]:   
        belief.append((pomdp.object_belief, pomdp.obstacle_belief))     
        for robot in robots:
            action, robot_env = pomdp.get_optimal_action_for_robot(robot) # robot_env has only 1 robot and 1 goal
            trajs[robot].append((robot_env.robots[0].position, action, robot_env.objects[0].position))
            next_state, obs = env.do_action(env.robots[robot], action)    
            if env.done:
                break                
            pomdp.propagate_obs(next_state, action, obs)
            
        count += 1
        env.render()
    
    return trajs, belief



if __name__ == "__main__":
    main()
