from envs.multiagent_env import MultiagentEnv
import numpy as np
from argparse import ArgumentParser
import sys, os
import random
from pomdp import POMDP
from utils import load_params
import pickle
from queue import Queue
import worlds
from time import sleep, clock
from threading import Thread, Lock

NUM_GENERATORS = 3
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
    generators = []

    if not os.path.isdir(dataset_folder): 
        os.mkdir(dataset_folder)
    for i in range(NUM_GENERATORS):
        generators.append(Generator("g"+str(i), params))
        generators[-1].start()
    
    
    while env_i < num_envs:
        # grid domain object
        for gen in generators:
            trajs, belief = [], []
            if gen.done:
                trajs, belief = gen.trajs, gen.belief
                gen.ready = True
                gen.done = False
                if trajs and belief:
                    with open("gen.txt", "a") as f:
                        print("Generated " + str(env_i+1) + " environment.\n")
                        f.write("Generated " + str(env_i+1) + " environment.\n")
                    with open(os.path.join(dataset_folder, "env"+str(env_i+1)), "wb") as f:
                        pickle.dump((trajs, belief), f)
                        env_i += 1
            if env_i >= num_envs:
                break

    for gen in generators:
        gen.join()
    print("Done.")


class Generator(Thread):

    def __init__(self, thread_id, params,):
        Thread.__init__(self)

        self.id = thread_id
        self.params = params
        self.ready = False
        self.done = True
        self.trajs = []
        self.belief = []


    def run(self):
        while True:
            if self.ready:
                self.done = False
                self.ready = False
                env = MultiagentEnv(params=self.params)
                self.generate_trajectories(env)
                self.done = True

            sleep(1)

    def generate_trajectories(self, env):
        robots = range(len(env.robots))
        pomdp = POMDP(env, env.params)
        
        #env.render()
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
                step["robot_position"] = robot_env.robots[0].position
                step["other_robots"] = env.get_all_robots_positions_excluding(robot)
                step["reward"] = reward
                step["optimal_action"] = action
                step["goals"] = [o.position for o in pomdp.env.objects]
                step["robot_goal"] = robot_env.objects[0].position
                if random.random() < env.params["epsilon"]:
                    action = random.choice(range(env.params["num_actions"]))
                step["action"] = action
                next_state, obs = env.do_action(env.robots[robot], action) 
                step["obs"] = env.map_obs(obs)
                step["return"] = pomdp.get_maximum_return()
                
                trajs[robot].append(step)
                if env.done:
                    break                
                pomdp.propagate_obs(next_state, action, obs)
            
            
            #env.render()
            count += 1

        if count >= env.params["max_iter"]:
            self.trajs = []
            self.belief = []
        else:
            self.trajs = trajs
            self.belief = belief

if __name__ == "__main__":
    main()
