from os import getcwd as pwd
from time import sleep, clock
from utils import load_params
from copy import deepcopy
# from control.nn_models.cnn import ConvNetModel
from threading import Thread, Lock
import random
import numpy as np
import argparse
import time, os
from math import exp
import tensorflow as tf
from queue import Queue
from models import full_map_cnn, FullMapCNN
import worlds
from envs.multiagent_env import *
from pomdp import POMDP
from data_pipeline import *
import tensorflow as tf

NUM_TRAINERS = 2
is_training_done = False

def main():
   
    global is_training_done       
    # import has to be done after path initialization
    params = load_params("dataset_params.yaml")

    trainers = []
    simulation_replays = Queue()    

    net_lock = Lock()
    agent = ReinforcementAgent(thread_id="agent", 
                               params=params, 
                               net_lock=net_lock, 
                               simulation_replays=simulation_replays)
    agent.start()
    sleep(1)
    while not agent.ready:
        sleep(1)

    for i in range(NUM_TRAINERS):
        trainer = ReinforcementTrainer(thread_id="t"+str(i+1), params=params)
        trainer.start()
        trainers.append(trainer)
    
    curr_sim = 0
    num_episodes = int(params["num_episodes"])
    while not is_training_done:
        for i, trainer in enumerate(trainers):
            if trainer.ready and not trainer.is_set:
                if curr_sim == num_episodes:
                    break
                curr_sim += 1
                trainer.reset()
                with net_lock:
                    sleep(0.5)
                    if (curr_sim+1) % 300 == 0:
                        with agent.model.graph.as_default():
                            agent.model.save_model("weights" + str(curr_sim+1))
                    copy_model(agent.model, trainer.model) 
                trainer.episode = curr_sim
                trainer.is_set = True
            
            if trainer.done:
                simulation_replays.put(trainer.replay)
                trainer.ready = True

        
        if curr_sim == num_episodes and \
            all(trainer.done for trainer in trainers) and \
            agent.done:   
            for i, trainer in enumerate(trainers):
                if not trainer.ready:
                    simulation_replays.put(trainer.replay)
                    trainer.join()

            while not simulation_replays.not_empty or not agent.done:
                sleep(1)
            is_training_done = True
            agent.model
            agent.join()
            break
        sleep(1)

    print("DONE!")

def copy_model(agent_model, trainer_model):
    weights = agent_model.get_trainable_weights()
    trainer_model.set_trainable_weights(weights)

class ReinforcementAgent(Thread):

    def __init__(self, thread_id, params, net_lock, simulation_replays):
        Thread.__init__(self)

        self.id = thread_id
        self.name = "agent_thread"
        self.net_lock = net_lock
        self.ready = False
        self.done = False
        self.params = params
        self.gamma = self.params["gamma"]
        self.simulation_replays = simulation_replays
        self.ready = True
        self.model = FullMapCNN(params)

    def get_batch(self):
        states = []
        q_values = []
        actions = []
        for i in range(self.params["file_batch"]):
            replay = self.get_data(self.simulation_replays.get())

            states.extend(replay[0])
            actions.extend(replay[1])
            q_values.extend(replay[2])
        indices = np.array(range(len(states)))
        np.random.shuffle(indices)
        states = np.array(states)[indices]
        actions = np.array(actions)[indices]
        q_values = np.array(q_values)[indices]
        batch_size = int(self.params["batch_percentage"] * len(states))
        return states[:batch_size], actions[:batch_size], q_values[:batch_size]

    def get_data(self, replay):
        states = []
        actions = []
        rewards = []
        if len(replay) <= 3:
            return [], [], []
        for i in range(len(replay)-1):
            states.append(replay[i][0])
            actions.append(replay[i][1])
            rewards.append(replay[i][2])

        states.append(replay[-1][0])  
        states = np.array(states)
         
        qs = self.model.predict(states)
        q_next = qs[1:]
        qs = qs[:-1]
        for i in range(len(actions)):
            qs[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])
        
        return states[:-1], actions, qs

    def train(self):
        global is_training_done
        while not is_training_done:
            self.done = False

            states, _, q_values = self.get_batch()
            if len(states) > 0:
                a = 0
                with self.net_lock:  
                    a = self.model.evaluate(states, q_values)
                    self.model.train(states, q_values)
                    a = self.model.evaluate(states, q_values)
                self.done = True
                sleep(1)

    def run(self):
        self.train()

class ReinforcementTrainer(Thread):

    def __init__(self, thread_id, params):
        Thread.__init__(self)

        self.id = thread_id
        self.name = "trainer " + thread_id
        self.replay = []
        self.ready = True
        self.is_set = False
        self.done = False
        self.episode = 0
        self.epsilon = params["epsilon"]
        self.epsilon_decay = params["epsilon_decay"]
        self.params = params
        self.model = FullMapCNN(params)

    def train(self):
        
        position = (random.choice(range(self.params["num_rows"])),
                    random.choice(range(self.params["num_cols"])))
        robots = [position]
        goals = [(8, 8)]
        grid = worlds.empty_with_robots_and_objects(self.params, robots, goals)
        env = MultiagentEnv(self.params, grid)
        robots = range(len(env.robots))
        pomdp = POMDP(env, self.params)
        count = 0
        #env.render()
        _, R = pomdp.build_mdp(env)
        total_reward = 0
        while not env.done and count < env.params["max_iter"]: 
            for robot in robots:
                data = get_data_as_matrix(robot, env, pomdp, self.params)

                qs = self.model.predict(data)
                if random.random() < self.epsilon:
                    action, _, _ = pomdp.get_optimal_action_for_robot(robot)
                else:
                    action = np.argmax(qs[0])

                state = env.robots[robot].position
                next_state, obs = env.do_action(env.robots[robot], action) 
                reward = R[action, env.ravel_state(state), env.ravel_state(next_state)]
                self.replay.append((data[0], action, reward))
                total_reward += reward
                if env.done:
                    break     
                pomdp.propagate_obs(next_state, action, obs)
                _, R = pomdp.build_mdp(env)           
            
            if self.id == "t1":
                env.render()
            count += 1
            
        self.epsilon *= self.epsilon_decay
        np.set_printoptions(precision=3)
        print("Episode: ", self.episode, " epsilon: ", self.epsilon)
        if count > 0:
            print("steps: ", count, " reward: ", total_reward, " avg: ", total_reward/count, "\n")

    def reset(self):
        self.replay = []
        self.ready = True
        self.is_set = False
        self.done = False
        self.spawn_rates = []

    def run(self, **kwargs):
        global is_training_done 
        while not is_training_done:
            if self.ready and self.is_set:
                self.ready = False
                self.is_set = False
                self.train()
                self.done = True
            sleep(1)


      

if __name__ == "__main__":
    main()
