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

TEST_DIR = os.path.join(os.getcwd(), "Datasets/test")

def main():
    global TEST_DIR
    params = load_params("dataset_params.yaml")
    if "test_folder" in params.keys():
        TEST_DIR = os.path.join(os.getcwd(), "Datasets", params["test_folder"])
    
 
    model = full_map_cnn(params)
    model.compile(tf.train.AdamOptimizer(), loss='mse', metrics=[action_acc, "mae"])
    model.load_weights("greedy_weights")

    online_test(model, params)



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

        next_data = np.concatenate((np.array(next_state["object_beliefs"])[..., np.newaxis],  \
                            np.array(next_state["obstacle_beliefs"])[..., np.newaxis], \
                            np.array(next_state["robots"])[..., np.newaxis], \
                            np.array(next_state["other_robots"])[..., np.newaxis], \
                            np.array(next_state["goals"])[..., np.newaxis]), axis = -1)
        

        next_Q = model.predict(next_data)
        argmax_Q = np.argmax(next_Q, axis=-1)
        labels = copy.deepcopy(next_Q)
        indices = (range(argmax_Q.size), argmax_Q)
        labels[indices] = cache["rewards"] + params["gamma"] * next_Q[indices]
        print(model.predict(data))
        print(cache["actions"])
        print(model.evaluate(data, labels))
        

class DataPipeline:

    def __init__(self, path, params):
        self.path = path
        self.listdir = os.listdir(path)
        self.params = params
        self.file_batch = self.params["file_batch"]
        self.count = 0
        self.num_passed = 0

    def get_next_batch(self):
        batch = []
        if self.count + self.file_batch >= len(self.listdir) - 1:
            batch = self.listdir[self.count:]
            self.num_passed += 1
            self.count = 0
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
                            next_state["robots"].append(step_robots)
                            next_state["other_robots"].append(step_other_robots)
                            next_state["object_beliefs"].append(step_objects)
                            next_state["obstacle_beliefs"].append(step_obstacles)
                            next_state["goals"].append(step_goals)
                            next_state["actions"].append(step_actions)
                            next_state["rewards"].append(step_rewards)
                            next_state["qs"].append(step_q)

        self.count += self.file_batch

        return current_state, next_state



if __name__ == "__main__":
    main()