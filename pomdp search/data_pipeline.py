import os, copy, pickle
import numpy as np


## OFFLINE TRAINING
class DataPipeline:
    def __init__(self, path, params):
        self.path = path
        self.listdir = os.listdir(path)
        self.params = params
        self.file_batch = self.params["file_batch"]
        self.count = 0
        self.num_passed = 0
        self.reset = False
        
    def get_next_batch(self):
        batch = []
        if self.count + self.file_batch >= len(self.listdir) - 1:
            batch = self.listdir[self.count:]
            self.num_passed += 1
            self.reset = True
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
                         "optimal_actions": []}
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
                        step_optimal_action = np.zeros(1)

                        step_robots[state["robot"]] = 1
                        step_actions = state["action"]
                        step_rewards = state["reward"]
                        step_optimal_action = state["optimal_action"] 
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
                            current_state["optimal_actions"].append(step_optimal_action)
                        if step > 0:
                            pass
                            next_state["robots"].append(step_robots)
                            next_state["other_robots"].append(step_other_robots)
                            next_state["object_beliefs"].append(step_objects)
                            next_state["obstacle_beliefs"].append(step_obstacles)
                            next_state["goals"].append(step_goals)
                            next_state["actions"].append(step_actions)
                            next_state["rewards"].append(step_rewards)
                            next_state["optimal_actions"].append(step_optimal_action)

        self.count += self.file_batch
        if self.reset:
            self.reset = False
            self.count = 0
        return current_state, next_state


def add_coords_to_data(data, params):
    shape = (params["num_rows"], params["num_cols"])
    i_channel = np.zeros(shape)
    j_channel = np.zeros(shape)
    rangi = np.linspace(0, 1, shape[0])*2 - 1
    rangj = np.linspace(0, 1, shape[1])*2 - 1
    for i in range(shape[1]):
        i_channel[:, i] = rangi
    for j in range(shape[0]):
        j_channel[j] = rangj

    data = np.concatenate((data, i_channel[..., np.newaxis], \
                                 j_channel[..., np.newaxis]), axis=-1)
    return data


#ONLINE TRAINING
def get_data_as_matrix(robot, env, pomdp, params):
    channels = params["data_channels"]
    data = np.zeros((*env.shape, 1))
    if "object_belief" in channels:
        data = np.concatenate((data, pomdp.object_belief.reshape(env.shape)[..., np.newaxis]), axis=-1)
    if "obstacle_belief" in channels:
        data = np.concatenate((data, pomdp.obstacle_belief.reshape(env.shape)[..., np.newaxis]), axis=-1)
    if "robot_position" in channels:
        robot_position = np.zeros((*env.shape, 1))
        pos = env.robots[robot].position
        robot_position[pos[0], pos[1], 0] = 1
        data = np.concatenate((data, robot_position[..., np.newaxis]), axis=-1)

    if "other_robots" in channels:
        other_robots = np.zeros((*env.shape, 1))
        for r in env.get_all_robots_positions_excluding(robot):
            other_robots[r[0], r[1], 0] = 1
        data = np.concatenate((data, other_robots), axis=-1)
    
    if "goals" in channels:
        goals = np.zeros((*env.shape, 1))
        for g in [o.position for o in pomdp.env.objects]:
            goals[g[0], g[1], 0] = 1
        data = np.concatenate((data, goals), axis=-1)
    
    if "global_goal_map" in channels:
        data = np.concatenate((data, get_goal_map(env, params)[..., np.newaxis]), axis=-1)
    
    if "single_goal_map" in channels:
        data = np.concatenate((data, get_goal_map(pomdp.env, params)[..., np.newaxis]), axis=-1)

    if bool(params["use_coords"]) == True:
        data = add_coords_to_data(data, params)
    return data[:, :, 1:]

def get_local_data_as_matrix(robot, env, pomdp, params):
    p = copy.deepcopy(params)
    if "robot_position" in params["data_channels"]:
        params["data_channels"].remove("robot_position")
    data = get_data_as_matrix(robot, env, pomdp, p)
    t = data
    pos = env.robots[robot].position
    r = 2*params["local_data_radius"]
    temp = np.zeros((env.shape[0]+2*r, env.shape[1]+2*r, data.shape[-1]))
    if "obstacle_belief" in params["data_channels"]:
        temp[:, :, 1] = 1
    if bool(params["use_coords"]):
        temp[:, :, -2:] = -1
    
    temp[r:env.shape[0]+r, r:env.shape[1] + r, :] = data
    data = temp[r+pos[0]-r//2:r+pos[0]+r//2+1, 
                 r+pos[1]-r//2:r+pos[1]+r//2+1, :]
    return data

def get_goal_map(env, params):
    delta = params["goal_map_decay"]
    size = params["goal_map_kernel"]
    goal_map = np.zeros(env.shape)
    
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            d = np.array([i, j]) - np.array([size//2, size//2])
            dist = np.linalg.norm(d)
            kernel[i, j] = np.exp(-delta*dist)

    for g in [o.position for o in env.objects]:
        s = size-1
        temp = np.zeros((env.shape[0]+2*s, env.shape[1]+2*s))
        temp[s:env.shape[0]+s, s:env.shape[1] + s] = goal_map
        temp[s+g[0]-size//2:s+g[0]+size//2+1, 
                 s+g[1]-size//2:s+g[1]+size//2+1] += kernel
        goal_map = temp[s:env.shape[0]+s, s:env.shape[1] + s]
    return goal_map