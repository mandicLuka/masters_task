import os, copy, pickle
import numpy as np


## OFFLINE TRAINING
class SingleStateDataPipeline:
    def __init__(self, path, params, shuffle=False):
        self.path = path
        self.listdir = os.listdir(path)
        self.params = params
        self.file_batch = self.params["file_batch"]
        self.count = 0
        self.num_passed = 0
        self.shuffle = shuffle
        self.reset = False
        
    def get_next_batch(self):
        batch = []
        if self.count + self.file_batch >= len(self.listdir) - 1:
            batch = self.listdir[self.count:]
            self.num_passed += 1
            self.reset = True
        else:
            batch = self.listdir[self.count:self.count+self.file_batch]
        
        states = []
        next_states = []
        actions = []
        rewards = []
        returns = []
        for name in batch:     
            with open(os.path.join(self.path, name), "rb") as f:
                trajs, beliefs = pickle.load(f)
                for j in trajs.keys():
                    for step in range(len(trajs[j])):
                        state = trajs[j][step]
                        state["object_belief"] = beliefs[step]["object_belief"]
                        state["obstacle_belief"] = beliefs[step]["obstacle_belief"]
                        state["return"] = 5
                        state["robot_position"] = state["robot"]
                        actions.append(state["action"])
                        rewards.append(state["reward"])
                        returns.append(state["return"])
                        if bool(self.params["use_local_data"]) == True:
                            data = local_data_as_matrix(state, self.params)
                        else:
                            data = global_data_as_matrix(state, self.params)
                        if step < len(trajs[j]) - 1:
                            states.append(data)
                        if step > 0:
                            next_states.append(data)

        self.count += self.file_batch
        if self.reset:
            self.reset = False
            self.count = 0

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        returns = np.array(returns)
        next_states = np.array(next_states)

        if self.shuffle:
            indices = np.array(range(len(data)))
            np.random.shuffle(indices)
            states = states[indices]     
            actions = actions[indices]
            rewards = rewards[indices]
            returns = returns[indices]
            next_states = next_states[indices]
        return states, actions, rewards, returns, next_states


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


# OFFLINE TRAINING
def global_data_as_matrix(state, params):

    shape = (params["num_rows"], params["num_cols"])
    channels = params["data_channels"]
    data = np.zeros((*shape, 1))

    if "object_belief" in channels:
        data = np.concatenate((data, state["object_belief"].reshape(shape)[..., np.newaxis]), axis=-1)
    if "obstacle_belief" in channels:
        data = np.concatenate((data, state["obstacle_belief"].reshape(shape)[..., np.newaxis]), axis=-1)
    if "robot_position" in channels:
        robot_position = np.zeros((*shape, 1))
        pos = state["robot_position"]
        robot_position[pos[0], pos[1], 0] = 1
        data = np.concatenate((data, robot_position[..., np.newaxis]), axis=-1)

    if "other_robots" in channels:
        other_robots = np.zeros((*shape, 1))
        for r in state["other_robots"]:
            other_robots[r[0], r[1], 0] = 1
        data = np.concatenate((data, other_robots), axis=-1)
    
    if "goals" in channels:
        goals = np.zeros((*shape, 1))
        for g in [o for o in state["goals"]]:
            goals[g[0], g[1], 0] = 1
        data = np.concatenate((data, goals), axis=-1)
    
    if "global_goal_map" in channels:
        data = np.concatenate((data, get_goal_map(state["goals"], params)[..., np.newaxis]), axis=-1)
    
    if "single_goal_map" in channels:
        data = np.concatenate((data, get_goal_map([state["robot_goal"]], params)[..., np.newaxis]), axis=-1)

    if bool(params["use_coords"]) == True:
        data = add_coords_to_data(data, params)
    return data[:, :, 1:]
    

def local_data_as_matrix(state, params):
    p = copy.deepcopy(params)
    shape = (params["num_rows"], params["num_cols"])
    if "robot_position" in params["data_channels"]:
        p["data_channels"].remove("robot_position")
    data = global_data_as_matrix(state, params)
    pos = state["robot_position"]
    r = 2*params["local_data_radius"]
    temp = np.zeros((shape[0]+2*r, shape[1]+2*r, data.shape[-1]))
    if "obstacle_belief" in params["data_channels"]:
        temp[:, :, 1] = 1
    if bool(params["use_coords"]):
        temp[:, :, -2:] = -1
    
    temp[r:shape[0]+r, r:shape[1] + r, :] = data
    data = temp[r+pos[0]-r//2:r+pos[0]+r//2+1, 
                 r+pos[1]-r//2:r+pos[1]+r//2+1, :]
    return data


#ONLINE TRAINING
def global_data_as_matrix_on(robot, env, pomdp, params):
    state = dict()
    state["object_belief"] = pomdp.object_belief
    state["obstacle_belief"] = pomdp.obstacle_belief
    state["robot_position"] = env.robots[robot].position
    state["other_robots"] = []
    for r in env.get_all_robots_positions_excluding(robot):
        state["other_robots"].append(r)
    state["goals"] = []
    for g in [o.position for o in env.objects]:
        state["goals"].append(g)
    
    state["robot_goal"] = pomdp.env.objects[0].position
    
    return global_data_as_matrix(state, params)


def local_data_as_matrix_on(robot, env, pomdp, params):
    state = dict()
    state["object_belief"] = pomdp.object_belief
    state["obstacle_belief"] = pomdp.obstacle_belief
    state["robot_position"] = env.robots[robot].position
    state["other_robots"] = []
    for r in env.get_all_robots_positions_excluding(robot):
        state["other_robots"].append(r)
    state["goals"] = []
    for g in [o.position for o in env.objects]:
        state["goals"].append(g)
    
    state["robot_goal"] = pomdp.env.objects[0].position
    return local_data_as_matrix(state, params)

def get_goal_map(goals, params):
    shape = (params["num_rows"], params["num_cols"])
    delta = params["goal_map_decay"]
    size = params["goal_map_kernel"]
    goal_map = np.zeros(shape)
    
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            d = np.array([i, j]) - np.array([size//2, size//2])
            dist = np.linalg.norm(d)
            kernel[i, j] = np.exp(-delta*dist)

    for g in [o for o in goals]:
        s = size-1
        temp = np.zeros((shape[0]+2*s, shape[1]+2*s))
        temp[s:shape[0]+s, s:shape[1] + s] = goal_map
        temp[s+g[0]-size//2:s+g[0]+size//2+1, 
                 s+g[1]-size//2:s+g[1]+size//2+1] += kernel
        goal_map = temp[s:shape[0]+s, s:shape[1] + s]
    return goal_map