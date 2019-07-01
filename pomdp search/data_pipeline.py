import os, copy, pickle
import numpy as np


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

    tempi = np.vstack([i_channel[np.newaxis, ...]]*data.shape[0])
    tempj = np.vstack([j_channel[np.newaxis, ...]]*data.shape[0])
    data = np.concatenate((data, tempi[..., np.newaxis], \
                                 tempj[..., np.newaxis]), axis=-1)
    return data

def get_data_as_matrix(robot, env, pomdp, params):
    robot_position = np.zeros(env.shape)
    other_robots = np.zeros(env.shape)
    goals = np.zeros(env.shape)
    robot_position[env.robots[robot].position] = 1
    for r in env.get_all_robots_positions_excluding(robot):
        other_robots[r] = 1
    for g in [o.position for o in pomdp.env.objects]:
        goals[g] = 1
    # data = np.concatenate((pomdp.object_belief.reshape(env.shape)[..., np.newaxis],  \
    #                         pomdp.obstacle_belief.reshape(env.shape)[..., np.newaxis], \
    #                         robot_position[..., np.newaxis], \
    #                         other_robots[..., np.newaxis], \
    #                         goals[..., np.newaxis]), axis = -1)[np.newaxis, ...]
    data = np.concatenate((pomdp.object_belief.reshape(env.shape)[..., np.newaxis],  \
                           robot_position[..., np.newaxis], \
                           goals[..., np.newaxis]), axis = -1)[np.newaxis, ...]
    if bool(params["use_coords"]) == True:
        data = add_coords_to_data(data, params)
    return data