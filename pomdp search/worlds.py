from envs.multiagent_env import *
import numpy as np

def empty_with_robots_and_objects(params, robots, goals):
    shape = (params["num_rows"], params["num_cols"]) 
    grid = np.ones(shape) * FREE
    for robot in robots:
        grid[robot] = ROBOT
    for goal in goals:
        grid[goal] = OBJECT
    return grid
