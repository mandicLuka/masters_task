import numpy as np


def greedy(env):
    robots = [np.array(robot.position) for robot in env.robots]
    objects = [np.array(obj.position) for obj in env.objects]
    
    distances = np.zeros((len(robots), len(objects)))
    for i, r in enumerate(robots):
        for j, o in enumerate(objects):
            distances[i, j] = np.linalg.norm(r-o, 1)

    min_goal = np.argmin(distances, axis=1)
    return np.array(min_goal, dtype='i')