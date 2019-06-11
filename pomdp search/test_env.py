import gym
import numpy as np


params = dict()
params["num_rows"] = 5
params["num_cols"] = 5
params["num_robots"] = 1
params["num_objects"] = 1
params["num_actions"] = 4
params["step_reward"] = -1
params["goal_reward"] = 30
params["obstacle_reward"] = -10
params["gamma"] = 0.98

num_states = params["num_rows"] * params["num_cols"]

grid = np.zeros((params["num_rows"], params["num_cols"]))
grid[2, 0] = 1
grid[0, 0] = 2
grid[0, 4] = 3
domain = gym.make("search_env:pomdp-v0", params=params, grid=grid)
domain.render()


# 0-free; 1-obst; 3-goal
left = 0
right = 1
curr = 0
front = 0
observation = left * 27 + curr * 9 + right * 3 + front



belief = np.ones(num_states) / (num_states) 
# bprime = domain.pomdp.propagate_act(belief, 0)
# domain.pomdp.plot_belief(bprime)

belief = domain.pomdp.propagate_obs(belief, 2, 0)
belief = domain.pomdp.propagate_obs(belief, 2, 1)
belief = domain.pomdp.propagate_obs(belief, 2, 2)
belief = domain.pomdp.propagate_obs(belief, 2, 3)
domain.render()

print ("Done.")