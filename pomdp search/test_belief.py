from pomdp import POMDP
import numpy as np

params = dict()
params['num_rows'] = 1
params['num_cols'] = 2
params['num_actions'] = 3
params["gamma"] = 0.9

num_states = params['num_rows'] * params['num_cols']
num_actions = params['num_actions']

# state: left = 0; right = 1;
# action: listen = 0; left = 1; right = 2
# obs: left = 0; left = 1;

T = np.zeros((num_actions, num_states, num_states))
Z = np.zeros((num_actions, num_states, 2))
R = np.zeros((num_actions, num_states, num_states))

# a, s, s'
# T[0,0,0] = 1
# T[0,0,1] = 0
# T[0,1,0] = 0
# T[0,1,1] = 1
# T[1,0,0] = 1
# T[1,0,1] = 0
# T[1,1,0] = 1
# T[1,1,1] = 0
# T[2,0,0] = 0
# T[2,0,1] = 1
# T[2,1,0] = 0
# T[2,1,1] = 1

# R[0,0,0] = -1
# R[0,0,1] = -1
# R[0,1,0] = -1
# R[0,1,1] = -1
# R[1,0,0] = -100
# R[1,0,1] = -100
# R[1,1,0] = -100
# R[1,1,1] = -100
# R[2,0,0] = 100
# R[2,0,1] = 100
# R[2,1,0] = 100
# R[2,1,1] = 100

# Z[0,0,0] = 0.15
# Z[0,0,1] = 0.85
# Z[0,1,0] = 0.85
# Z[0,1,1] = 0.15
# Z[1,0,0] = 0
# Z[1,0,1] = 0
# Z[1,1,0] = 0
# Z[1,1,1] = 0
# Z[2,0,0] = 0
# Z[2,0,1] = 0
# Z[2,1,0] = 0
# Z[2,1,1] = 0

# belief = np.array([0.5, 0.5])
# pdp = pomdp.POMDP(T, R, Z, params)

num_rows = 5
num_cols = 5
num_states = num_rows * num_cols

