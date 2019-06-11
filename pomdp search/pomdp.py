import mdptoolbox
import numpy as np
import mdptoolbox.example
import search_env.envs.search_env as env
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from copy import deepcopy

class POMDP:

    def __init__(self, environment, params):
        self.env = environment
        self.Q = None
        self.V = None    

        self.num_rows = self.env.num_rows
        self.num_cols = self.env.num_cols

        self.num_states = self.num_cols * self.num_rows
        self.shape = (self.num_rows, self.num_cols)
        self.num_actions = self.env.num_actions
        self.gamma = params["gamma"]
        self.params = params
        self.num_objects = self.env.num_objects

        # dict of dict (state x 2) measurementes for objects
        # and obstacles
        self.measurements = dict()
        for i in range(self.num_states):
            state = self.env.unravel_state(i)
            self.measurements[state] = {env.OBJECT: [], \
                                        env.OBSTACLE: []}
        
        self.object_belief = np.ones(self.num_rows * self.num_cols)*0.5
        self.obstacle_belief = np.ones(self.num_rows * self.num_cols)*0.5
        self.visited = np.zeros((self.num_rows * self.num_cols), dtype="bool")

    def solve(self):
        T, R = env.build_pomdp()
        V = self.compute_V(T, R, max_iter=10000)
        self.compute_Q(T, R, V)

    def compute_V(self, T, R, max_iter=10000):
        value_iteration = mdptoolbox.mdp.ValueIteration(T, R, self.gamma, max_iter=max_iter)
        value_iteration.run()
        self.V = value_iteration.V
        return self.V

    def compute_Q(self, T, R, V):
        self.Q = np.zeros([self.num_states, self.num_actions], 'f')
        for action in range(self.num_actions):
            R_action_state = np.multiply(T[action], R[action]).sum(1) 

        self.Q[:, action] = R_action_state + self.gamma * T[action].dot(V)
        return self.Q


    def build_pomdp(self): 
        T = np.zeros((self.num_actions, self.num_states, self.num_states), 'f')
        R = np.zeros((self.num_actions, self.num_states, self.num_states), 'f') 
  
        env.render()
        for state_num in range(self.num_states):
            for action_num, action in enumerate(env.Action):

                state = env.unravel_state(state_num)
                if env.is_out_of_bounds(state):
                    continue
                
                ##### REWARDS AND TRANSITIONS#####
                if env.is_action_valid(state, action):
                    next_state = env.get_next_state_from_action(state, action)
                    next_state_num = env.ravel_state(next_state)
                    if env.is_object(next_state):
                        R[action_num, state_num, next_state_num] += self.params["goal_reward"]
                    elif env.is_obstacle(next_state):
                        R[action_num, state_num, next_state_num] += self.params["obstacle_reward"]
                    elif env.is_robot(next_state):
                        R[action_num, state_num, next_state_num] += self.params["obstacle_reward"]
                else:
                    next_state = state
                    next_state_num = env.ravel_state(state)
                
                R[action_num, state_num, next_state_num] += self.params["step_reward"]
            
                ##### TRANSITIONS #####
                T[action_num, state_num, next_state_num] = 1.0
        return T, R

    def propagate_obs(self, state, action, obs):
        self.visited[self.env.ravel_state(state)] = True
        rel = self.get_obs_reliability(obs)
        for state in rel.keys():
            
            measurement = rel[state][env.OBJECT]
            self.measurements[state][env.OBJECT].append(measurement)
            measurement = rel[state][env.OBSTACLE]
            self.measurements[state][env.OBSTACLE].append(measurement)
        self.object_belief = self.get_belief(env.OBJECT)
        self.obstacle_belief = self.get_belief(env.OBSTACLE)
        X = np.zeros((self.num_states, 3))
        for i in range(self.num_states):
            if self.visited[i]:
                continue
            state = self.env.unravel_state(i)
            X[i] = np.array([state[0], state[1], 100*self.object_belief[i]])

        model = KMeans(n_clusters=8)
        y = model.fit_predict(X)
        a = model.cluster_centers_
        b = np.reshape(model.labels_, (1, -1))
        num_samples_in_cluters = np.zeros(self.params["num_clusters"])
        for i in range(self.params["num_clusters"]):
            temp = np.where(np.array(b) == i)
            num_samples_in_cluters[i] = temp[0].size
        plot_clusters(X, y, self.shape[0])

    def get_obs_reliability(self, obs):
        total_rel = dict()
        state_rel = 0
        for state in obs.keys():
            o, direction = obs[state]
            
            if direction == "left" or direction == "right":
                state_rel = 0.7
            elif direction == "front":
                state_rel = 0.8
            elif direction == "curr":
                state_rel = 1
            else:
                assert "Direction unknown"
            
            if state not in total_rel.keys():
                total_rel[state] = dict()

            if int(o) == env.OBJECT:
                total_rel[state][env.OBJECT] = state_rel
            else:
                total_rel[state][env.OBJECT] = 1 - state_rel
            if int(o) == env.OBSTACLE:
                total_rel[state][env.OBSTACLE] = state_rel
            else:
                total_rel[state][env.OBSTACLE] = 1 - state_rel
        return total_rel

    def get_belief(self, obj):
        belief = np.ones(self.num_states) * 0.5
        for i in range(self.num_states):
            state = self.env.unravel_state(i)
            mes = self.measurements[state][obj]
            if mes:
                a = 1 - (1 - np.array(self.measurements[state][obj])).prod()
                b = np.array(self.measurements[state][obj]).prod()
                if np.isclose(a , 1):
                    belief[i] = a
                elif np.isclose(b, 0):
                    belief[i] = b
                else:
                    belief[i] = (a+b)/2
        return belief



def plot_clusters(X, y, shape):
    x1 = deepcopy(X[:, 0])
    x2 = deepcopy([X[:, 1]])
    x1, x2 = x2, (shape-1)*np.ones_like(x1) - x1
    plt.scatter(x1, x2, c=y, marker='o')
    plt.show()