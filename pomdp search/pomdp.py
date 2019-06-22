import mdptoolbox
import numpy as np
import mdptoolbox.example
import envs.multiagent_env as ma_env
from envs.multiagent_env import FREE, OBSTACLE, ROBOT, OBJECT
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from copy import deepcopy
from policies import greedy



class POMDP:
    def __init__(self, env, params):
        self.env = env
        self.Q = None
        self.V = None    

        self.num_rows = env.num_rows
        self.num_cols = env.num_cols

        self.num_states = self.num_cols * self.num_rows
        self.shape = (self.num_rows, self.num_cols)
        self.num_actions = self.env.num_actions
        self.gamma = params["gamma"]
        self.params = params
        self.num_objects = env.num_objects

        # dict of dict (state x 2) measurementes for objects
        # and obstacles
        self.measurements = dict()
        for i in range(self.num_states):
            self.measurements[i] = {OBJECT: [], \
                                        OBSTACLE: []}
        
        self.object_belief = np.ones(self.num_rows * self.num_cols)*0.5
        self.obstacle_belief = np.ones(self.num_rows * self.num_cols)*0.5
        self.visited = np.zeros((self.num_rows * self.num_cols), dtype="bool")
        for robot in env.robots:
            state = self.env.ravel_state(robot.position)
            self.object_belief[state] = 0
            self.obstacle_belief[state] = 0
            self.visited[state] = True
            self.measurements[state][OBJECT].append(0)
            self.measurements[state][OBSTACLE].append(0)

        self.clusters = np.zeros(self.params["num_clusters"])
        self.cluster_num_samples = np.zeros(self.params["num_clusters"])
        self.cluster_mean_belief = np.zeros(self.params["num_clusters"])

    def compute_V(self, T, R, max_iter=10):
        value_iteration = mdptoolbox.mdp.ValueIteration(T, R, self.gamma, max_iter=max_iter)
        value_iteration.run()
        V = np.array(value_iteration.V)
        obstacles = self.env.find_on_map("obstacle")
        for obstacle in obstacles:
            V[self.env.ravel_state(obstacle.position)] = 0
        return V, np.array(value_iteration.policy)

    def get_optimal_action_for_robot(self, robot):
        policy = greedy(self.env)
        robot_goal_env = self.build_policy_env(robot, int(policy[robot]))
        T, R = self.build_mdp(self.env)
        V, robot_action_policy = self.compute_V(T, R)
        # self.plot_V(V)
        # self.plot_V(robot_action_policy)
        # self.env.render()
        robot_position = self.env.robots[robot].position
        s = self.env.ravel_state(robot_position)
        action = int(robot_action_policy[s])
        
        next_V = 0
        if self.env.is_action_valid(robot_position, action):
            ns = self.env.ravel_state(self.env.get_next_state_from_action(robot_position, action))
            reward = R[action, s, ns]
            next_V = V[ns]
        else:
            reward = R[action, s, s]
            next_V = V[s]
        q = reward + self.gamma * next_V
        return action, reward, q, robot_goal_env

    def build_mdp(self, env): 
        T = np.zeros((self.num_actions, self.num_states, self.num_states), 'f')
        R = np.zeros((self.num_actions, self.num_states, self.num_states), 'f') 
  
        for state_num in range(self.num_states):
            for action_num, action in enumerate(ma_env.Action):
                state = env.unravel_state(state_num)

                ##### REWARDS #####
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
                if self.visited[next_state_num]:
                    R[action_num, state_num, next_state_num] += self.params["visited_reward"]

                ##### TRANSITIONS #####
                T[action_num, state_num, next_state_num] = 1.0
        return T, R

    def propagate_obs(self, curr_state, action, obs):
        self.visited[self.env.ravel_state(curr_state)] = True
        rel = self.get_obs_reliability(obs)
        for state in rel.keys():
            i_state = self.env.ravel_state(state)
            measurement = rel[state][OBJECT]
            self.measurements[i_state][OBJECT].append(measurement)
            measurement = rel[state][OBSTACLE]
            self.measurements[i_state][OBSTACLE].append(measurement)
        self.object_belief = self.get_belief(OBJECT)
        self.obstacle_belief = self.get_belief(OBSTACLE)

        # X = np.zeros((self.num_states-np.where(self.visited == True)[0].size, 3))
        # count = 0
        # for i in range(self.num_states):
        #     if self.visited[i]:
        #         continue
        #     state = self.env.unravel_state(i)
        #     X[count] = np.array([state[0], state[1], self.params["belief_scale"]*self.object_belief[i]])
        #     count += 1

        # model = KMeans(n_clusters=10)
        # model.fit(X)
        # self.clusters = np.array(model.cluster_centers_[:, :2], dtype='i')
        # b = np.reshape(model.labels_, (1, -1))
        # for i in range(self.params["num_clusters"]):
        #     temp = np.where(b == i)
        #     self.cluster_num_samples[i] = temp[0].size
        #     self.cluster_mean_belief[i] = (X[temp[1], 2].sum())  \
        #        / (self.params["belief_scale"] * self.cluster_num_samples[i])
        
        #plot_clusters(X, y, self.shape[0])

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

            if int(o) == OBJECT:
                total_rel[state][OBJECT] = state_rel
            else:
                total_rel[state][OBJECT] = 1 - state_rel
            if int(o) == OBSTACLE:
                total_rel[state][OBSTACLE] = state_rel
            else:
                total_rel[state][OBSTACLE] = 1 - state_rel
        return total_rel

    def get_belief(self, obj):
        belief = np.ones(self.num_states) * 0.5
        for i in range(self.num_states):
            mes = self.measurements[i][obj]
            if mes:
                a = 1 - (1 - np.array(self.measurements[i][obj])).prod()
                b = np.array(self.measurements[i][obj]).prod()
                if np.isclose(a , 1):
                    belief[i] = 0 # dunno what is better, 1 or 0
                elif np.isclose(b, 0):
                    belief[i] = 0
                else:
                    belief[i] = (a+b)/2
        return belief

    def build_policy_env(self, robot, goal):
        grid = deepcopy(self.env.grid)
        grid[grid == OBJECT] = FREE
        grid[self.env.objects[goal].position] = OBJECT
        grid[grid == ROBOT] = OBSTACLE
        grid[self.env.robots[robot].position] = ROBOT
        return ma_env.MultiagentEnv(self.params, grid)

    def plot_V(self, V):
        np.set_printoptions(precision=1)
        for i in range(self.num_rows):
            print(V[i*self.num_cols:(i+1)*self.num_cols])


def plot_clusters(X, y, shape):
    x1 = deepcopy(X[:, 0])
    x2 = deepcopy([X[:, 1]])
    x1, x2 = x2, (shape-1)*np.ones_like(x1) - x1
    plt.scatter(x1, x2, c=y, marker='o')
    plt.show()