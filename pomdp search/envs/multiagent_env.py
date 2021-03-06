import sys
from contextlib import closing
from six import StringIO
from gym import utils, Env
from gym.spaces import Discrete
from queue import Queue
import numpy as np
from enum import IntEnum
from collections import OrderedDict
from sklearn.cluster import KMeans
import pickle
import random

class Action(IntEnum):
    """
        Action enumeration. Environment supports 4 
        actions: UP, LEFT, RIGHT and DOWN
    """
    UP = 0
    LEFT = 1
    RIGHT = 2 
    DOWN = 3

class Robot():
    """
        Base Robot class represents robot on map. For future work
        to add different robots.
    """
    def __init__(self,  position = (0, 0)):
        self.x = position[0]
        self.y = position[1]
        self.position = position

class Object():
    """
        Base object class represents object on map. For future work
        to add different objects.
    """
    def __init__(self,  position = (0, 0)):
        self.x = position[0]
        self.y = position[1]
        self.position = position

class Obstacle():
    """
        Base obstacle class.
    """
    def __init__(self,  position = (0, 0)):
        self.x = position[0]
        self.y = position[1]
        self.position = position

### TYPES OF CELLS ###
FREE = 0
OBSTACLE = 1
ROBOT = 2
OBJECT = 3

# TODO load robots, objects and other stuff from config
class MultiagentEnv(Env):
    """
        Environment used for multi-agent search mission.
        The task is for N robots to find M objects on map.
    """
    def __init__(self, params, grid=None):
        """
            Can be created with given grid. If grid is None
            create random grid with obstacle probability
            given in params. Robots and goals are also randomly
            placed on map.
        """
        self.t = 0
        self.params = params
        self.num_rows = params['num_rows']
        self.num_cols = params['num_cols']
        if grid is None:
            self.grid = self.generate_random_grid(self.num_rows, self.num_cols, params["obstacle_prob"])
        else:
            self.grid = grid

        self.shape = self.grid.shape

        self.num_states = self.num_rows * self.num_cols
        self.num_actions = params["num_actions"]

        self.fields = [FREE, OBSTACLE, ROBOT, OBJECT]
        self.num_fields = len(self.fields)
        self.totwal_reward = 0

        self.observation_cells = ["left", "curr", "right", "front"]
        self.num_observation_cells = len(self.observation_cells)

        self.robots = []
        self.objects = []

        if grid is None:
            if isinstance(params["num_robots"], (list, )):
                self.num_robots = random.choice(range(params["num_robots"][0], params["num_robots"][1]+1))
            else:
                self.num_robots = params["num_robots"]
            if isinstance(params["num_objects"], (list, )):
                self.num_objects = random.choice(range(params["num_objects"][0], params["num_objects"][1]+1))
            else:
                self.num_objects = params["num_objects"]
            self.init_robots_objects()
        else:
            self.robots = self.find_on_map("robot")
            self.objects = self.find_on_map("object")
            self.num_robots = len(self.robots)
            self.num_objects = len(self.objects)

            

        self.done = False

    def init_env(self):
        self.__init__(self.params)

    def find_on_map(self, obj):
        """Gets and prints the spreadsheet's header columns

        Parameters
        ----------
        obj : "robot", "object" or "obstacle"
            type of object to find
       
        Returns
        -------
        list
            a list of all objects of type obj on map
        """
        objects = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if obj == "robot" and self.grid[i, j] == ROBOT:
                    objects.append(Robot((i, j)))
                if obj == "object" and self.grid[i, j] == OBJECT:
                    objects.append(Object((i, j)))
                if obj == "obstacle" and self.grid[i, j] == OBSTACLE:
                    objects.append(Obstacle((i, j)))
        return objects

    def init_robots_objects(self):
        """Randomly place robots and objects on map.

        This method ensures that every goal is reachable.
        Number of robots and objects is loaded from params.
        """
        free_states = np.nonzero((self.grid == FREE).flatten())[0]

        while True:
            free_spots = np.random.choice(free_states, size=self.num_robots+self.num_objects, replace=False)
            robot_start_states = [self.unravel_state(x) for x in free_spots[:self.num_robots]]
            object_states = [self.unravel_state(x) for x in free_spots[self.num_robots:]]

            all_goals_reachable = True
            for goal in object_states:
                continue_outer_loop = False
                for start in robot_start_states:
                    if self.path_exists(start, goal):
                        continue_outer_loop = True
                        break

                if continue_outer_loop:
                    continue
                all_goals_reachable = False
                break
                
            if all_goals_reachable:
                self.robots = []
                self.objects = []
                for robot in robot_start_states:
                    self.grid[robot] = ROBOT 
                    self.robots.append(Robot(robot))
                for obj in object_states:
                    self.grid[obj] = OBJECT
                    self.objects.append(Object(obj))
                return 

    @staticmethod
    def generate_random_grid(rows, cols, obstacle_prob=0.25):
        """Generate random grid with shape (rows, cols) 
        with obstacle probability obstacle_prob

        Parameters
        ----------
        rows : int
            number of rows in grid
        cols : int
            number of cols in grid
       
        Returns
        -------
        grid : numpy array((rows, cols))
        """
        grid = np.zeros((rows, cols), dtype='i') # all free

        rand_field = np.random.rand(rows, cols)
        grid[rand_field < obstacle_prob] = OBSTACLE
        return grid

    @staticmethod
    def generate_empty_grid(rows, cols):
        """Generate empty grid with shape (rows, cols)

        Parameters
        ----------
        rows : int
            number of rows in grid
        cols : int
            number of cols in grid
       
        Returns
        -------
        grid : numpy array((rows, cols))
        """
        return np.zeros((rows, cols), dtype='i')

    @staticmethod
    def transform_grid(grid):
        """Transforms grid to render style grid.

        Parameters
        ----------
        grid : numpy array((rows, cols))
       
        Returns
        -------
        transformed_grid : numpy array((rows, cols))
        """
        transformed_grid = np.zeros(grid.shape, dtype='c')
        transformed_grid[grid == FREE] = "."
        transformed_grid[grid == OBSTACLE] = "#"
        transformed_grid[grid == ROBOT] = "R"
        transformed_grid[grid == OBJECT] = "G"

        return transformed_grid

    def unravel_state(self, state):
        return np.unravel_index(state, self.shape)

    def ravel_state(self, state):
        return np.ravel_multi_index(state, self.shape)

    def do_action(self, robot, action):
        action = Action(action)
        state = robot.position
    
        obs = self.sample_observation(robot.position, action)
        if self.is_action_valid(robot.position, action):
            next_state = self.get_next_state_from_action(state, action)
            if self.is_object(next_state):
                for i, obj in enumerate(self.objects):
                    if obj.position == next_state:
                        del self.objects[i]
                        break         
                self.num_objects -= 1
                if self.num_objects == 0:
                    self.done = True
                    
            self.grid[next_state] = ROBOT
            self.grid[robot.position] = FREE
            robot.position = next_state
            return next_state, obs
        return state, obs

    def is_out_of_bounds(self, state):
        row, col = state
        if row < 0 or row > self.num_rows-1 or col < 0 or col > self.num_cols-1:
            return True
        return False

    def is_free(self, state):
        if self.is_out_of_bounds(state):
            return False
        return self.grid[state] == FREE

    def is_obstacle(self, state):
        if self.is_out_of_bounds(state):
            return False
        return self.grid[state] == OBSTACLE

    def is_robot(self, state):
        if self.is_out_of_bounds(state):
            return False
        return self.grid[state] == ROBOT

    def is_object(self, state):
        if self.is_out_of_bounds(state):
            return False
        return self.grid[state] == OBJECT

    def is_action_valid(self, state, action):
        next_state = self.get_next_state_from_action(state, action)
        return not self.is_obstacle(next_state) \
               and not self.is_robot(next_state) \
               and not self.is_out_of_bounds(next_state)        

    def path_exists(self, start, goal):
        closed = []
        q = []
        if not self.is_out_of_bounds(start) and not self.is_out_of_bounds(goal):
            q.append(start)

            while len(q) > 0:   
                state = q.pop()
                if state not in closed:
                    closed.append(state)
                if state == goal:
                    return True

                for action in self.get_valid_actions(state):
                    if not self.is_action_valid(state, action):
                        continue
                    next_state = self.get_next_state_from_action(state, action)
                    if next_state not in closed and next_state not in q:
                        q.append(next_state)
        
        return False

    def get_valid_actions(self, state):
        actions = []
        for action in Action:    
            if self.is_action_valid(state, action):
                actions.append(action)
        return actions

    def get_next_state_from_action(self, state, action):
        row, col = state
        if Action(action) == Action.UP:
            row = row-1     
        elif Action(action) == Action.LEFT:
            col = col-1
        elif Action(action) == Action.RIGHT:
            col = col+1
        elif Action(action) == Action.DOWN:
            row = row+1
        else:
            assert False, "Action not valid"
        return (row, col)

    def get_front_probs(self, state):
        probs = dict()
        if self.grid[state] == FREE or self.grid[state] == ROBOT:
            probs[FREE], probs[OBSTACLE], probs[OBJECT] = 0.8, 0.1, 0.1
        elif self.grid[state] == OBSTACLE:
            probs[FREE], probs[OBSTACLE], probs[OBJECT] = 0.1, 0.8, 0.1
        elif self.grid[state] == OBJECT:
            probs[FREE], probs[OBSTACLE], probs[OBJECT] = 0.1, 0.1, 0.8
        return probs

    def get_side_probs(self, state):
        probs = dict()
        if self.grid[state] == FREE or self.grid[state] == ROBOT:
            probs[FREE], probs[OBSTACLE], probs[OBJECT] = 0.6, 0.2, 0.2
        elif self.grid[state] == OBSTACLE:
            probs[FREE], probs[OBSTACLE], probs[OBJECT] = 0.2, 0.6, 0.2
        elif self.grid[state] == OBJECT:
            probs[FREE], probs[OBSTACLE], probs[OBJECT] = 0.2, 0.2, 0.6
        return probs

    def get_curr_probs(self, state):
        probs = dict()
        if self.grid[state] == FREE or self.grid[state] == ROBOT:
            probs[FREE], probs[OBSTACLE], probs[OBJECT] = 1, 0.0, 0.0
        elif self.grid[state] == OBSTACLE:
            probs[FREE], probs[OBSTACLE], probs[OBJECT] = 0.0, 1, 0.0
        elif self.grid[state] == OBJECT:
            probs[FREE], probs[OBSTACLE], probs[OBJECT] = 0.0, 0.0, 1
        return probs

    @staticmethod
    def get_probs_as_matrix(probs):
        key = list(probs.keys())[0] # take first key
        p_mat = np.zeros((len(probs), len(probs[key])))
        for i, obs_state in enumerate(probs.keys()):
            for j, cell_state in enumerate(probs[obs_state].keys()):
                 p_mat[i, j] = probs[obs_state][cell_state]
        return p_mat

    def sample_observation(self, state, action):
        observation = dict()
        action = Action(action)
        states = self.get_observation_states(state, action)
        probs = self.get_observation_probs(state, action)

        if not self.is_out_of_bounds(states["left"]):
            observation[states["left"]] = (np.random.choice(list(probs["left"].keys()), p=list(probs["left"].values())), "left")

        observation[states["curr"]] = (np.random.choice(list(probs["curr"].keys()), p=list(probs["curr"].values())), "curr")

        if not self.is_out_of_bounds(states["right"]):
            observation[states["right"]] = (np.random.choice(list(probs["right"].keys()), p=list(probs["right"].values())), "right")

        if not self.is_out_of_bounds(states["front"]): 
            observation[states["front"]] = (np.random.choice(list(probs["front"].keys()), p=list(probs["front"].values())), "front")
        return observation
    
    def get_observation_probs(self, state, action):      
        probs = OrderedDict(zip(self.observation_cells, [OrderedDict() for x in self.observation_cells]))
        states = self.get_observation_states(state, action)

        if not self.is_out_of_bounds(states["left"]):
            probs["left"] = self.get_side_probs(states["left"])
        else:
            probs["left"] = {FREE:0, OBSTACLE:1, OBJECT:0}

        probs["curr"] = self.get_curr_probs(states["curr"])

        if not self.is_out_of_bounds(states["right"]):
            probs["right"] = self.get_side_probs(states["right"])
        else:
            probs["right"] = {FREE:0, OBSTACLE:1, OBJECT:0}

        if not self.is_out_of_bounds(states["front"]):
            probs["front"] = self.get_front_probs(states["front"])
        else:
            probs["front"] = {FREE:0, OBSTACLE:1, OBJECT:0}
        return probs

    def get_observation_states(self, state, action):
        next_state = None
        if action == Action.UP:
            if self.is_action_valid(state, action):
                next_state = self.get_next_state_from_action(state, Action.UP)
            else:  
                next_state = state
            left = self.get_next_state_from_action(next_state, Action.LEFT)
            right = self.get_next_state_from_action(next_state, Action.RIGHT)
            front = self.get_next_state_from_action(next_state, Action.UP)

        elif action == Action.LEFT:
            if self.is_action_valid(state, action):
                next_state = self.get_next_state_from_action(state, Action.LEFT)
            else:  
                next_state = state
            left = self.get_next_state_from_action(next_state, Action.DOWN)
            right = self.get_next_state_from_action(next_state, Action.UP)
            front = self.get_next_state_from_action(next_state, Action.LEFT)

        elif action == Action.RIGHT:
            if self.is_action_valid(state, action):
                next_state = self.get_next_state_from_action(state, Action.RIGHT)
            else:  
                next_state = state
            left = self.get_next_state_from_action(next_state, Action.UP)
            right = self.get_next_state_from_action(next_state, Action.DOWN)
            front = self.get_next_state_from_action(next_state, Action.RIGHT)

        elif action == Action.DOWN:
            if self.is_action_valid(state, action):
                next_state = self.get_next_state_from_action(state, Action.DOWN)
            else:  
                next_state = state
            left = self.get_next_state_from_action(next_state, Action.RIGHT)
            right = self.get_next_state_from_action(next_state, Action.LEFT)
            front = self.get_next_state_from_action(next_state, Action.DOWN)
        else:
            assert False, "Action not valid"
        states = {"curr":next_state,
                  "left":left,
                  "right":right}

        # if action is invalid, front obs cannot be seen
        if state == next_state:
            states["front"] = (-1, -1)
        else:
            states["front"] = front
        return states

    def render(self, mode='human'):
        grid = self.transform_grid(self.grid)
        border = "".join(["# " for i in range(self.num_cols+2)])
        print(border)
        for i in range(self.num_rows):
            print("# " + "".join([x.decode('utf-8')+" " for x in grid[i]]) + "# ")
        print(border)


    def map_obs(self, obs):
        a = np.zeros(self.shape)
        for key in obs.keys():
            a[key] = obs[key][0]

    def get_all_robots_positions_excluding(self, robot):
        if isinstance(robot, int):
            return [x.position for i, x in enumerate(self.robots) if not i == robot]
        return [x.position for x in self.robots if not x.position == robot.position]
