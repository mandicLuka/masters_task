import gym
import numpy as np
from argparse import ArgumentParser
import sys, os
import gym
import random
from yaml import safe_load as load_config
from search_env.envs.search_env import Action

# TODO load maps from json
MAPS = {
    "prva":[
    "#######",
    "#R.#.G#",
    "#..#..#",
    "#..#..#",
    "#..#..#",
    "#.....#",
    "#######"
    ]
}

def create_dataset(path, rows, cols, num_envs, env_trajs, params):
    """
    :param path: path for data file. use separate folders for training and test data
    :param rows: grid rows
    :param cols: grid cols
    :param num_env: number of environments in the dataset (grids)
    :param env_trajs: number of trajectories per environment (different initial state, goal, initial belief)
    :param params: path to dataset parameters

    """

    params = load_params(params)

    # randomize seeds, set to previous value to determinize random numbers
    np.random.seed()
    random.seed()

    

    for env_i in range(num_envs):
        print ("Generating %d. environment"%(env_i))
        # grid domain object
        domain = gym.make("search_env:pomdp-v0", params=params)
        domain.generate_trajectories(env_trajs)

    print ("Done.")

def load_params(params_path: str):

    if not os.path.isfile(params_path):
        raise Exception("Could not find file " + params_path)

    with open(params_path) as config_file:
        params = load_config(config_file)
    return params

def main():
    parser = ArgumentParser(description='Generate search trajectories')
    parser.add_argument('--dataset_folder', type=str, default="Datasets", help='Directory for datasets')
    parser.add_argument('--params', type=str, default="dataset_params.yaml", help='File with dataset params')
    parser.add_argument('--train', type=int, default=10000, help='Number of training environments')
    parser.add_argument('--test', type=int, default=500, help='Number of test environments')
    parser.add_argument('--X', type=int, default=20, help='Num rows in grid')
    parser.add_argument('--Y', type=int, default=20, help='Num cols in grid')
    parser.add_argument('--trajs', type=int, default=5,
         help='Number of trajectories per environment in the training set. 5 by default.')

    args = parser.parse_args()
    dataset_folder = os.path.join(os.getcwd(), args.dataset_folder)
    params = os.path.join(os.getcwd(), args.params)
    if not os.path.isdir(dataset_folder): 
        os.mkdir(dataset_folder)

    # training data
    create_dataset(os.path.join(dataset_folder, '/train/'), rows=args.X, cols=args.Y, \
                    num_envs=args.train, env_trajs=args.trajs, params=params)

    # test data
    create_dataset(os.path.join(dataset_folder, '/test/'), rows=args.X, cols=args.Y, \
                    num_envs=args.test, env_trajs=1, params=params)

if __name__ == "__main__":
    main()
