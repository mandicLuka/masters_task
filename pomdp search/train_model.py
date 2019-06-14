import tensorflow as tf
import numpy as np
from models import full_map_cnn
from utils import load_params

def main():
    params = load_params("dataset_params.yaml")
    model = full_map_cnn(params)
    



if __name__ == "__main__":
    main()