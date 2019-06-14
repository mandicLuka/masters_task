from yaml import safe_load as load_config
import os 




def load_params(params_path):

    if not os.path.isfile(params_path):
        raise Exception("Could not find file " + params_path)

    with open(params_path) as config_file:
        params = load_config(config_file)
    return params