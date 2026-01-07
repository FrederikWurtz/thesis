import numpy as np
from master.configs.config_utils import load_config_file
from pprint import pprint


config = load_config_file()


if __name__ == "__main__":
    pprint(config)