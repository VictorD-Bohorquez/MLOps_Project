from classes.data_manager import Data_Manager
import argparse
import yaml
from sklearn.model_selection import train_test_split

def load_data(path):
    with open(path) as con:
        config = yaml.safe_load(con)
    data_manager = Data_Manager(path)
    data_manager.load_data()
    data_manager.split_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Loading process, needs a configuration file.')
    parser.add_argument('--config', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    load_data(path=args.config)