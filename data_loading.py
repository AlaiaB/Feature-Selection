import pandas as pd
import json

class DataLoader:
    def __init__(self, data_path, json_path):
        self.data_path = data_path
        self.json_path = json_path

    def load_data(self):
        self.data = pd.read_csv(self.data_path)

    def load_json(self):
        with open(self.json_path) as json_file:
            self.json_data = json.load(json_file)
