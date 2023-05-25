import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Preprocessing:
    def __init__(self, data):
        self.data = data

    def convert_to_factor(self, cols):
        le = LabelEncoder()
        for col in cols:
            self.data[col] = le.fit_transform(self.data[col])

    def filter_data(self, conditions):
        for condition in conditions:
            self.data = self.data[condition]

    def split_data(self, test_size=0.2):
        train, test = train_test_split(self.data, test_size=test_size)
        return train, test

    def get_wave(self, wave):
        return self.data[self.data['Ola'] == wave]

    def split_by_wave(self):
        waves = self.data['Ola'].unique()
        return {wave: self.get_wave(wave) for wave in waves}
    
    def sample_by_time(self, sample_type='first'):
        if sample_type == 'first':
            return self.data.groupby('REGISTRO').first().reset_index()
        elif sample_type == 'last':
            return self.data.groupby('REGISTRO').last().reset_index()
        elif sample_type == 'random':
            return self.data.groupby('REGISTRO').sample(n=1).reset_index()
        else:
            raise ValueError("Invalid sample_type. Expected 'first', 'last', or 'random'.")

    def sample_all_by_time(self):
        return {'first': self.sample_by_registro('first'),
                'last': self.sample_by_registro('last'),
                'random': self.sample_by_registro('random')}
