import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataImporter:

    def __init__(self):
        pass

    def get_filename(self, person_number, activity):
        # wristandthigh/._026_downstairs
        return f'wristandthigh/0{person_number}_{activity}.csv'

    def get_data_and_print(self, person_number, activity):
        path = self.get_filename(person_number, activity)
        data = pd.read_csv(path, names=['s1', 's2', 's3', 's4', 's5', 's6'])
        self.print_data_basics(data)
        self.draw_values(data)


    def print_data_basics(self, data):
        print(data.shape)
        print('\n')
        print(data.head())
        print('\n')
        print(data.info())
        print('\n')

    def draw_values(self, data):
        # plt.plot(range(len(data['s1'])), data['s1'])
        # plt.show()
        plt.subplot(2, 1, 1)
        plt.plot(range(500), data['s1'][:500], 'r')
        plt.plot(range(500), data['s2'][:500], 'b')
        plt.plot(range(500), data['s3'][:500], 'g')

        plt.subplot(2, 1, 2)
        plt.plot(range(500), data['s4'][:500], 'r')
        plt.plot(range(500), data['s5'][:500], 'g')
        plt.plot(range(500), data['s6'][:500], 'b')

        plt.show()
