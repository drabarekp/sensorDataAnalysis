import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import seaborn as sb

class DataImporter:

    def __init__(self):
        pass

    def get_filename(self, person_number, activity):
        return f'wristandthigh/0{person_number}_{activity}.csv'

    def get_data_and_print(self, person_number, activity):
        path = self.get_filename(person_number, activity)
        data = pd.read_csv(path, names=['s1', 's2', 's3', 's4', 's5', 's6'])
        print(f"---------------PERSON:{person_number}, ACTIVITY={activity}---------------")
        self.print_data_basics(data)
        self.draw_values(data, range(500), f"sensor values for person {person_number} while {activity}")

        # fourier-transform
        used_records_num = len(data['s1'])
        fft_data = pd.DataFrame(fft(data), columns=['s1', 's2', 's3', 's4', 's5', 's6'])
        self.draw_fft_values(np.abs(fft_data), f"Fourier transformed sensor data for person {person_number} while {activity}")



    def print_data_basics(self, data):
        print(data.head())
        print(data.info())
        print("MEAN")
        print(data.mean())
        print("STD DEV:")
        print(data.std())
        print('\n')

    def draw_values(self, data, _range, title):
        # plt.plot(range(len(data['s1'])), data['s1'])
        # plt.show()

        plt.subplot(2, 1, 1)
        plt.title(title)
        plt.plot(_range, data['s1'][:len(_range)], 'r')
        plt.plot(_range, data['s2'][:len(_range)], 'b')
        plt.plot(_range, data['s3'][:len(_range)], 'g')

        plt.subplot(2, 1, 2)
        plt.plot(_range, data['s4'][:len(_range)], 'r')
        plt.plot(_range, data['s5'][:len(_range)], 'g')
        plt.plot(_range, data['s6'][:len(_range)], 'b')

        plt.show()

        sb.heatmap(data.corr(), cmap="Blues", annot=True)
        plt.show()

    def draw_fft_values(self, data, title):
        N = len(data['s1'])
        x = fftfreq(N, 1)[:N//2]

        plt.subplot(2, 1, 1)
        plt.title(title)
        plt.plot(x, data['s1'][:N//2], 'r')
        plt.plot(x, data['s2'][:N//2], 'b')
        plt.plot(x, data['s3'][:N//2], 'g')

        plt.subplot(2, 1, 2)
        plt.plot(x, data['s4'][:N//2], 'r')
        plt.plot(x, data['s5'][:N//2], 'g')
        plt.plot(x, data['s6'][:N//2], 'b')

        plt.show()
