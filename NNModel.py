import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ActivityDictionary import *
from DataImporter import *
import random
from scipy.fft import fft, fftfreq

from NNLearningUtils import test_loop


class SensorDataset(Dataset):
    def __init__(self, members_range=train_numbers):
        self.descriptors = []
        self.inputs = []
        # self.inputs_from_person = 25
        for person_num in members_range:
            print(person_num)
            for activity in activities_names:
                raw_data = DataImporter().get_data(person_num, activity)
                sensor_data_length = len(raw_data['s1'])
                for _ in range(int(sensor_data_length / 100)):
                    start = random.randrange(0, sensor_data_length - 501)
                    stuff = self.get_input(raw_data, start)
                    self.inputs.append([torch.FloatTensor(stuff), self.encode_label(activity)])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

    def get_input(self, raw_movement, start):
        input = []
        for sensor in ['s1', 's2', 's3', 's4', 's5', 's6']:
            placeholder = raw_movement[sensor][start:start + 500:10]
            input.extend(placeholder)
        return input

    def encode_label(self, label):
        activity_index = activities_names.index(label)
        return activity_index


class SensorNeuralNetwork(nn.Module):
    def __init__(self, stack = nn.Sequential(
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, len(activities_names)),
        )
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        # self.lstm = nn.LSTM(3000, 300, 2)
        self.linear_relu_stack = stack

    def forward(self, x):
        x = self.flatten(x)
        # lstm_out, _ = self.lstm(x)
        logits = self.linear_relu_stack(x)
        return logits

    def validate(self):
        validate_dataloader = DataLoader(SensorDataset(range(54, 58)), shuffle=False)
        actual, predicted = test_loop(validate_dataloader, self, nn.CrossEntropyLoss())
        cm = confusion_matrix(actual, predicted)
        return cm
