import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ActivityDictionary import *
from DataImporter import *


class SensorDataset(Dataset):
    def __init__(self, members_range=train_numbers):
        self.descriptors = []

        for person_num in members_range:
            for activity in activities_names:
                self.descriptors.append([person_num, activity])

    def __len__(self):
        return len(self.descriptors)

    def __getitem__(self, idx):
        current = self.descriptors[idx]
        _, label = current
        movement = self.get_input(DataImporter().get_data(current[0], current[1]))
        return torch.FloatTensor(movement), self.encode_label(label)

    def get_input(self, raw_movement):
        input = []
        for sensor in ['s1', 's2', 's3', 's4', 's5', 's6']:
            placeholder = raw_movement[sensor][0:-1:10][:50]
            input.extend(placeholder)
        return input

    def encode_label(self, label):
        activity_index = activities_names.index(label)
        return activity_index


train_dataloader = DataLoader(SensorDataset(train_numbers), batch_size=16)
test_dataloader = DataLoader(SensorDataset(range(54, 58)), batch_size=16)


class SensorNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, len(activities_names)),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = SensorNeuralNetwork()
