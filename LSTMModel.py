import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader, Dataset

from ActivityDictionary import train_numbers, activities_names
from DataImporter import DataImporter


class LSTMSensorDataset(Dataset):
    def __init__(self, members_range=train_numbers):
        self.descriptors = []
        self.inputs = []
        for person_num in members_range:
            print(person_num)
            for activity in activities_names:
                raw_data = DataImporter().get_data(person_num, activity)
                sensor_data_length = len(raw_data['s1'])
                for _ in range(int(sensor_data_length / 100)):
                    start = random.randrange(0, sensor_data_length - 3001)
                    self.inputs.append([torch.FloatTensor(raw_data[start:start+3000:20].to_numpy()), self.encode_label(activity)])



    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

    def encode_label(self, label):
        activity_index = activities_names.index(label)
        return activity_index


def create_loaders(bs=128, jobs=0):
    train_dl = DataLoader(LSTMSensorDataset(), bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(LSTMSensorDataset(range(54, 58)), shuffle=False, num_workers=jobs)
    return train_dl, valid_dl


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t for t in (h0, c0)]


class CyclicLR(_LRScheduler):

    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]


def cosine(t_max, eta_min=0):
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2

    return scheduler

