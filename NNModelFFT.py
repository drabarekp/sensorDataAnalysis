import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ActivityDictionary import *
from DataImporter import *
import random
from scipy.fft import fft, fftfreq

from NNLearningUtils import test_loop


class SensorDatasetFFT(Dataset):
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
                    start = random.randrange(0, sensor_data_length - 2001)
                    stuff = self.get_input(raw_data, start)
                    self.inputs.append([torch.FloatTensor(stuff), self.encode_label(activity)])

        for idx in range(len(self.descriptors)):
            stuff = self.get_item_from_label(idx)
            self.inputs.append(stuff)
            if idx % 1000 == 0:
                print(idx)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

    def get_input(self, raw_movement, start):
        input = []
        for sensor in ['s1', 's2','s3','s4','s5','s6']:
            placeholder = raw_movement[sensor][start:start + 500:10]
            fft_placeholder = fft(np.array(placeholder))
            input.extend(np.abs(fft_placeholder))
            # input.extend(np.concatenate((np.abs(fft_placeholder), np.imag(fft_placeholder))))
        return input

    def encode_label(self, label):
        activity_index = activities_names.index(label)
        return activity_index


class SensorNeuralNetworkFFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(1200, 600),
        #     nn.ReLU(),
        #     nn.Linear(600, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, len(activities_names)),
        # )

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, len(activities_names)),
        )

    def forward(self, x):

        x = self.flatten(x)
        # lstm_out, _ = self.lstm(x)
        logits = self.linear_relu_stack(x)
        return logits

    def validate(self):
        validate_dataloader = DataLoader(SensorDatasetFFT(range(54, 58)), shuffle=True)
        matrix = self.this_test_loop(validate_dataloader, self, nn.CrossEntropyLoss())
        return matrix

    def this_test_loop(self, dataloader, model, loss_fn):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        matrix = np.zeros(shape=(9, 9), )
        print(matrix[0][0])
        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                matrix[pred.argmax(1)][y] += 1

        test_loss /= num_batches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return matrix
