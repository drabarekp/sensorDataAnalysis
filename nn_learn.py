import torch
from torch import nn

from NNLearningUtils import train_loop, test_loop
from NNModel import SensorNeuralNetwork, test_dataloader, train_dataloader

loss_fn = nn.CrossEntropyLoss()
model = SensorNeuralNetwork()
learning_rate = 1e-2

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 100
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
