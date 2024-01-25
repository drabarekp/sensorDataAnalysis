import torch
from torch import nn
from torch.utils.data import DataLoader

from ActivityDictionary import train_numbers
from NNLearningUtils import train_loop, test_loop
from NNModel import SensorNeuralNetwork, SensorDataset
from NNModelFFT import SensorNeuralNetworkFFT, SensorDatasetFFT

# cuda
print("CUDA:" + str(torch.cuda.is_available()))

if torch.cuda.is_available():
    torch.device('cuda')
else:
    torch.device('cpu')
# ------

model = SensorNeuralNetworkFFT()
# model.load_state_dict(torch.load("nn_models/n7"))

train_dataloader = DataLoader(SensorDatasetFFT(train_numbers), batch_size=64, shuffle=True)
test_dataloader = DataLoader(SensorDatasetFFT(range(54, 58)), batch_size=64, shuffle=True)

loss_fn = nn.CrossEntropyLoss()

learning_rate = 2e-4

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 1000
for t in range(epochs):
    print(f"Epoch {t + 1}-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

    if t % 5 == 0:
        torch.save(model.state_dict(), "nn_models/n_10.pt")

print("Done!")
