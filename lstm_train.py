import torch
from torch import nn
from torch.utils.data import DataLoader

from ActivityDictionary import train_numbers
from LSTMModel import LSTMModel, create_loaders, CyclicLR, cosine
from torch.nn import functional as F
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

model = LSTMModel(6, 60, 2, 9)
train_l, test_l = create_loaders(bs=256)


iterations_per_epoch = len(train_l)
lr = 0.005
n_epochs = 20000
best_acc = 0
patience, trials = 100, 0
criterion = nn.CrossEntropyLoss()
opt = torch.optim.RMSprop(model.parameters(), lr=lr)
sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/100))


for epoch in range(1, n_epochs + 1):

    for i, (x_batch, y_batch) in enumerate(train_l):
        model.train()
        sched.step()
        opt.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        opt.step()

    model.eval()
    correct, total = 0, 0

    predicted = []
    actual = []
    for x_val, y_val in test_l:
        out = model(x_val)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_val.size(0)
        correct += (preds == y_val).sum().item()

    acc = correct / total

    print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')

    if epoch % 20 == 0:
        torch.save(model.state_dict(), 'best_8.pth')
        print(f'Epoch {epoch} model saved with accuracy: {acc:2.2%}')

# best lr = 0.00005 len = 500/10 79.8%
# best_1 lr = 0.001 len = 700/10 85.57%
# best_2 lr = 0.001 len = 700/10 89.99%
# best_3 lr = 0.001 len = 700/10 ~~85%
# best_4 lr = 0.001 len = 400/10 ~~85%
# best_5 lr = 0.005 len = 700/10 ~~88%
# best_6 lr = 0.005 len=700:20 ~~88% NOT SAVED! 60
# best_6 lr = 0.005 len=1400:20 ~~88% NOT SAVED! 60hidden_dim
# best_6 lr = 0.005 len=3000:50 ~~86% NOT 40hidden_dim
# best_7 lr = 0.005 len=3000:20 ~~??% 60hidden_dim 90.96