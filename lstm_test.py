import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader

from ActivityDictionary import train_numbers, activities_names
from LSTMModel import LSTMModel, create_loaders, CyclicLR, cosine, LSTMSensorDataset
from torch.nn import functional as F
from NNLearningUtils import train_loop, test_loop
from NNModel import SensorNeuralNetwork, SensorDataset
from NNModelFFT import SensorNeuralNetworkFFT, SensorDatasetFFT
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# cuda
print("CUDA:" + str(torch.cuda.is_available()))

if torch.cuda.is_available():
    torch.device('cuda')
else:
    torch.device('cpu')
# ------

model = LSTMModel(6, 60, 2, 9)
model.load_state_dict(torch.load("best_4.pth"))
test_l = DataLoader(LSTMSensorDataset(range(54, 58)), shuffle=False, num_workers=0)
total, correct = 0, 0
predicted = []
actual = []

for x_val, y_val in test_l:
    out = model(x_val)
    preds = F.log_softmax(out, dim=1).argmax(dim=1)
    total += y_val.size(0)
    correct += (preds == y_val).sum().item()
    predicted.append(preds.item())
    actual.append(y_val.item())
    cm = confusion_matrix(predicted, actual)

acc = correct / total

cm = confusion_matrix(predicted, actual)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=activities_names)
disp.plot()
plt.show()

print(acc)