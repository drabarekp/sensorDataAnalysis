import torch
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import ActivityDictionary
from NNModel import SensorNeuralNetwork
from NNModelFFT import SensorNeuralNetworkFFT

print("CUDA:" + str(torch.cuda.is_available()))

if torch.cuda.is_available():
    torch.device('cuda')
else:
    torch.device('cpu')
# ------

model = SensorNeuralNetwork()
model.load_state_dict(torch.load("nn_models/n_05.pt"))
matrix = model.validate()

disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=ActivityDictionary.activities_names)
disp.plot(values_format='.0f')
plt.show()
