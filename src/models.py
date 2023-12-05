import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU(), nn.Dropout(dropout_rate)]
        for i in range(len(hidden_layers) - 1):
            layers += [nn.Linear(hidden_layers[i], hidden_layers[i + 1]), nn.ReLU(), nn.Dropout(dropout_rate)]
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def setup_model(train_data, hidden_layers, dropout_rate):
    """
    Sets up the neural network model.

    :param train_data: DataFrame containing training data.
    :param hidden_layers: List of integers defining the hidden layers.
    :param dropout_rate: Dropout rate for regularization.
    :return: Initialized model.
    """
    input_size = train_data.shape[1] - 1
    model = MLP(input_size, hidden_layers, dropout_rate)
    return model