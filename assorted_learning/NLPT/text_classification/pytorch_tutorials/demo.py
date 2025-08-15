import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(MultiLayerPerceptron, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output


input = torch.randn(2, 5)
model = MultiLayerPerceptron(5, 3)
model(input)

list(model.parameters())





