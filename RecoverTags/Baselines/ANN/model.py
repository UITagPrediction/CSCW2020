import torch.nn as nn
import torch
# define the number of neurons for input layer, hidden layer and output layer
# define learning rate and number of epoch on training
input_size = 500
hidden_size = 50
num_classes = 2

# Neural Network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = x.view (-1)
        x = x.view(-1,50)
        out = self.fc1(x)
        # out = out.view(-1, 500)
        out = out.view(-1, 10, 50)
        out = torch.max(out, 1)[0]
        out = self.relu(out)
        out = self.fc2(out)
        return out