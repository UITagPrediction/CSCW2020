import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from loader import data_generator
from model import Net


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data loader
train_loader, valid_loader, test_loader = data_generator()

# Hyperparameters
num_epochs = 500
learning_rate = 0.001

model = Net(500,50,2)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# store all losses for visualisation
all_losses = []
all_losses2 = []

# train the model by batch
for epoch in range(num_epochs):
    total = 0
    correct = 0
    total_loss = 0
    for step, (batch_x, batch_y) in enumerate(train_loader):
        X = batch_x
        Y = batch_y.long()

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = model(X)
        loss = criterion(outputs, Y)
        all_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if (epoch % 50 == 0):
            _, predicted = torch.max(outputs, 1)
            # calculate and print accuracy
            total = total + predicted.size(0)
            correct = correct + sum(predicted.data.numpy() == Y.data.numpy())
            total_loss = total_loss + loss
            
    if (epoch % 50 == 0):
        print('Epoch [%d/%d], Loss: %.4f, Accuracy: %.2f %%'
              % (epoch + 1, num_epochs,
                 total_loss, 100 * correct/total))
        for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.6
    
    # Valid 
    for batch_x, batch_y in valid_loader:
        X = batch_x
        Y = batch_y.long()
    outputs = model(X)
    loss = criterion(outputs, Y)
    all_losses2.append(loss.item())
    _, predicted = torch.max(outputs, 1)
    total = predicted.size(0)
    correct = predicted.data.numpy() == Y.data.numpy()
    if (epoch % 50 == 0):
        print('Validing Accuracy: %.2f %%' % (100 * sum(correct)/total))

plt.figure()
plt.plot(all_losses)
plt.plot(all_losses2)
plt.show()


# Test
for batch_x, batch_y in test_loader:
    X = batch_x
    Y = batch_y.long()
outputs = model(X)
_, predicted = torch.max(outputs, 1)
total = predicted.size(0)
correct = predicted.data.numpy() == Y.data.numpy()
print('Testing Accuracy: %.2f %%' % (100 * sum(correct)/total))