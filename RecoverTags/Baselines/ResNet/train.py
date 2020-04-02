import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from loader import data_generator
from model import initialize_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Recovering Missing Semantics')
parser.add_argument('--tag', type=str, default='blue',
                    help='model to train (default: blue)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-4)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--pretrain', action='store_false',
                    help='use pretrain network (default: true)')

args = parser.parse_args()
torch.manual_seed(args.seed)
print(args)

# data loader
train_loader, valid_loader, test_loader = data_generator(args.tag)

# Hyperparameters
epochs = args.epochs
learning_rate = args.lr
num_classes = 2
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=args.pretrain)
model = model.to(device)
# print(model)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

optimizer = getattr(optim, args.optim)(params_to_update, lr=learning_rate, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

######################    
# train the model #
######################
def train_model():
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    print('train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc

######################    
# valid the model #
######################
def valid_model():
    model.eval() # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = running_corrects.double() / len(valid_loader.dataset)
    print('valid Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc

######################    
# test the model #
######################
def test_model():
    model.load_state_dict(torch.load('backup/checkpoint.pt'))
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)
    print('test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


if __name__ == "__main__":
    train_loss_history = []
    valid_loss_history = []
    train_acc_history = []
    valid_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)

        train_loss, train_acc = train_model()
        valid_loss, valid_acc = valid_model()

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        train_acc_history.append(train_acc)
        valid_acc_history.append(valid_acc)

        if valid_acc>best_acc:
            best_acc = valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    print('Best val Acc: {:4f}'.format(best_acc))
    if not os.path.exists('backup'):
        os.mkdir('backup')
    torch.save(best_model_wts, 'backup/checkpoint.pt')
    
    # visualise loss diagram and accuracy diagram
    plt.figure(1)
    plt.plot(train_loss_history)
    plt.plot(valid_loss_history)
    maxposs = valid_acc_history.index(max(valid_acc_history))+1 
    # plt.axvline(maxposs, linestyle='--', color='r')
    plt.gca().legend(('Train','Validation', 'Early Stopping Checkpoint'))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.figure(2)
    plt.plot(train_acc_history)
    plt.plot(valid_acc_history)
    maxposs = valid_acc_history.index(max(valid_acc_history))+1 
    # plt.axvline(maxposs, linestyle='--', color='r')
    plt.gca().legend(('Train','Validation','Early Stopping Checkpoint'))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy: %')
    plt.show()

    test_model()
